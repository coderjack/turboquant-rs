// File-based storage for TurboQuant compressed vectors (Lloyd-Max variant)
//
// File layout:
//   [IndexHeader: 16 bytes]
//   [Metadata: dim(4) + bits(1) = 5 bytes]
//   [Vector data: N × bytes_per_vector]
//
// Each vector is serialized as:
//   [original_norm: f32 LE, 4 bytes]
//   [residual_norm: f32 LE, 4 bytes]
//   [packed_indices: ceil(dim × bits / 8) bytes]
//   [residual_signs: ceil(dim/8) bytes]

use crate::compression::lloyd_max::ScalarQuantVector;
use crate::compression::qjl::BitVector;
use crate::turboquant::LloydMaxVector;
use crate::TurboError;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER_SIZE: usize = 16; // 4 magic + 4 version + 4 num_vectors + 4 bytes_per_vector
const METADATA_SIZE: usize = 5; // 4 dim + 1 bits

#[derive(Debug, Clone, Copy)]
struct IndexHeader {
    magic: [u8; 4],
    version: u32,
    num_vectors: u32,
    bytes_per_vector: u32,
}

impl IndexHeader {
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.num_vectors.to_le_bytes());
        buf[12..16].copy_from_slice(&self.bytes_per_vector.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        Self {
            magic,
            version: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            num_vectors: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            bytes_per_vector: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        }
    }
}

/// File-based storage for Lloyd-Max compressed vectors.
pub struct VectorStorage {
    file: File,
    path: PathBuf,
    header: IndexHeader,
    dim: usize,
    bits: u8,
}

impl VectorStorage {
    fn compute_bpv(dim: usize, bits: u8) -> u32 {
        let index_bytes = (dim * bits as usize + 7) / 8;
        let sign_bytes = (dim + 7) / 8;
        // original_norm(4) + residual_norm(4) + packed_indices + residual_signs
        (8 + index_bytes + sign_bytes) as u32
    }

    /// Create a new storage file.
    pub fn create(path: &Path, dim: usize, bits: u8) -> Result<Self, TurboError> {
        let bytes_per_vector = Self::compute_bpv(dim, bits);
        let header = IndexHeader {
            magic: *b"TQLM", // TurboQuant Lloyd-Max
            version: 1,
            num_vectors: 0,
            bytes_per_vector,
        };

        let mut file = File::create(path)?;
        file.write_all(&header.to_bytes())?;
        file.write_all(&(dim as u32).to_le_bytes())?;
        file.write_all(&[bits])?;
        file.flush()?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
            bits,
        })
    }

    /// Open an existing storage file.
    pub fn open(path: &Path) -> Result<Self, TurboError> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut buf)?;
        let header = IndexHeader::from_bytes(&buf);

        if &header.magic != b"TQLM" {
            return Err(TurboError::Storage(
                "invalid magic for TurboQuant Lloyd-Max file".into(),
            ));
        }

        let mut meta = [0u8; METADATA_SIZE];
        file.read_exact(&mut meta)?;
        let dim = u32::from_le_bytes(meta[0..4].try_into().unwrap()) as usize;
        let bits = meta[4];

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
            bits,
        })
    }

    fn data_offset(&self) -> usize {
        HEADER_SIZE + METADATA_SIZE
    }

    fn serialize(&self, lv: &LloydMaxVector) -> Vec<u8> {
        let bpv = self.header.bytes_per_vector as usize;
        let mut buf = Vec::with_capacity(bpv);
        buf.extend_from_slice(&lv.original_norm.to_le_bytes());
        buf.extend_from_slice(&lv.residual_norm.to_le_bytes());
        buf.extend_from_slice(&lv.scalar.data);
        buf.extend_from_slice(&lv.residual_signs.0);
        buf
    }

    fn deserialize(&self, data: &[u8]) -> LloydMaxVector {
        let original_norm = f32::from_le_bytes(data[0..4].try_into().unwrap());
        let residual_norm = f32::from_le_bytes(data[4..8].try_into().unwrap());

        let index_bytes = (self.dim * self.bits as usize + 7) / 8;
        let sign_bytes = (self.dim + 7) / 8;

        let mut offset = 8;
        let packed_indices = data[offset..offset + index_bytes].to_vec();
        offset += index_bytes;
        let residual_signs = data[offset..offset + sign_bytes].to_vec();

        LloydMaxVector {
            scalar: ScalarQuantVector {
                data: packed_indices,
                dim: self.dim,
                bits: self.bits,
            },
            residual_signs: BitVector(residual_signs),
            residual_norm,
            original_norm,
        }
    }

    /// Append a compressed vector.
    pub fn append(&mut self, lv: &LloydMaxVector) -> Result<(), TurboError> {
        let data = self.serialize(lv);
        let expected = self.header.bytes_per_vector as usize;
        if data.len() != expected {
            return Err(TurboError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&data)?;

        self.header.num_vectors += 1;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;

        Ok(())
    }

    /// Get the vector at a given index.
    pub fn get(&self, idx: usize) -> LloydMaxVector {
        let bpv = self.header.bytes_per_vector as usize;
        let offset = self.data_offset() + idx * bpv;

        let mut file = File::open(&self.path).expect("failed to open file for reading");
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        let mut buf = vec![0u8; bpv];
        file.read_exact(&mut buf).unwrap();

        self.deserialize(&buf)
    }

    /// Get all vectors.
    pub fn get_all(&self) -> Vec<LloydMaxVector> {
        let n = self.header.num_vectors as usize;
        let bpv = self.header.bytes_per_vector as usize;

        let mut file = File::open(&self.path).expect("failed to open file for reading");
        file.seek(SeekFrom::Start(self.data_offset() as u64))
            .unwrap();

        let mut all = Vec::with_capacity(n);
        for _ in 0..n {
            let mut buf = vec![0u8; bpv];
            file.read_exact(&mut buf).unwrap();
            all.push(self.deserialize(&buf));
        }
        all
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.header.num_vectors as usize
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.header.num_vectors == 0
    }

    /// Bytes per vector.
    pub fn bytes_per_vector(&self) -> usize {
        self.header.bytes_per_vector as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turboquant::LloydMaxCompressor;
    use tempfile::tempdir;

    #[test]
    fn test_storage_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.tqlm");
        let dim = 32;

        let comp = LloydMaxCompressor::new(dim, 42, 99, 4);
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        let lv = comp.compress(&v);

        {
            let mut storage = VectorStorage::create(&path, dim, 4).unwrap();
            storage.append(&lv).unwrap();
            storage.append(&lv).unwrap();
            assert_eq!(storage.len(), 2);
        }

        {
            let storage = VectorStorage::open(&path).unwrap();
            assert_eq!(storage.len(), 2);
            let loaded = storage.get(0);
            assert_eq!(loaded.scalar.data, lv.scalar.data);
            assert_eq!(loaded.residual_signs.0, lv.residual_signs.0);
            assert_eq!(loaded.residual_norm, lv.residual_norm);
            assert_eq!(loaded.original_norm, lv.original_norm);
        }
    }

    #[test]
    fn test_get_all() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.tqlm");
        let dim = 16;

        let comp = LloydMaxCompressor::new(dim, 10, 20, 3);

        let mut storage = VectorStorage::create(&path, dim, 3).unwrap();
        for i in 0..5 {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            storage.append(&comp.compress(&v)).unwrap();
        }

        let all = storage.get_all();
        assert_eq!(all.len(), 5);
    }
}
