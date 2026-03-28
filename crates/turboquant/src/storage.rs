// Memory-mapped binary file management for packed vectors

use crate::compression::{polarquant::PolarVector, qjl::BitVector};
use crate::TurboError;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER_SIZE: usize = 16; // 4 magic + 4 version + 4 num_vectors + 4 bytes_per_vector

/// Header for binary index files.
#[derive(Debug, Clone, Copy)]
pub struct IndexHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub num_vectors: u32,
    pub bytes_per_vector: u32,
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

// ---------------------------------------------------------------------------
// MmapBitVectors -- file-based storage for QJL BitVectors
// ---------------------------------------------------------------------------

/// File-based storage for packed bit vectors (QJL).
pub struct MmapBitVectors {
    file: File,
    path: PathBuf,
    header: IndexHeader,
    #[allow(dead_code)]
    dim: usize,
}

impl MmapBitVectors {
    /// Create a new storage file.
    pub fn create(path: &Path, dim: usize) -> Result<Self, TurboError> {
        let bytes_per_vector = ((dim + 7) / 8) as u32;
        let header = IndexHeader {
            magic: *b"TQJL",
            version: 1,
            num_vectors: 0,
            bytes_per_vector,
        };

        let mut file = File::create(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
        })
    }

    /// Open an existing storage file.
    pub fn open(path: &Path) -> Result<Self, TurboError> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut buf)?;
        let header = IndexHeader::from_bytes(&buf);

        if &header.magic != b"TQJL" {
            return Err(TurboError::Storage("invalid magic for QJL file".into()));
        }

        let dim = header.bytes_per_vector as usize * 8; // approximate; may be slightly larger than real dim

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
        })
    }

    /// Append a BitVector.
    pub fn append(&mut self, bv: &BitVector) -> Result<(), TurboError> {
        let expected = self.header.bytes_per_vector as usize;
        if bv.0.len() != expected {
            return Err(TurboError::DimensionMismatch {
                expected,
                got: bv.0.len(),
            });
        }

        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&bv.0)?;

        self.header.num_vectors += 1;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;

        Ok(())
    }

    /// Get the BitVector at a given index.
    pub fn get(&self, idx: usize) -> BitVector {
        let bpv = self.header.bytes_per_vector as usize;
        let offset = HEADER_SIZE + idx * bpv;

        let mut file = File::open(&self.path).expect("failed to open file for reading");
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        let mut buf = vec![0u8; bpv];
        file.read_exact(&mut buf).unwrap();

        BitVector(buf)
    }

    /// Get all BitVectors.
    pub fn get_all(&self) -> Vec<BitVector> {
        let n = self.header.num_vectors as usize;
        let bpv = self.header.bytes_per_vector as usize;

        let mut file = File::open(&self.path).expect("failed to open file for reading");
        file.seek(SeekFrom::Start(HEADER_SIZE as u64)).unwrap();

        let mut all = Vec::with_capacity(n);
        for _ in 0..n {
            let mut buf = vec![0u8; bpv];
            file.read_exact(&mut buf).unwrap();
            all.push(BitVector(buf));
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

// ---------------------------------------------------------------------------
// MmapPolarVectors -- file-based storage for PolarQuant vectors
// ---------------------------------------------------------------------------

/// File-based storage for PolarQuant vectors.
pub struct MmapPolarVectors {
    file: File,
    path: PathBuf,
    header: IndexHeader,
    dim: usize,
    angle_bits: u8,
}

impl MmapPolarVectors {
    /// Compute bytes per vector for polar storage.
    fn compute_bpv(dim: usize, angle_bits: u8) -> u32 {
        let num_pairs = (dim + 1) / 2;
        let angle_bytes = if angle_bits == 4 {
            (num_pairs + 1) / 2
        } else {
            num_pairs
        };
        let radii_bytes = num_pairs;
        (angle_bytes + radii_bytes) as u32
    }

    /// Create a new storage file.
    pub fn create(path: &Path, dim: usize, angle_bits: u8) -> Result<Self, TurboError> {
        let bytes_per_vector = Self::compute_bpv(dim, angle_bits);
        let header = IndexHeader {
            magic: *b"TQPL",
            version: 1,
            num_vectors: 0,
            bytes_per_vector,
        };

        let mut file = File::create(path)?;
        file.write_all(&header.to_bytes())?;

        // Write extra metadata: dim (4 bytes) + angle_bits (1 byte)
        file.write_all(&(dim as u32).to_le_bytes())?;
        file.write_all(&[angle_bits])?;
        file.flush()?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
            angle_bits,
        })
    }

    /// Open an existing storage file.
    pub fn open(path: &Path) -> Result<Self, TurboError> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut buf)?;
        let header = IndexHeader::from_bytes(&buf);

        if &header.magic != b"TQPL" {
            return Err(TurboError::Storage("invalid magic for PolarQuant file".into()));
        }

        // Read extra metadata
        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf) as usize;

        let mut ab_buf = [0u8; 1];
        file.read_exact(&mut ab_buf)?;
        let angle_bits = ab_buf[0];

        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            dim,
            angle_bits,
        })
    }

    /// Data offset (after header + metadata).
    fn data_offset(&self) -> usize {
        HEADER_SIZE + 5 // 4 bytes dim + 1 byte angle_bits
    }

    /// Serialize a PolarVector to bytes (angles ++ radii).
    fn serialize_pv(pv: &PolarVector) -> Vec<u8> {
        let mut buf = Vec::with_capacity(pv.angles.len() + pv.radii.len());
        buf.extend_from_slice(&pv.angles);
        buf.extend_from_slice(&pv.radii);
        buf
    }

    /// Deserialize bytes into a PolarVector.
    fn deserialize_pv(&self, data: &[u8]) -> PolarVector {
        let num_pairs = (self.dim + 1) / 2;
        let angle_bytes = if self.angle_bits == 4 {
            (num_pairs + 1) / 2
        } else {
            num_pairs
        };

        PolarVector {
            angles: data[..angle_bytes].to_vec(),
            radii: data[angle_bytes..].to_vec(),
            dim: self.dim,
            angle_bits: self.angle_bits,
        }
    }

    /// Append a PolarVector.
    pub fn append(&mut self, pv: &PolarVector) -> Result<(), TurboError> {
        let data = Self::serialize_pv(pv);
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

    /// Get the PolarVector at a given index.
    pub fn get(&self, idx: usize) -> PolarVector {
        let bpv = self.header.bytes_per_vector as usize;
        let offset = self.data_offset() + idx * bpv;

        let mut file = File::open(&self.path).expect("failed to open file for reading");
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        let mut buf = vec![0u8; bpv];
        file.read_exact(&mut buf).unwrap();

        self.deserialize_pv(&buf)
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.header.num_vectors as usize
    }

    /// Whether the storage is empty.
    pub fn is_empty(&self) -> bool {
        self.header.num_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::{polarquant::PolarQuantCompressor, qjl::QjlCompressor};
    use tempfile::tempdir;

    #[test]
    fn test_bitvector_storage_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.qjl");
        let dim = 32;

        let qjl = QjlCompressor::new(dim, 42);
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        let bv = qjl.compress(&v);

        {
            let mut storage = MmapBitVectors::create(&path, dim).unwrap();
            storage.append(&bv).unwrap();
            storage.append(&bv).unwrap();
            assert_eq!(storage.len(), 2);
        }

        {
            let storage = MmapBitVectors::open(&path).unwrap();
            assert_eq!(storage.len(), 2);
            let loaded = storage.get(0);
            assert_eq!(loaded.0, bv.0);

            let all = storage.get_all();
            assert_eq!(all.len(), 2);
            assert_eq!(all[0].0, bv.0);
            assert_eq!(all[1].0, bv.0);
        }
    }

    #[test]
    fn test_polar_storage_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.polar");
        let dim = 32;

        let comp = PolarQuantCompressor::new(dim, 42, 4, 8);
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        let pv = comp.compress(&v);

        {
            let mut storage = MmapPolarVectors::create(&path, dim, 4).unwrap();
            storage.append(&pv).unwrap();
            assert_eq!(storage.len(), 1);
        }

        {
            let storage = MmapPolarVectors::open(&path).unwrap();
            assert_eq!(storage.len(), 1);
            let loaded = storage.get(0);
            assert_eq!(loaded.angles, pv.angles);
            assert_eq!(loaded.radii, pv.radii);
            assert_eq!(loaded.dim, pv.dim);
        }
    }
}
