use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use turboquant::TurboIndex;

use crate::document::{Document, DocumentType, MemoryStats, RecallResult, SyncStats};
use crate::embedder::Embedder;
use crate::ranker::{rank, RankingConfig};
use crate::AgentMemoryError;

const INDEX_SUBDIR: &str = "turbo_index";
const DOCUMENTS_FILE: &str = "documents.json";
const HASHES_FILE: &str = "file_hashes.json";

/// Persistent cross-session memory. Stores documents on disk with a TurboIndex
/// for semantic recall. Supports incremental sync of file-backed documents.
pub struct PersistentMemory {
    embedder: Arc<dyn Embedder>,
    index: TurboIndex,
    documents: Vec<Document>,
    storage_dir: PathBuf,
    file_hashes: HashMap<String, String>,
    next_id: u64,
}

impl PersistentMemory {
    /// Open or create a persistent memory store at the given directory.
    pub fn open(storage_dir: &Path, embedder: Arc<dyn Embedder>) -> Result<Self, AgentMemoryError> {
        std::fs::create_dir_all(storage_dir)?;

        let index_path = storage_dir.join(INDEX_SUBDIR);
        let documents_path = storage_dir.join(DOCUMENTS_FILE);
        let hashes_path = storage_dir.join(HASHES_FILE);

        let documents: Vec<Document> = if documents_path.exists() {
            let data = std::fs::read_to_string(&documents_path)?;
            serde_json::from_str(&data)
                .map_err(|e| AgentMemoryError::Serialization(e.to_string()))?
        } else {
            Vec::new()
        };

        let file_hashes: HashMap<String, String> = if hashes_path.exists() {
            let data = std::fs::read_to_string(&hashes_path)?;
            serde_json::from_str(&data)
                .map_err(|e| AgentMemoryError::Serialization(e.to_string()))?
        } else {
            HashMap::new()
        };

        let next_id = documents.iter().map(|d| d.id).max().unwrap_or(0) + 1;

        let index = if index_path.exists() {
            TurboIndex::open(&index_path)?
        } else {
            TurboIndex::create(&index_path, embedder.dim(), 42, 99)?
        };

        Ok(Self {
            embedder,
            index,
            documents,
            storage_dir: storage_dir.to_path_buf(),
            file_hashes,
            next_id,
        })
    }

    /// Ingest a single document. Returns the assigned document ID.
    pub fn ingest(
        &mut self,
        content: &str,
        doc_type: DocumentType,
        source_path: Option<&str>,
    ) -> Result<u64, AgentMemoryError> {
        let id = self.next_id;
        self.next_id += 1;

        let mut doc = Document::new(id, doc_type, content.to_string());
        // Rough token estimate: ~4 chars per token
        doc.token_count = content.len() / 4;
        doc.source_path = source_path.map(|s| s.to_string());

        let embedding = self.embedder.embed(content);
        self.index.insert(id, &embedding)?;

        if let Some(path) = source_path {
            let hash = sha256_file_content(content);
            self.file_hashes.insert(path.to_string(), hash);
        }

        self.documents.push(doc);
        self.save()?;

        Ok(id)
    }

    /// Ingest all files matching a glob pattern. Returns the number of files ingested.
    pub fn ingest_glob(
        &mut self,
        pattern: &str,
        doc_type: DocumentType,
    ) -> Result<usize, AgentMemoryError> {
        let paths: Vec<PathBuf> = glob::glob(pattern)
            .map_err(|e| AgentMemoryError::Embedder(format!("invalid glob pattern: {e}")))?
            .filter_map(|r| r.ok())
            .filter(|p| p.is_file())
            .collect();

        let mut count = 0;
        for path in paths {
            let path_str = path.to_string_lossy().to_string();

            // Skip if already ingested with same hash.
            if let Some(existing_hash) = self.file_hashes.get(&path_str) {
                let content = std::fs::read_to_string(&path)?;
                let hash = sha256_file_content(&content);
                if &hash == existing_hash {
                    continue;
                }
            }

            let content = std::fs::read_to_string(&path)?;
            self.ingest(&content, doc_type.clone(), Some(&path_str))?;
            count += 1;
        }

        Ok(count)
    }

    /// Incremental sync: re-index changed files, remove deleted files.
    pub fn sync(&mut self) -> Result<SyncStats, AgentMemoryError> {
        let mut stats = SyncStats {
            added: 0,
            updated: 0,
            removed: 0,
            unchanged: 0,
        };

        // Collect documents with source_paths to check.
        let docs_with_paths: Vec<(u64, String)> = self
            .documents
            .iter()
            .filter_map(|d| d.source_path.as_ref().map(|p| (d.id, p.clone())))
            .collect();

        let mut to_remove = Vec::new();
        let mut to_update = Vec::new();

        for (id, path) in &docs_with_paths {
            let file_path = Path::new(path);
            if !file_path.exists() {
                to_remove.push(*id);
            } else {
                let content = std::fs::read_to_string(file_path)?;
                let new_hash = sha256_file_content(&content);
                let old_hash = self.file_hashes.get(path.as_str());

                if old_hash.map_or(true, |h| h != &new_hash) {
                    to_update.push((*id, path.clone(), content, new_hash));
                } else {
                    stats.unchanged += 1;
                }
            }
        }

        // Remove deleted files.
        for id in &to_remove {
            self.index.delete(*id)?;
            self.documents.retain(|d| d.id != *id);
            // Remove file hash for the source path.
            if let Some(path) = docs_with_paths.iter().find(|(did, _)| did == id) {
                self.file_hashes.remove(&path.1);
            }
            stats.removed += 1;
        }

        // Update changed files.
        for (id, path, content, new_hash) in to_update {
            // Remove old entry.
            self.index.delete(id)?;
            self.documents.retain(|d| d.id != id);

            // Re-ingest with new content (gets a new ID).
            let doc_type = DocumentType::Memory; // default; could be smarter
            self.ingest(&content, doc_type, Some(&path))?;
            self.file_hashes.insert(path, new_hash);
            stats.updated += 1;
        }

        self.save()?;
        Ok(stats)
    }

    /// Recall relevant prior knowledge.
    pub fn recall(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RecallResult>, AgentMemoryError> {
        self.recall_typed(query, top_k, &[])
    }

    /// Recall with optional document type filter.
    pub fn recall_typed(
        &self,
        query: &str,
        top_k: usize,
        types: &[DocumentType],
    ) -> Result<Vec<RecallResult>, AgentMemoryError> {
        if self.documents.is_empty() {
            return Ok(vec![]);
        }

        let query_embedding = self.embedder.embed(query);
        let search_results = self.index.search(&query_embedding, 50);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let config = RankingConfig {
            relevance_weight: 0.7,
            recency_weight: 0.3,
            always_keep_recent: 0,
        };

        let ranked = rank(&search_results, &self.documents, now, &config);

        let mut results: Vec<RecallResult> = ranked
            .into_iter()
            .filter_map(|r| {
                let doc = self.documents.iter().find(|d| d.id == r.id)?;
                // Apply type filter if specified.
                if !types.is_empty() && !types.contains(&doc.doc_type) {
                    return None;
                }
                Some(RecallResult {
                    document: doc.clone(),
                    relevance_score: r.relevance_score,
                    recency_score: r.recency_score,
                    combined_score: r.combined_score,
                })
            })
            .collect();

        results.truncate(top_k);
        Ok(results)
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let mut by_type = HashMap::new();
        for doc in &self.documents {
            *by_type.entry(doc.doc_type.clone()).or_insert(0) += 1;
        }

        let index_size = std::fs::metadata(self.storage_dir.join(INDEX_SUBDIR))
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        MemoryStats {
            total_documents: self.documents.len(),
            total_tokens: self.documents.iter().map(|d| d.token_count).sum(),
            index_size_bytes: index_size,
            by_type,
        }
    }

    fn save(&self) -> Result<(), AgentMemoryError> {
        let docs_json = serde_json::to_string_pretty(&self.documents)
            .map_err(|e| AgentMemoryError::Serialization(e.to_string()))?;
        std::fs::write(self.storage_dir.join(DOCUMENTS_FILE), docs_json)?;

        let hashes_json = serde_json::to_string_pretty(&self.file_hashes)
            .map_err(|e| AgentMemoryError::Serialization(e.to_string()))?;
        std::fs::write(self.storage_dir.join(HASHES_FILE), hashes_json)?;

        Ok(())
    }
}

fn sha256_file_content(content: &str) -> String {
    let hash = Sha256::digest(content.as_bytes());
    format!("{:x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::MockEmbedder;
    use tempfile::tempdir;

    fn make_persistent(dir: &Path) -> PersistentMemory {
        let embedder = Arc::new(MockEmbedder::new(32));
        PersistentMemory::open(dir, embedder).unwrap()
    }

    #[test]
    fn test_ingest_and_recall() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");
        let mut mem = make_persistent(&storage);

        mem.ingest("Rust ownership and borrowing rules", DocumentType::Memory, None)
            .unwrap();
        mem.ingest("Python garbage collection internals", DocumentType::Memory, None)
            .unwrap();
        mem.ingest("Rust lifetime annotations guide", DocumentType::Memory, None)
            .unwrap();

        let results = mem.recall("Rust memory management", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ingest_file_and_sync_update() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");

        // Create a file to ingest.
        let file_path = dir.path().join("test_doc.md");
        std::fs::write(&file_path, "original content about algorithms").unwrap();

        let mut mem = make_persistent(&storage);
        let file_str = file_path.to_string_lossy().to_string();
        mem.ingest(
            "original content about algorithms",
            DocumentType::Plan,
            Some(&file_str),
        )
        .unwrap();

        assert_eq!(mem.documents.len(), 1);

        // Modify the file.
        std::fs::write(&file_path, "updated content about data structures").unwrap();

        let stats = mem.sync().unwrap();
        assert_eq!(stats.updated, 1);
        assert_eq!(stats.removed, 0);

        // After sync, the document content should reflect the update.
        let updated_doc = mem
            .documents
            .iter()
            .find(|d| d.source_path.as_deref() == Some(&file_str));
        assert!(updated_doc.is_some());
        assert!(updated_doc.unwrap().content.contains("data structures"));
    }

    #[test]
    fn test_sync_removes_deleted_files() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");

        let file_path = dir.path().join("ephemeral.txt");
        std::fs::write(&file_path, "temporary content").unwrap();

        let mut mem = make_persistent(&storage);
        let file_str = file_path.to_string_lossy().to_string();
        mem.ingest("temporary content", DocumentType::Memory, Some(&file_str))
            .unwrap();

        assert_eq!(mem.documents.len(), 1);

        // Delete the file.
        std::fs::remove_file(&file_path).unwrap();

        let stats = mem.sync().unwrap();
        assert_eq!(stats.removed, 1);
        assert_eq!(mem.documents.len(), 0);
    }

    #[test]
    fn test_recall_typed() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");
        let mut mem = make_persistent(&storage);

        mem.ingest("plan for optimizing search", DocumentType::Plan, None)
            .unwrap();
        mem.ingest("RCA for search timeout bug", DocumentType::RCA, None)
            .unwrap();
        mem.ingest("memory note about search patterns", DocumentType::Memory, None)
            .unwrap();

        let plans = mem
            .recall_typed("search optimization", 10, &[DocumentType::Plan])
            .unwrap();
        for r in &plans {
            assert_eq!(r.document.doc_type, DocumentType::Plan);
        }
    }

    #[test]
    fn test_persistence_across_opens() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");

        {
            let mut mem = make_persistent(&storage);
            mem.ingest("persistent data about Rust", DocumentType::Memory, None)
                .unwrap();
            mem.ingest("persistent data about Python", DocumentType::Memory, None)
                .unwrap();
        }

        // Re-open.
        let mem = make_persistent(&storage);
        assert_eq!(mem.documents.len(), 2);

        let results = mem.recall("Rust", 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_stats() {
        let dir = tempdir().unwrap();
        let storage = dir.path().join("memory");
        let mut mem = make_persistent(&storage);

        mem.ingest("doc one", DocumentType::Plan, None).unwrap();
        mem.ingest("doc two", DocumentType::RCA, None).unwrap();
        mem.ingest("doc three", DocumentType::Plan, None).unwrap();

        let stats = mem.stats();
        assert_eq!(stats.total_documents, 3);
        assert_eq!(*stats.by_type.get(&DocumentType::Plan).unwrap(), 2);
        assert_eq!(*stats.by_type.get(&DocumentType::RCA).unwrap(), 1);
    }
}
