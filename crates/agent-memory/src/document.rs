use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DocumentType {
    Plan,
    Memory,
    RCA,
    CommitContext,
    ConversationTurn,
    ToolResult,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: u64,
    pub doc_type: DocumentType,
    pub source_path: Option<String>,
    pub content: String,
    pub content_preview: String,
    pub token_count: usize,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(id: u64, doc_type: DocumentType, content: String) -> Self {
        let preview = content.chars().take(200).collect();
        Self {
            id,
            doc_type,
            source_path: None,
            content,
            content_preview: preview,
            token_count: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecallResult {
    pub document: Document,
    pub relevance_score: f32,
    pub recency_score: f32,
    pub combined_score: f32,
}

#[derive(Debug, Clone)]
pub struct SyncStats {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub unchanged: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_documents: usize,
    pub total_tokens: usize,
    pub index_size_bytes: usize,
    pub by_type: HashMap<DocumentType, usize>,
}
