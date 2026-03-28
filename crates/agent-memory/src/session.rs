use std::sync::Arc;

use turboquant::TurboIndex;

use crate::budget::select_within_budget;
use crate::document::{Document, DocumentType};
use crate::embedder::Embedder;
use crate::ranker::{rank, RankingConfig};
use crate::AgentMemoryError;

/// In-memory session context manager. Tracks conversation turns within a
/// single session and selects the most relevant context within a token budget.
pub struct SessionMemory {
    embedder: Arc<dyn Embedder>,
    index: TurboIndex,
    documents: Vec<Document>,
    turn_counter: usize,
    _tmpdir: tempfile::TempDir,
}

impl SessionMemory {
    /// Create a new session memory backed by a temporary directory.
    pub fn new(embedder: Arc<dyn Embedder>) -> Result<Self, AgentMemoryError> {
        let tmpdir = tempfile::tempdir()?;
        let index_path = tmpdir.path().join("session_index");
        let dim = embedder.dim();
        let index = TurboIndex::create(&index_path, dim, 42, 99)?;

        Ok(Self {
            embedder,
            index,
            documents: Vec::new(),
            turn_counter: 0,
            _tmpdir: tmpdir,
        })
    }

    /// Add a conversation turn. Returns the assigned document ID.
    pub fn add_turn(
        &mut self,
        role: &str,
        content: &str,
        token_count: usize,
    ) -> Result<u64, AgentMemoryError> {
        self.turn_counter += 1;
        let id = self.turn_counter as u64;

        let prefixed = format!("{role}: {content}");
        let mut doc = Document::new(id, DocumentType::ConversationTurn, prefixed);
        doc.token_count = token_count;
        doc.metadata
            .insert("role".to_string(), role.to_string());
        doc.metadata
            .insert("turn_number".to_string(), self.turn_counter.to_string());

        let embedding = self.embedder.embed(&doc.content);
        self.index.insert(id, &embedding)?;
        self.documents.push(doc);

        Ok(id)
    }

    /// Select the most relevant context within the given token budget.
    ///
    /// 1. Embed the current message.
    /// 2. Search the TurboIndex for top-50 candidates.
    /// 3. Rank by relevance + recency.
    /// 4. Select within token budget, always keeping the most recent turns.
    /// 5. Return documents in original turn order.
    pub fn select_context(
        &self,
        current_message: &str,
        token_budget: usize,
        config: Option<RankingConfig>,
    ) -> Result<Vec<&Document>, AgentMemoryError> {
        if self.documents.is_empty() {
            return Ok(vec![]);
        }

        let config = config.unwrap_or_default();
        let query_embedding = self.embedder.embed(current_message);
        let search_results = self.index.search(&query_embedding, 50);

        let ranked = rank(
            &search_results,
            &self.documents,
            self.turn_counter as u64,
            &config,
        );

        let selected_ids = select_within_budget(
            &ranked,
            &self.documents,
            token_budget,
            config.always_keep_recent,
        );

        // Return documents in original turn order (by ID, which is monotonically increasing).
        let mut result: Vec<&Document> = self
            .documents
            .iter()
            .filter(|d| selected_ids.contains(&d.id))
            .collect();
        result.sort_by_key(|d| d.id);

        Ok(result)
    }

    /// Number of turns added so far.
    pub fn turn_count(&self) -> usize {
        self.turn_counter
    }

    /// Total tokens across all stored documents.
    pub fn total_tokens(&self) -> usize {
        self.documents.iter().map(|d| d.token_count).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::MockEmbedder;

    fn make_session() -> SessionMemory {
        let embedder = Arc::new(MockEmbedder::new(32));
        SessionMemory::new(embedder).unwrap()
    }

    #[test]
    fn test_add_turns_and_count() {
        let mut session = make_session();
        for i in 0..10 {
            session
                .add_turn("user", &format!("message {i}"), 20)
                .unwrap();
        }
        assert_eq!(session.turn_count(), 10);
        assert_eq!(session.total_tokens(), 200);
    }

    #[test]
    fn test_select_context_budget() {
        let mut session = make_session();
        // Add 10 turns, each 50 tokens
        for i in 0..10 {
            session
                .add_turn("user", &format!("topic {i} about Rust programming"), 50)
                .unwrap();
        }

        // Budget of 200 tokens = at most 4 turns
        let context = session
            .select_context("Rust programming question", 200, None)
            .unwrap();
        let total_tokens: usize = context.iter().map(|d| d.token_count).sum();
        assert!(total_tokens <= 200, "total_tokens={total_tokens}");
        assert!(!context.is_empty());
    }

    #[test]
    fn test_recent_turns_always_included() {
        let mut session = make_session();
        // Add 10 turns
        for i in 0..10 {
            session
                .add_turn("user", &format!("turn {i} content"), 20)
                .unwrap();
        }

        // Budget allows all 10 turns (200 tokens). With always_keep_recent=4,
        // the last 4 turns (IDs 7,8,9,10) should always be present.
        let config = RankingConfig {
            relevance_weight: 0.6,
            recency_weight: 0.4,
            always_keep_recent: 4,
        };
        let context = session
            .select_context("anything", 200, Some(config))
            .unwrap();

        let ids: Vec<u64> = context.iter().map(|d| d.id).collect();
        // Last 4 turn IDs should be present
        for expected_id in 7..=10 {
            assert!(
                ids.contains(&expected_id),
                "expected turn {expected_id} to be in context, got {ids:?}"
            );
        }
    }

    #[test]
    fn test_select_context_chronological_order() {
        let mut session = make_session();
        for i in 0..5 {
            session
                .add_turn("user", &format!("turn {i}"), 10)
                .unwrap();
        }

        let context = session.select_context("turn", 1000, None).unwrap();
        let ids: Vec<u64> = context.iter().map(|d| d.id).collect();
        // IDs should be in ascending order
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "expected chronological order, got {ids:?}");
        }
    }

    #[test]
    fn test_empty_session() {
        let session = make_session();
        let context = session.select_context("hello", 1000, None).unwrap();
        assert!(context.is_empty());
    }
}
