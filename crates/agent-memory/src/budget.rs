use crate::document::Document;
use crate::ranker::RankedResult;

/// Select documents within a token budget.
///
/// 1. Always keep the `always_keep_recent` most recent documents (by timestamp).
/// 2. Sort remaining candidates by combined_score descending.
/// 3. Greedily add until budget exhausted.
/// 4. Return selected document IDs in chronological (timestamp) order.
pub fn select_within_budget(
    ranked: &[RankedResult],
    documents: &[Document],
    token_budget: usize,
    always_keep_recent: usize,
) -> Vec<u64> {
    if ranked.is_empty() {
        return vec![];
    }

    // Build a list of (id, timestamp, token_count) for ranked documents.
    let mut candidates: Vec<(u64, u64, usize)> = ranked
        .iter()
        .filter_map(|r| {
            documents
                .iter()
                .find(|d| d.id == r.id)
                .map(|d| (d.id, d.timestamp, d.token_count))
        })
        .collect();

    // Sort by timestamp descending to find the most recent ones.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));

    let mut selected = std::collections::HashSet::new();
    let mut used_tokens = 0usize;

    // Step 1: Always keep the N most recent documents (if budget allows).
    for &(id, _ts, tokens) in candidates.iter().take(always_keep_recent) {
        if used_tokens + tokens <= token_budget {
            selected.insert(id);
            used_tokens += tokens;
        }
    }

    // Step 2: Greedily add remaining by combined_score descending.
    // ranked is already sorted by score descending.
    for r in ranked {
        if selected.contains(&r.id) {
            continue;
        }
        if let Some(doc) = documents.iter().find(|d| d.id == r.id) {
            if used_tokens + doc.token_count <= token_budget {
                selected.insert(r.id);
                used_tokens += doc.token_count;
            }
        }
    }

    // Step 3: Return in chronological (timestamp) order.
    let mut result: Vec<(u64, u64)> = selected
        .into_iter()
        .filter_map(|id| {
            documents
                .iter()
                .find(|d| d.id == id)
                .map(|d| (d.id, d.timestamp))
        })
        .collect();
    result.sort_by_key(|&(_, ts)| ts);

    result.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::DocumentType;
    use crate::ranker::RankedResult;
    use std::collections::HashMap;

    fn make_doc(id: u64, timestamp: u64, tokens: usize) -> Document {
        Document {
            id,
            doc_type: DocumentType::ConversationTurn,
            source_path: None,
            content: format!("doc {id}"),
            content_preview: format!("doc {id}"),
            token_count: tokens,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    fn make_ranked(id: u64, score: f32) -> RankedResult {
        RankedResult {
            id,
            relevance_score: score,
            recency_score: 0.5,
            combined_score: score,
        }
    }

    #[test]
    fn test_budget_respected() {
        let docs = vec![
            make_doc(1, 100, 50),
            make_doc(2, 200, 50),
            make_doc(3, 300, 50),
            make_doc(4, 400, 50),
        ];
        let ranked = vec![
            make_ranked(4, 0.9),
            make_ranked(3, 0.8),
            make_ranked(2, 0.7),
            make_ranked(1, 0.6),
        ];

        // Budget of 120 tokens: can fit at most 2 docs (each 50 tokens)
        // With always_keep_recent=0, pick top 2 by score: ids 4, 3
        let selected = select_within_budget(&ranked, &docs, 120, 0);
        assert_eq!(selected.len(), 2);
        let total: usize = selected
            .iter()
            .filter_map(|id| docs.iter().find(|d| d.id == *id))
            .map(|d| d.token_count)
            .sum();
        assert!(total <= 120);
    }

    #[test]
    fn test_recent_always_kept() {
        let docs = vec![
            make_doc(1, 100, 30), // oldest, highest score
            make_doc(2, 200, 30),
            make_doc(3, 300, 30), // 2nd most recent
            make_doc(4, 400, 30), // most recent
        ];
        let ranked = vec![
            make_ranked(1, 0.99),
            make_ranked(2, 0.50),
            make_ranked(3, 0.30),
            make_ranked(4, 0.10),
        ];

        // Budget of 90 tokens = 3 docs. always_keep_recent=2 means docs 4 and 3 are kept.
        // Remaining budget = 30 tokens, best remaining by score is doc 1.
        let selected = select_within_budget(&ranked, &docs, 90, 2);
        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&3));
        assert!(selected.contains(&4));
        assert!(selected.contains(&1));
    }

    #[test]
    fn test_chronological_ordering() {
        let docs = vec![
            make_doc(1, 100, 20),
            make_doc(2, 300, 20),
            make_doc(3, 200, 20),
        ];
        let ranked = vec![
            make_ranked(2, 0.9),
            make_ranked(3, 0.8),
            make_ranked(1, 0.7),
        ];

        let selected = select_within_budget(&ranked, &docs, 1000, 0);
        // Should be in timestamp order: 1 (100), 3 (200), 2 (300)
        assert_eq!(selected, vec![1, 3, 2]);
    }

    #[test]
    fn test_empty() {
        let selected = select_within_budget(&[], &[], 1000, 0);
        assert!(selected.is_empty());
    }
}
