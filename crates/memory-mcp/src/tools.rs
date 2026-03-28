use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;

use agent_memory::document::DocumentType;

use crate::state::{resolve_project_dir, ServerState};

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct IngestParams {
    /// Text content to ingest directly (use this OR path/glob, not both).
    pub content: Option<String>,
    /// File path to ingest.
    pub path: Option<String>,
    /// Glob pattern to ingest multiple files (e.g., '~/.claude/plans/*.md').
    pub glob: Option<String>,
    /// Type of document.
    #[serde(default = "default_doc_type")]
    pub doc_type: Option<String>,
    /// Project directory to scope the memory index (default: current working directory).
    pub project_dir: Option<String>,
}

fn default_doc_type() -> Option<String> {
    Some("Custom".to_string())
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Natural language query to search for.
    pub query: String,
    /// Number of results to return (default: 5).
    #[serde(default = "default_top_k")]
    pub top_k: Option<usize>,
    /// Filter by document types.
    pub doc_types: Option<Vec<String>>,
    /// Project directory to scope the search.
    pub project_dir: Option<String>,
}

fn default_top_k() -> Option<usize> {
    Some(5)
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct SyncParams {
    /// Project directory to sync.
    pub project_dir: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
pub struct StatsParams {
    /// Project directory.
    pub project_dir: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper: parse doc_type string to DocumentType
// ---------------------------------------------------------------------------

fn parse_doc_type(s: &str) -> DocumentType {
    match s {
        "Plan" => DocumentType::Plan,
        "Memory" => DocumentType::Memory,
        "RCA" => DocumentType::RCA,
        "CommitContext" => DocumentType::CommitContext,
        "Custom" => DocumentType::Custom("Custom".to_string()),
        other => DocumentType::Custom(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct MemoryServer {
    pub state: Arc<ServerState>,
    tool_router: ToolRouter<Self>,
}

impl MemoryServer {
    pub fn new(state: Arc<ServerState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl MemoryServer {
    /// Ingest documents into the semantic memory index. Supports individual text, file paths, or glob patterns.
    #[tool(name = "memory_ingest")]
    async fn memory_ingest(
        &self,
        Parameters(params): Parameters<IngestParams>,
    ) -> Result<String, String> {
        let project_dir = resolve_project_dir(params.project_dir.as_deref());
        let doc_type = parse_doc_type(
            params
                .doc_type
                .as_deref()
                .unwrap_or("Custom"),
        );

        let memory = self
            .state
            .get_memory(&project_dir)
            .await
            .map_err(|e| format!("Failed to open memory: {e}"))?;

        // Ingest from glob pattern.
        if let Some(glob_pattern) = &params.glob {
            let mut mem = memory.write().await;
            let count = mem
                .ingest_glob(glob_pattern, doc_type)
                .map_err(|e| format!("Glob ingest failed: {e}"))?;
            return Ok(format!("Ingested {count} files matching '{glob_pattern}'"));
        }

        // Ingest from file path.
        if let Some(path) = &params.path {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read file '{path}': {e}"))?;
            let mut mem = memory.write().await;
            let id = mem
                .ingest(&content, doc_type, Some(path))
                .map_err(|e| format!("Ingest failed: {e}"))?;
            return Ok(format!("Ingested file '{path}' (doc id: {id})"));
        }

        // Ingest from direct content.
        if let Some(content) = &params.content {
            let mut mem = memory.write().await;
            let id = mem
                .ingest(content, doc_type, None)
                .map_err(|e| format!("Ingest failed: {e}"))?;
            let preview: String = content.chars().take(80).collect();
            return Ok(format!(
                "Ingested text (doc id: {id}, {len} chars): \"{preview}...\"",
                len = content.len()
            ));
        }

        Err("No content, path, or glob provided. Supply at least one.".to_string())
    }

    /// Recall relevant prior knowledge from the semantic memory index. Returns documents ranked by semantic similarity to your query.
    #[tool(name = "memory_recall")]
    async fn memory_recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<String, String> {
        let project_dir = resolve_project_dir(params.project_dir.as_deref());
        let top_k = params.top_k.unwrap_or(5);

        let memory = self
            .state
            .get_memory(&project_dir)
            .await
            .map_err(|e| format!("Failed to open memory: {e}"))?;

        let doc_types: Vec<DocumentType> = params
            .doc_types
            .unwrap_or_default()
            .iter()
            .map(|s| parse_doc_type(s))
            .collect();

        let mem = memory.read().await;
        let results = if doc_types.is_empty() {
            mem.recall(&params.query, top_k)
        } else {
            mem.recall_typed(&params.query, top_k, &doc_types)
        }
        .map_err(|e| format!("Recall failed: {e}"))?;

        if results.is_empty() {
            return Ok("No relevant documents found.".to_string());
        }

        let mut output = format!("Found {} relevant documents:\n\n", results.len());
        for (i, r) in results.iter().enumerate() {
            let doc = &r.document;
            let source = doc
                .source_path
                .as_deref()
                .unwrap_or("(inline)");
            output.push_str(&format!(
                "--- Result {num} (score: {score:.3}, type: {dtype:?}) ---\n\
                 Source: {source}\n\
                 {content}\n\n",
                num = i + 1,
                score = r.combined_score,
                dtype = doc.doc_type,
                source = source,
                content = doc.content,
            ));
        }

        Ok(output)
    }

    /// Synchronize the memory index with files on disk. Re-indexes changed files, removes deleted files.
    #[tool(name = "memory_sync")]
    async fn memory_sync(
        &self,
        Parameters(params): Parameters<SyncParams>,
    ) -> Result<String, String> {
        let project_dir = resolve_project_dir(params.project_dir.as_deref());

        let memory = self
            .state
            .get_memory(&project_dir)
            .await
            .map_err(|e| format!("Failed to open memory: {e}"))?;

        let mut mem = memory.write().await;
        let stats = mem.sync().map_err(|e| format!("Sync failed: {e}"))?;

        Ok(format!(
            "Sync complete: {added} added, {updated} updated, {removed} removed, {unchanged} unchanged",
            added = stats.added,
            updated = stats.updated,
            removed = stats.removed,
            unchanged = stats.unchanged,
        ))
    }

    /// Get statistics about the semantic memory index.
    #[tool(name = "memory_stats")]
    async fn memory_stats(
        &self,
        Parameters(params): Parameters<StatsParams>,
    ) -> Result<String, String> {
        let project_dir = resolve_project_dir(params.project_dir.as_deref());

        let memory = self
            .state
            .get_memory(&project_dir)
            .await
            .map_err(|e| format!("Failed to open memory: {e}"))?;

        let mem = memory.read().await;
        let stats = mem.stats();

        let mut type_breakdown = String::new();
        for (dtype, count) in &stats.by_type {
            type_breakdown.push_str(&format!("  {dtype:?}: {count}\n"));
        }
        if type_breakdown.is_empty() {
            type_breakdown = "  (none)\n".to_string();
        }

        Ok(format!(
            "Memory stats for project: {project_dir}\n\
             Total documents: {total}\n\
             Total tokens (approx): {tokens}\n\
             Index size: {size} bytes\n\
             By type:\n{breakdown}",
            total = stats.total_documents,
            tokens = stats.total_tokens,
            size = stats.index_size_bytes,
            breakdown = type_breakdown,
        ))
    }
}

#[tool_handler]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(rmcp::model::Implementation::new(
                "memory-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Semantic memory server for AI coding agents. \
                 Use memory_ingest to store documents, memory_recall to search, \
                 memory_sync to re-index changed files, and memory_stats for index info."
                    .to_string(),
            )
    }
}
