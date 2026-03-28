mod state;
mod tools;

use std::sync::Arc;

use rmcp::ServiceExt;
use tracing_subscriber::EnvFilter;

use state::ServerState;
use tools::MemoryServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (logs to stderr so they don't interfere with MCP stdio).
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    tracing::info!("memory-mcp server starting");

    // Create server state with MockEmbedder (real ONNX model not yet available).
    let state = Arc::new(ServerState::with_mock_embedder());
    let server = MemoryServer::new(state);

    // Serve over stdio (MCP standard transport).
    let service = server.serve(rmcp::transport::stdio()).await?;

    tracing::info!("memory-mcp server running, awaiting requests on stdio");

    // Block until the service shuts down.
    service.waiting().await?;

    tracing::info!("memory-mcp server stopped");
    Ok(())
}
