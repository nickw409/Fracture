use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    tracing::info!("Fracture inference server (CUDA backend)");

    // TODO: Parse CLI args (model path, port, max_seq_len)
    // TODO: Initialize CUDA backend
    // TODO: Load model weights
    // TODO: Create engine
    // TODO: Start HTTP server

    let router = fracture_server::create_router();
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    tracing::info!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, router).await?;

    Ok(())
}
