mod config;
mod dataset;
mod export;
mod jsonl;
mod openai;
mod pipeline;
mod types;
mod util;

use anyhow::Context;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use tokio::runtime::Builder;

use crate::config::load_config;
use crate::util::log;

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Rust rewrite of the V1 MMLU synthesis pipeline"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Synthesize {
        #[arg(long, default_value = "config.toml")]
        config: PathBuf,
    },
    Export {
        #[arg(long, default_value = "data/mmlu_variants.jsonl")]
        input: PathBuf,
        #[arg(long, default_value = "data/mmlu_variants.parquet")]
        output: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Synthesize { config } => (|| {
            let config = load_config(&config)?;
            let runtime = Builder::new_multi_thread()
                .enable_all()
                .worker_threads(config.runtime.worker_threads)
                .max_blocking_threads(config.runtime.max_blocking_threads)
                .build()
                .with_context(|| "Failed to build Tokio runtime")?;
            runtime.block_on(pipeline::run(config))
        })(),
        Commands::Export { input, output } => export::run(&input, &output),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            log(format!("Fatal error: {error:#}"));
            ExitCode::FAILURE
        }
    }
}
