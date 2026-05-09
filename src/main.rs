mod answer;
mod cli;
mod config;
mod generation;
mod input;
mod openai;
mod output;
mod pipeline;
mod prompt;
mod text;
mod types;
mod util;

#[cfg(test)]
mod test_support;

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::pipeline::synthesize;

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let result = match Cli::parse().command {
        Command::Synthesize { config, limit } => synthesize(&config, limit).await,
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
