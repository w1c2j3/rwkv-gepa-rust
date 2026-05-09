use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about = "Lean JSONL -> RWKV training pipeline")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    Synthesize {
        #[arg(long, default_value = "mode.toml")]
        config: PathBuf,
        #[arg(long)]
        limit: Option<usize>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn synthesize_defaults_to_mode_toml() {
        let cli = Cli::try_parse_from(["bin", "synthesize"]).expect("cli should parse");
        match cli.command {
            Command::Synthesize { config, limit } => {
                assert_eq!(config, PathBuf::from("mode.toml"));
                assert_eq!(limit, None);
            }
        }
    }

    #[test]
    fn synthesize_accepts_config_and_limit() {
        let cli = Cli::try_parse_from([
            "bin",
            "synthesize",
            "--config",
            "custom.toml",
            "--limit",
            "7",
        ])
        .expect("cli should parse");
        match cli.command {
            Command::Synthesize { config, limit } => {
                assert_eq!(config, PathBuf::from("custom.toml"));
                assert_eq!(limit, Some(7));
            }
        }
    }
}
