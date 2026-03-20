use std::fs::OpenOptions as StdOpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use tokio::fs::OpenOptions;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub struct JsonlSender<T> {
    sender: mpsc::Sender<T>,
}

pub struct JsonlWriter<T> {
    sender: JsonlSender<T>,
    handle: JoinHandle<Result<()>>,
    _marker: PhantomData<T>,
}

impl<T> JsonlWriter<T>
where
    T: Serialize + Send + 'static,
{
    pub async fn open(path: &Path, append: bool, channel_capacity: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("Failed to create parent directory for {}", path.display())
            })?;
        }
        if append {
            ensure_trailing_newline(path)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(path)
            .await
            .with_context(|| format!("Failed to open JSONL file {}", path.display()))?;

        let (tx, mut rx) = mpsc::channel::<T>(channel_capacity);
        let path_string = path.display().to_string();
        let handle = tokio::spawn(async move {
            let mut writer = BufWriter::with_capacity(1 << 20, file);
            while let Some(record) = rx.recv().await {
                let mut encoded = serde_json::to_vec(&record).with_context(|| {
                    format!("Failed to serialize JSONL record for {path_string}")
                })?;
                encoded.push(b'\n');
                writer
                    .write_all(&encoded)
                    .await
                    .with_context(|| format!("Failed to write JSONL record to {path_string}"))?;
            }
            writer
                .flush()
                .await
                .with_context(|| format!("Failed to flush JSONL writer for {path_string}"))?;
            Ok(())
        });

        Ok(Self {
            sender: JsonlSender { sender: tx },
            handle,
            _marker: PhantomData,
        })
    }

    pub fn sender(&self) -> JsonlSender<T> {
        self.sender.clone()
    }

    pub async fn close(self) -> Result<()> {
        drop(self.sender);
        self.handle
            .await
            .map_err(|error| anyhow!("JSONL writer task join failed: {error}"))?
    }
}

impl<T> Clone for JsonlSender<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

impl<T> JsonlSender<T>
where
    T: Send + 'static,
{
    pub async fn send(&self, value: T) -> Result<()> {
        self.sender
            .send(value)
            .await
            .map_err(|_| anyhow!("JSONL writer channel closed unexpectedly"))
    }
}

fn ensure_trailing_newline(path: &Path) -> Result<()> {
    let mut file = match StdOpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!("Failed to inspect existing JSONL file {}", path.display())
            });
        }
    };

    let length = file
        .metadata()
        .with_context(|| format!("Failed to stat JSONL file {}", path.display()))?
        .len();
    if length == 0 {
        return Ok(());
    }

    file.seek(SeekFrom::End(-1))
        .with_context(|| format!("Failed to seek JSONL file {}", path.display()))?;
    let mut tail = [0_u8; 1];
    file.read_exact(&mut tail)
        .with_context(|| format!("Failed to read JSONL tail byte from {}", path.display()))?;

    if tail[0] != b'\n' {
        file.seek(SeekFrom::End(0))
            .with_context(|| format!("Failed to seek JSONL end {}", path.display()))?;
        file.write_all(b"\n")
            .with_context(|| format!("Failed to append newline to {}", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde::Serialize;

    use super::*;

    #[derive(Debug, Serialize)]
    struct TestRecord<'a> {
        value: &'a str,
    }

    fn temp_path(name: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rwkv-gepa-rust-v1-{name}-{}-{unique}.jsonl",
            std::process::id()
        ))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn append_mode_adds_newline_before_new_record() {
        let path = temp_path("jsonl-writer");
        std::fs::write(&path, "{\"value\":\"existing\"}").expect("seed file");

        let writer = JsonlWriter::open(&path, true, 4)
            .await
            .expect("writer should open");
        writer
            .sender()
            .send(TestRecord { value: "next" })
            .await
            .expect("send should succeed");
        writer.close().await.expect("writer should close");

        let output = std::fs::read_to_string(&path).expect("file should be readable");
        assert_eq!(output, "{\"value\":\"existing\"}\n{\"value\":\"next\"}\n");

        let _ = std::fs::remove_file(path);
    }
}
