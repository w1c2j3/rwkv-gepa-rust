## rwkv-gepa-rust-v1

Rust 重写版的 V1 流水线，目标是直接作为独立项目维护。当前覆盖两个命令：

- `synthesize`: 拉取 MMLU、生成变体、分发到 answer model 并落 JSONL
- `export`: 将变体 JSONL 导出为 parquet

```bash
cargo run --release -- synthesize --config config.toml

cargo run --release -- export \
  --input fixtures/mmlu_variants.sample.jsonl \
  --output /tmp/mmlu_variants.parquet
```

### 快速开始

1. 复制 `config.toml` 为你自己的本地配置，例如 `config.local.toml`
2. 填入真实 `base_url`、`api_key`、模型名和输出路径
3. 运行：

```bash
cargo run --release -- synthesize --config config.local.toml
```

如果你不想在线拉数据集，可以在 `config.local.toml` 中指定：

```toml
[mmlu]
local_jsonl_path = "fixtures/source_questions.sample.jsonl"
```

本地 JSONL 的每一行都应是一个题目对象，至少包含：

```json
{"question":"...","choices":["...","...","...","..."],"answer":"A"}
```

可选字段：

- `sample_id`
- `subject`

### 设计重点

- 使用 Tokio 多线程 runtime 跑高并发请求。
- Hugging Face 数据集通过 `datasets-server` 分页并发拉取。
- OpenAI 兼容接口复用长连接，避免 Python 版本里频繁的客户端开销。
- JSONL 采用独立异步 writer 任务，通过 channel 串行落盘，减少锁竞争和反复打开文件。
- 请求并发放在 `config.toml` 的 `[concurrency]` 段里调节，Tokio runtime 线程放在 `[runtime]` 段里调节。
- `mmlu.rows_api_url` 默认指向 Hugging Face `datasets-server`，也可以切到你自己的镜像或本地假服务。
- `Cargo.toml` 带了 release 优化配置，后续直接拆成独立项目时不需要再补一轮构建参数。

### 配置约定

- `config.toml` 只保留占位符，不放真实密钥。
- 请求级并发放在 `[concurrency]`。
- Tokio 线程配置放在 `[runtime]`。
- `mmlu.rows_api_url` 可指向 Hugging Face，也可指向镜像或本地假服务。
- `mmlu.local_jsonl_path` 可切换为本地题目源；一旦设置，就不再请求远端数据集。

### 目录

- `src/`: 主程序和流水线实现。
- `config.toml`: 默认配置模板，保留占位符，不提交真实密钥。
- `fixtures/`: 本地/CI smoke 测试用样例数据。
- `fixtures/source_questions.sample.jsonl`: 本地题目源样例。
- `.github/workflows/ci.yml`: 该 Rust 项目自带的 CI。
- `docs/github-remote-init.md`: 新仓库初始化和推送模板。
- `LICENSE`: 默认使用 MIT 许可证。

### 验证

```bash
cargo fmt --check
cargo test
cargo run -- export --input fixtures/mmlu_variants.sample.jsonl --output /tmp/rust-v1.parquet
```

### 注意

- 运行 `synthesize` 需要访问 Hugging Face 和你配置的 OpenAI 兼容接口。
- 如果配置了 `mmlu.local_jsonl_path`，`synthesize` 的题目输入可以完全来自本地 JSONL。
- `export --input ...` 可以指定任意本地路径，但它处理的是“变体 JSONL 导出为 parquet”的下游步骤，不是 `synthesize` 的题目源配置。
- `answer.max_concurrency` 仅保留给旧配置兼容，新的并发入口以 `[concurrency]` 和 `[runtime]` 为准。
- 如果你要把这个目录直接拷出去，整目录复制即可，不依赖父仓库的其它 Rust 文件。
