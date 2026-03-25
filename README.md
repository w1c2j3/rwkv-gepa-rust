## rwkv-gepa-rust-v1

一个更通用的 Rust 版 JSONL 合成流水线，目标是直接产出可训练的 RWKV JSONL：

- 从本地 JSONL 或 JSON array 读取原始样本
- 按 TOML 里的字段映射提取 `sample_id`、`subject`、`prompt`、`ref_answer`
- 用 `generator` 模型基于原始 `prompt` 生成 N 条 `user`
- 从 `answer_models` 里稳定伪随机选一个可思维链模型生成 `assistant`
- 最终只落一个 JSONL，直接用于训练
- 运行时会显示 `generate` 和 `answer` 两段进度条
- `answer` 阶段每成功一条都会立刻追加写入输出 JSONL，避免中途中断时整批结果丢失

```bash
cargo run --release -- synthesize --config config.toml

cargo run --release -- export \
  --input data/rwkv_train.jsonl \
  --output /tmp/rwkv_train.parquet
```

### 快速开始

1. 准备你自己的本地 JSONL
2. 在 `config.toml` 里配置输入字段映射、模型地址、模型名和 `question_count`
3. 运行：

```bash
cargo run --release -- synthesize --config config.toml
```

如果只想跑前 N 条，可以直接传：

```bash
cargo run --release -- synthesize --config config.toml --limit 10
```

### 输入 JSONL

程序不再假设输入一定是多选题，也不要求必须是 `question + choices + answer`。
如果输入里有 `context` 这种大字符串，也可以直接通过 `context.xxx` 读取内部块。

它会按 TOML 里的 path 配置去提取：

- `sample_id_path` 或 `sample_id_paths`
- `subject_path` 或 `subject_paths`
- `prompt_paths`
- `ref_answer_path`

例如这类输入：

```json
{"sample_id":"demo_001","subject":"coding","prompt":"You are a Rust expert.","question":"Write a parser for JSONL.","answer":"Use serde_json line by line."}
```

对应配置：

```toml
[input]
sample_id_path = "sample_id"
subject_path = "subject"
prompt_paths = ["prompt", "question"]
ref_answer_path = "answer"
prompt_joiner = "\n\n"
```

这样最终归一化后的 `prompt` 就是 `prompt + question` 的拼接结果。

像 `data-worry.json` 这种格式：

```json
{"context":"[lp]\nen-ar_EG\n\n[domain]\nnews\n\n[document_id]\ntest-en-news_beverly_press.3585\n\n[segment_id]\n1\n\n[reference_target]\n...\n\n[translation_prompt]\n..."}
```

可以直接这样配：

```toml
[input]
sample_id_paths = ["context.document_id", "context.segment_id"]
sample_id_joiner = "#"
subject_paths = ["context.lp", "context.domain"]
subject_joiner = "/"
prompt_paths = ["context.translation_prompt"]
ref_answer_path = "context.reference_target"
```

支持点路径，比如：

- `meta.id`
- `task.subject`
- `messages.0.content`

### 输出 JSONL

每行一个训练样本，核心字段是：

- `user`
- `assistant`

同时保留一些元信息：

- `generator_model`
- `answer_model`
- `user`
- `assistant`
- `record_id`
- `sample_id`
- `subject`
- `prompt`
- `ref_answer`

输出示例：

```json
{"generator_model":"gpt-5.4","answer_model":"qwen3-30b-a3b-thinking-2507","user":"Write a Rust function that streams a JSONL file and parses each line safely.","assistant":"<think>\n...\n</think>\n\nHere is a Rust implementation ...","record_id":"demo_001_q000","sample_id":"demo_001","subject":"coding","prompt":"You are a Rust expert.\n\nWrite a parser for JSONL.","ref_answer":"Use serde_json line by line."}
```

这里不做“回答必须正确”的强校验。只要模型能产生有价值的回答，哪怕有错误，思维链本身也可以作为训练信号保留。

### 配置说明

- `[input]` 负责输入 JSONL/JSON array 和字段映射
- `sample_id_paths`、`subject_paths` 支持多字段拼接
- `context.xxx` 会自动解析 `context` 大字符串里的块
- `input.prompt_paths` 是一个数组，会按顺序拼接成最终 `prompt`
- `[generator]` 控制生成 `user` 的模型和数量
- `generator.question_count` 是每个原始样本生成多少条训练样本
- `[[answer_models]]` 可以写任意多个，只要外接接口支持
- `api_key_env` 支持从环境变量读取密钥，优先建议这样配，不要把真实 token 写进仓库
- `[output]` 现在只需要一个 `jsonl`
- `[run]`、`[concurrency]` 都有默认值，不写会走内置默认配置
- 如果代理环境下出现 TLS/握手异常，可先保持 `force_http1 = true`；如果你本机能直连外网，也可以设 `disable_env_proxy = true`

### 现在的取舍

- 不再限定输入必须是 MMLU 多选题
- 不再落 `original/variants/responses` 三段中间文件
- 中间 JSON 只在内存里解析，不做中间态持久化
- 不再做答案正确性门控
- 仍保留 Tokio 并发、长连接复用和 OpenAI 兼容接口

### 验证

```bash
cargo fmt --check
cargo test
cargo check
```
