## rwkv-gepa-rust-v1

一版更窄的 Rust 合成流水线，专门做这件事：

- 从本地 JSONL 读取样本，坏行直接跳过，不做修复
- 固定读取顶层字段：`task_id`、`sample_index`、`completions_id`、`context`、`ref_answer`
- 用一个显式配置好的 OpenAI 兼容模型生成 `variant_count` 条变种 `user`
- 为每条变种稳定随机选一个答案模型作答，保留思维链，不做正确性门控
- 生成或回答某一题失败时直接跳过，不中断整批
- 用一个追加写入的输出 JSONL 记录阶段状态，最终 `done` 行保留 RWKV 微调可用的 `text` 字段
- 用 `task_id + status` 做阶段级断点续跑

当前实现只保留一个最终输出文件，也不再有旧的 `original/variants/responses` 三段中间文件。
旧 `config.toml` 字段名和旧输出 JSONL 形状都不再兼容，直接按新格式重跑。

### 运行

```bash
cargo run --release -- synthesize --config mode.toml
```

只跑前 N 条：

```bash
cargo run --release -- synthesize --config mode.toml --limit 10
```

### 配置

每个模型都必须在 TOML 里显式写出：

- `endpoint`
- `model_name`
- `api_key`

生成模型和答案模型都走 OpenAI 兼容的 chat completions 接口。TOML 里直接填写完整 endpoint，例如 `https://api.openai.com/v1/chat/completions` 或中转站自己的完整 chat endpoint。当前代码只按一种固定返回结构解析：`choices[0].message.content`，可选读取 `choices[0].message.reasoning_content`。第一阶段会额外带上 `response_format = { type = "json_object" }`，强制模型返回合法 JSON。

示例：

```toml
[generator]
endpoint = "https://api.ablai.top/v1/chat/completions"
model_name = "gpt-5.4"
api_key = "YOUR_API_KEY"
variant_count = 4

[[answer_models]]
endpoint = "https://api.ablai.top/v1/chat/completions"
model_name = "deepseek-reasoner"
api_key = "YOUR_API_KEY"
```

### 输入结构

程序支持：

- 只接受 JSONL
- 坏 JSONL 行会直接跳过，不做修复
- 固定使用顶层字段，不再从配置里写 JSONPath
- `sample_id = task_id + "_" + sample_index + "_" + completions_id`
- `prompt = context`
- `ref_answer = ref_answer`

例如：

```toml
[input]
dataset_path = "cmmlu_task122_failed_contexts.jsonl"
```

像旧 `cmmlu_task122_failed_contexts.jsonl` 这种顶层对象结构是可用的。需要的字段缺失时，该样本会被直接跳过。

### 输出格式

输出 JSONL 是一个按 `task_id` 追加的阶段日志，核心字段是：

- `task_id`
- `status`
- `user`
- `assistant`
- `text`

其中：

- `status` 是流水线运行状态，不再保留输入里的原始状态；当前会写入 `generated` 或 `done`
- `user` 是第一阶段生成出来的变种问题，也就是 `questions[].user`
- `assistant` 是回答模型的原始回答拼接结果：如果返回了 `reasoning_content`，就会包成 `<think>...</think>` 并和正文直接拼接
- `text` 是训练用字段，格式固定为 `User: {user}\nAssistant: {assistant}`

示例：

```json
{"task_id":"demo_001_q000","sample_id":"demo_001","subject":"coding","prompt":"什么是冒泡排序？","ref_answer":"一种基础排序算法","status":"done","generator_model":"gpt-5.4","user":"请解释一下什么是冒泡排序","assistant":"<think>\n...\n</think>\n\n冒泡排序是一种基础排序算法。它会重复遍历数组，比较相邻两个元素，如果顺序错误就交换它们。每一轮都会把一个较大的元素“冒”到后面。","text":"User: 请解释一下什么是冒泡排序\nAssistant: <think>\n...\n</think>\n\n冒泡排序是一种基础排序算法。它会重复遍历数组，比较相邻两个元素，如果顺序错误就交换它们。每一轮都会把一个较大的元素“冒”到后面。","answer_model":"deepseek-reasoner"}
```

### 续跑逻辑

恢复时只认新输出格式里的两种显式状态：

- 最新一条记录 `status = generated`：跳过生成，直接回答
- 最新一条记录 `status = done`：整题跳过

没有记录就视为待处理，但这不是落盘状态。

程序会为同一个 `task_id` 追加写入阶段记录：

- 生成成功后立即写一条 `generated`
- 回答成功后再追加一条 `done`

恢复时按同一个 `task_id` 的最新状态决定跳过哪个阶段。这样如果生成完成后中断，下次会直接复用已落盘的 `user`，不会再次调用生成模型。

### 静态检查

```bash
cargo fmt --check
cargo check
```
