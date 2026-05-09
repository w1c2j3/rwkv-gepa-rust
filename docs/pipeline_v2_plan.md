# 正式数据生成流水线 V2 规划

## 1. 背景

当前项目中的 `src/main.rs` 已经能够跑通一条可用的数据生成流程，但它更接近实验性实现：

- 输入归一化、提示词、模型调用、落盘、恢复逻辑都集中在一个文件里
- 提示词仍然是硬编码在程序中
- 任务类型虽然已经从 MCQ 专用逻辑中抽离了一部分，但“生成策略”仍然不够可配置
- 随着后续要支持选择题、开放题、翻译题、代码题等不同任务，继续在单文件里扩展会导致维护成本快速上升

另一方面，当前数据来源与输入形式并没有本质变化：

- 上游仍然会导出 `context` 或 `text`
- 原始数据仍然来自评测导出的错误样本
- 单轮、多轮样本都可以在导出阶段先规整好

因此，V2 的核心目标不是推翻现有输入来源，而是在保持输入格式稳定的前提下，重新设计一条正式、可扩展、可维护的数据生成流水线。

## 2. 现状共识

以下内容可以直接作为 V2 的已知前提：

1. 输入源不变。
2. 上游导出的主字段仍然是：
   - `context`
   - `text`
3. 归一化后的核心输入仍然可以统一成：
   - `SourceSample { sample_id, source_user, source_meta }`
4. answer 阶段应继续保存完整模型输出，而不是只保留最终答案。
5. 输出目录应继续拆分中间态与最终态，不再混写。

## 3. V2 设计目标

### 3.1 主目标

1. 将提示词从代码中抽离，改为从配置读取。
2. 支持按任务类型或生成策略切换不同 prompt。
3. 将 `main.rs` 中的逻辑拆分成职责明确的模块。
4. 保持核心 orchestration 通用，不把某一类题型硬编码进主流程。
5. 为后续扩展选择题、开放题、多轮任务、翻译任务等场景预留结构。

### 3.2 非目标

1. 不在 V2 第一阶段追求一次性支持所有任务细节。
2. 不改变当前输入数据源格式。
3. 不要求立刻实现复杂的自动判分策略。
4. 不要求立刻把所有旧逻辑都迁移成插件体系。

## 4. 设计原则

### 4.1 输入稳定，策略可替换

输入来源可以保持稳定，但“如何生成新样本”必须可配置。
也就是说，输入归一化层尽量通用，生成策略层则按任务类型切换。

### 4.2 Core 只负责编排

主流程只负责：

1. 读取样本
2. 归一化
3. 选择生成策略
4. 调用生成模型
5. 调用 answer 模型
6. 落盘
7. 恢复运行

主流程不应直接包含大段 prompt 文本，也不应写死任务类型细节。

### 4.3 Prompt 作为配置资产

Prompt 不应继续硬编码在 `main.rs` 中。
更合理的做法是：

- 通过配置文件指定 prompt profile
- profile 再指向对应的 prompt 模板文件
- 模板文件使用占位符渲染，例如 `{{source_prompt}}`、`{{variant_count}}`

### 4.4 比较器和校验器是策略，而不是主流程的一部分

未来不同任务的“正确性判定”会不同：

- 选择题可以用答案字母或标签比较
- 开放题不能简单字符串对比
- 翻译题可能需要更宽松的规则
- 代码题可能需要另外的验证机制

因此，比较器和校验器要作为可替换策略设计，而不是直接写死在 pipeline 中。

## 5. 目标目录结构

建议将当前单文件结构逐步拆成以下模块：

```text
src/
  main.rs
  config.rs
  types.rs
  input.rs
  prompts.rs
  client.rs
  pipeline.rs
  storage.rs
  resume.rs
  compare.rs
  profiles.rs
```

各模块职责建议如下：

### `main.rs`

- CLI 入口
- 加载配置
- 调用 pipeline

### `config.rs`

- 定义 TOML 配置结构
- 负责配置默认值与校验

### `types.rs`

- 放通用数据结构
- 如 `SourceSample`、`PendingTask`、`OutputRow`

### `input.rs`

- 读取 JSONL
- 样本归一化
- 从 `context/text` 中抽取 `source_user`

### `prompts.rs`

- 读取 prompt 模板
- 执行占位符渲染
- 组织生成 prompt / 校验 prompt / 其他辅助 prompt

### `profiles.rs`

- 定义不同任务类型或策略 profile
- 例如：
  - `multiple_choice`
  - `open_qa`
  - `translation`
  - `code_generation`

### `client.rs`

- 模型调用封装
- 支持可选 `system_prompt`
- 支持 `system + user` 消息
- 提取 reasoning / content
- 屏蔽第三方接口兼容细节

### `pipeline.rs`

- 编排 generate / answer / compare / store 的顺序
- 不直接关心 prompt 具体内容
- 不直接关心输出目录细节

### `storage.rs`

- 管理输出目录
- 写 `generate`
- 写 `done`
- 后续可扩展为 `done/success` 和 `done/failed`

### `resume.rs`

- 从落盘结果恢复运行状态
- 判断哪些任务已完成，哪些仍需继续

### `compare.rs`

- 不同任务类型的答案比较策略
- 支持后续扩展

## 6. Prompt 外置设计

### 6.1 目标

将生成 prompt 从代码中抽离成模板文件，再由配置指定当前任务使用哪套模板。

### 6.2 配置层建议

可以在 TOML 中增加类似结构：

```toml
[prompt_profile]
name = "multiple_choice"
generation_prompt_file = "prompts/multiple_choice/generation.md"
validation_prompt_file = "prompts/multiple_choice/validation.md"
```

对于开放题，则改为：

```toml
[prompt_profile]
name = "open_qa"
generation_prompt_file = "prompts/open_qa/generation.md"
validation_prompt_file = "prompts/open_qa/validation.md"
```

### 6.3 模板变量建议

模板至少支持以下变量：

- `{{source_prompt}}`
- `{{variant_count}}`
- `{{sample_id}}`
- `{{source_meta_json}}`
- `{{accepted_examples_json}}`
- `{{feedback}}`

### 6.4 Prompt 分类建议

第一阶段先支持两大类即可：

1. 选择题 profile
2. 开放题 profile

后续再逐步加：

3. 翻译 profile
4. 代码生成 profile
5. 代码补全 profile
6. 分类判断 profile

## 7. 输入层设计

输入层继续保持通用，原则是：

1. 优先读取 `context`
2. 若无 `context` 再读取 `text`
3. 优先从单轮对话结构中提取 `User:` 内容
4. 若提取失败，再退回完整归一化文本

对于多轮输入，不在主流程中写死解析细节。
如果未来多轮输入的导出格式更明确，应优先在上游导出阶段完成规整，再交给当前流水线消费。

这意味着：

- V2 的输入层仍然通用
- 多轮/单轮的细节尽量在导出阶段处理
- 生成流水线只消费已经被整理好的“可生成输入”

## 8. 输出层设计

建议后续统一使用：

```text
data/<dataset_name>/
  generate/
    tasks.jsonl
  done/
    success.jsonl
    failed.jsonl
```

说明：

- `generate/tasks.jsonl` 用于保存中间态任务
- `done/success.jsonl` 保存可接受的最终样本
- `done/failed.jsonl` 保存答错或校验失败的最终样本

这样有几个好处：

1. 中间态和终态分离
2. 便于 resume
3. 便于后续只导出成功样本
4. 便于分析失败样本

## 9. 生成与判定的第二设计

当前已出现一个明确的新方向：

1. 生成模型不仅生成新的 `user`，还生成一个标准 `answer`
2. answer 模型保留完整输出
3. 再将 answer 模型的结果与标准 `answer` 做比较
4. 正确进入 `success`
5. 错误进入 `failed`

这个方向可以显著减少“生成后靠模型质检反复重试”的成本。

但必须注意，这种设计天然更适合：

- 选择题
- 判断题
- 短答案题
- 格式非常稳定的任务

对开放题、代码题、翻译题则不能直接套同样的比较规则。
因此，V2 要把“生成策略”和“比较策略”拆开：

- 生成策略决定 prompt 如何写
- 比较策略决定结果如何判定

## 10. 配置体系建议

在保留当前配置骨架的前提下，后续建议扩展为：

```toml
[input]
dataset_path = "..."

[generator]
variant_count = 4
...

[prompt_profile]
name = "multiple_choice"
generation_prompt_file = "prompts/multiple_choice/generation.md"
validation_prompt_file = "prompts/multiple_choice/validation.md"

[[answer_models]]
endpoint = "..."
model_name = "..."
api_key = "..."
system_prompt = "..."

[output]
dataset_root = "data"

[run]
resume = true

[concurrency]
generate_requests = 16
answer_requests = 16
```

后续如有需要，再增加：

- `compare_strategy`
- `extract_strategy`
- `task_family`

## 11. 迁移路径

建议分三步走。

### 第一阶段：整理结构，不改行为

目标：

- 把 `main.rs` 拆模块
- 让项目编译通过
- 保持当前行为尽量不变

这一阶段重点是“代码组织重构”，不是功能重写。

### 第二阶段：Prompt 外置与 profile 化

目标：

- 将生成 prompt / 校验 prompt 从代码中移出
- 配置中指定 prompt profile
- 支持至少两类 profile：选择题、开放题

### 第三阶段：引入生成答案 + 比较路由

目标：

- 让生成模型输出 `user + answer`
- answer 模型输出完整结果
- 比较后写入 `success/failed`

这一步应先从选择题场景落地，再逐步推广到其他任务。

## 12. 运行中的任务是否会受后续修改影响

从运行机制上看，通常不会。

原因如下：

1. 已经启动的 Rust 可执行文件会把当前代码加载到进程内存中。
2. 之后再修改源码，不会改变已经运行中的进程逻辑。
3. 即使重新编译，新生成的二进制也只会影响后续新启动的任务，不会改变已经运行中的那一批。

但有几个例外要注意：

1. 如果手动修改当前运行任务正在使用的配置文件、输入文件或输出文件，可能会影响运行结果或恢复逻辑。
2. 如果程序设计成运行时动态读取外部 prompt 文件，那么修改这些 prompt 文件可能影响运行中的进程。
3. 如果对同一个输出目录下的文件进行人工改写，可能干扰当前任务的落盘与 resume。

因此，推荐操作原则是：

1. 运行中的任务继续使用当前版本，不动其配置和输出目录。
2. 新设计在新的配置、目录或分支上进行。
3. 等当前批次跑完后，再切换到新框架。

## 13. 下一步建议

建议接下来按这个顺序推进：

1. 先完成 `main.rs` 的模块拆分
2. 再引入 prompt 外置
3. 再加 prompt profile
4. 最后再做“生成答案 + answer 比较 + success/failed 路由”

这样可以避免一次性改太多，导致系统同时发生结构变化和行为变化，难以定位问题。
