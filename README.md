# ComfyUI-HTTP

ComfyUI 的 HTTP 请求与 LLM 集成节点包，提供通用 HTTP 请求、轮询、异步任务、并发编排以及 LLM 多模态生成等能力。所有节点均位于 **Malette** 分类下。

---

## 安装

将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录下：

```bash
cd ComfyUI/custom_nodes
git clone <repo_url> ComfyUI-HTTP
```

安装依赖：

```bash
pip install -r requirements.txt
```

重启 ComfyUI 即可在节点列表中看到新增节点。

---

## 节点一览

| 节点名称 | 显示名称 | 说明 |
|---|---|---|
| `COMMON_HTTP_REQUEST` | 通用HTTP请求 | 发送单次 HTTP 请求 |
| `POLLING_HTTP_REQUEST` | 轮询HTTP请求 | 周期性轮询直到满足条件 |
| `ASYNC_TASK_HTTP_REQUEST` | 异步任务HTTP请求 | 提交任务 + 自动轮询结果 |
| `CONCURRENT_HTTP_REQUEST` | 并发HTTP请求 | 并发编排多个请求，支持异步轮询 |
| `LLMImageGenerate` | LLM 图像生成 | 调用 LLM API 生成图像 |
| `LLMSmartGenerate` | LLM 智能生成 | 多模态 LLM 调用（文本/图片/视频/音频） |
| `LLMResponseImageParser` | LLM响应图片解析 | 从 LLM 响应中解析图片 |
| `LLMResponseSmartParser` | LLM响应智能解析 | 从 LLM 响应中解析文本/图片/视频/音频 |
| `MaletteTextConcatenate` | 文本拼接 | 多段文本拼接，支持多种分隔符 |
| `MaletteImageBatchMerge` | 图片批次合并 | 将多个图片/批次合并为一个批次 |

---

## HTTP 节点

### 通用HTTP请求 (`COMMON_HTTP_REQUEST`)

发送单次 HTTP 请求并返回响应。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `method` | 枚举 | ✅ | — | 请求方法：POST / GET / PUT / DELETE / PATCH / HEAD |
| `params` | STRING | ✅ | — | 请求参数 JSON（通过连线输入） |
| `api_key` | STRING | ✅ | `""` | API 密钥，设置到 `Authorization: Bearer` 头 |
| `api_endpoint` | STRING | ✅ | `/api/v1/common` | 请求地址 |
| `headers` | STRING | ❌ | `""` | 自定义请求头 JSON，会与默认头合并 |
| `timeout` | INT | ❌ | `600` | 超时时间（秒），范围 1-3600 |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `response` | STRING | 响应内容 JSON 字符串 |

---

### 轮询HTTP请求 (`POLLING_HTTP_REQUEST`)

周期性发送 HTTP 请求，直到响应满足指定条件或达到最大尝试次数。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `method` | 枚举 | ✅ | — | 请求方法 |
| `api_endpoint` | STRING | ✅ | `""` | 请求地址 |
| `params` | STRING | ✅ | `{}` | 请求参数 JSON |
| `poll_interval` | INT | ✅ | `3` | 轮询间隔（秒），范围 1-60 |
| `max_attempts` | INT | ✅ | `10` | 最大轮询次数，范围 1-100 |
| `success_condition` | 枚举 | ✅ | — | 成功条件类型：`status_field` / `custom_jsonpath` / `status_code_only` |
| `condition_field` | STRING | ✅ | `status` | 状态字段路径，如 `data.status`、`result.state` |
| `expected_value` | STRING | ✅ | `completed` | 期望的状态值 |
| `api_key` | STRING | ❌ | `""` | API 密钥 |
| `headers` | STRING | ❌ | `""` | 自定义请求头 JSON |
| `timeout` | INT | ❌ | `30` | 单次请求超时时间（秒） |
| `stop_on_error` | BOOLEAN | ❌ | `True` | 遇到错误时是否立即停止 |

**成功条件说明：**

- **`status_field`**：检查响应中指定字段路径的值是否等于期望值
- **`custom_jsonpath`**：支持多值匹配（逗号分隔）、包含判断（`contains:xxx`）、否定判断（`not:xxx`）
- **`status_code_only`**：仅检查 HTTP 状态码是否为 200

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `response` | STRING | 最后一次响应内容 |
| `attempts` | INT | 实际轮询次数 |
| `success` | BOOLEAN | 是否成功 |
| `final_status` | STRING | 最终状态描述 |

---

### 异步任务HTTP请求 (`ASYNC_TASK_HTTP_REQUEST`)

提交异步任务 + 自动轮询结果的一体化节点。先发送提交请求，从响应中提取 `task_id`，然后自动轮询直到任务完成。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `submit_method` | 枚举 | ✅ | — | 提交请求方法 |
| `submit_endpoint` | STRING | ✅ | `""` | 提交请求地址 |
| `submit_params` | STRING | ✅ | `{}` | 提交请求参数 JSON |
| `task_id_field` | STRING | ✅ | `data.task_id` | 从提交响应中提取任务 ID 的字段路径 |
| `poll_method` | 枚举 | ✅ | — | 轮询请求方法 |
| `poll_endpoint` | STRING | ✅ | `""` | 轮询地址，支持 `{task_id}` 占位符 |
| `poll_params` | STRING | ✅ | `{}` | 轮询参数 JSON，支持 `{task_id}` 占位符 |
| `poll_interval` | INT | ✅ | `3` | 轮询间隔（秒） |
| `max_attempts` | INT | ✅ | `20` | 最大轮询次数 |
| `success_condition` | 枚举 | ✅ | — | 轮询成功条件类型 |
| `condition_field` | STRING | ✅ | `data.status` | 轮询响应中的状态字段路径 |
| `expected_value` | STRING | ✅ | `completed` | 期望状态值 |
| `api_key` | STRING | ❌ | `""` | API 密钥 |
| `headers` | STRING | ❌ | `""` | 自定义请求头 JSON |
| `timeout` | INT | ❌ | `30` | 单次请求超时时间（秒） |
| `stop_on_error` | BOOLEAN | ❌ | `True` | 遇到错误时是否立即停止 |
| `task_id_placeholder` | STRING | ❌ | `{task_id}` | 轮询地址和参数中的任务 ID 占位符 |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `submit_response` | STRING | 提交请求的响应 |
| `task_id` | STRING | 提取到的任务 ID |
| `poll_response` | STRING | 最终轮询响应 |
| `attempts` | INT | 轮询次数 |
| `success` | BOOLEAN | 是否成功 |
| `final_status` | STRING | 最终状态描述 |

---

### 并发HTTP请求 (`CONCURRENT_HTTP_REQUEST`)

并发编排多个 HTTP 请求，每个请求可独立配置 URL、method、参数、请求头，并支持标记为异步任务自动轮询。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `requests_json` | STRING | ✅ | `[]` | 请求描述 JSON 数组（详见下方格式说明） |
| `max_concurrency` | INT | ✅ | `5` | 最大并发数，范围 1-50 |
| `default_api_key` | STRING | ❌ | `""` | 全局默认 API 密钥，各请求可单独覆盖 |
| `default_headers` | STRING | ❌ | `""` | 全局默认请求头 JSON，各请求可追加或覆盖 |
| `default_timeout` | INT | ❌ | `60` | 全局默认超时时间（秒） |
| `stop_on_error` | BOOLEAN | ❌ | `False` | 任一请求失败时是否中止剩余请求 |

**`requests_json` 格式说明：**

每个数组元素是一个 JSON 对象，描述一个独立的请求任务：

```json
[
  {
    "url": "https://api-a.com/generate",
    "method": "POST",
    "params": {"prompt": "画一只猫"},
    "headers": {"X-Custom": "value1"},
    "api_key": "sk-xxx"
  },
  {
    "url": "https://api-b.com/query",
    "method": "GET",
    "params": {"id": 123}
  },
  {
    "url": "https://api-c.com/async-task",
    "method": "POST",
    "params": {"prompt": "生成视频"},
    "async": true,
    "task_id_field": "data.task_id",
    "poll_url": "https://api-c.com/status?task_id={task_id}",
    "poll_method": "GET",
    "poll_params": {},
    "poll_interval": 3,
    "max_poll_attempts": 30,
    "condition_field": "data.status",
    "expected_value": "completed"
  }
]
```

**每个请求对象支持的字段：**

| 字段 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `url` | ✅ | — | 请求地址 |
| `method` | ❌ | `POST` | 请求方法 |
| `params` | ❌ | `{}` | 请求参数 |
| `headers` | ❌ | — | 请求级别自定义头，与全局 headers 合并 |
| `api_key` | ❌ | — | 请求级别 API 密钥，覆盖全局 |
| `timeout` | ❌ | 全局值 | 请求级别超时时间 |
| `async` | ❌ | `false` | 设为 `true` 启用异步轮询模式 |
| `task_id_field` | ❌ | `data.task_id` | 从提交响应中提取 task_id 的字段路径 |
| `poll_url` | ❌ | — | 轮询地址，支持 `{task_id}` 占位符 |
| `poll_method` | ❌ | `GET` | 轮询请求方法 |
| `poll_params` | ❌ | `{}` | 轮询参数，支持 `{task_id}` 占位符 |
| `poll_interval` | ❌ | `3` | 轮询间隔（秒） |
| `max_poll_attempts` | ❌ | `20` | 最大轮询次数 |
| `condition_field` | ❌ | `data.status` | 轮询响应中的状态字段路径 |
| `expected_value` | ❌ | `completed` | 期望的完成状态值 |
| `task_id_placeholder` | ❌ | `{task_id}` | 占位符字符串 |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `responses` | STRING | 所有响应的 JSON 数组，按原始顺序排列 |
| `success_count` | INT | 成功请求数 |
| `fail_count` | INT | 失败请求数 |
| `errors` | STRING | 错误详情 JSON（key 为请求索引） |

---

## LLM 节点

### LLM 图像生成 (`LLMImageGenerate`)

调用兼容 OpenAI API 的 LLM 服务生成图像。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `prompt` | STRING | ✅ | — | 图像生成提示词 |
| `api_base` | STRING | ✅ | — | API 基础地址 |
| `model` | STRING | ❌ | — | 模型名称 |
| `auth_token` | STRING | ❌ | `""` | 认证令牌 |
| `headers` | STRING | ❌ | `""` | 自定义请求头 JSON |
| `image` | IMAGE | ❌ | — | 参考图片输入 |
| `image_urls` | STRING | ❌ | — | 图片 URL（每行一个） |
| `timeout` | INT | ❌ | `60` | 超时时间（秒） |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `text` | STRING | 文本响应 |
| `image` | IMAGE | 生成的图片张量 |
| `has_image` | BOOLEAN | 是否包含图片 |
| `request` | STRING | 请求 JSON |
| `response` | STRING | 响应 JSON |

---

### LLM 智能生成 (`LLMSmartGenerate`)

多模态 LLM 调用节点，支持文本、图片、视频、音频的输入和输出。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `api_base` | STRING | ✅ | — | API 基础地址 |
| `model` | STRING | ❌ | — | 模型名称 |
| `auth_token` | STRING | ❌ | `""` | 认证令牌 |
| `headers` | STRING | ❌ | `""` | 自定义请求头 JSON |
| `prompt` | STRING | ❌ | `""` | 文本提示词 |
| `image` | IMAGE | ❌ | — | 图片输入 |
| `image_urls` | STRING | ❌ | — | 图片 URL（每行一个） |
| `video` | VIDEO | ❌ | — | 视频输入 |
| `video_urls` | STRING | ❌ | — | 视频 URL（每行一个） |
| `audio` | AUDIO | ❌ | — | 音频输入 |
| `audio_urls` | STRING | ❌ | — | 音频 URL（每行一个） |
| `output_mode` | 枚举 | ❌ | `auto` | 输出模式：auto / text_only / image_only / video_only / audio_only / multimodal |
| `use_modalities` | BOOLEAN | ❌ | `True` | 是否在请求中设置 modalities 字段 |
| `sampling_mode` | 枚举 | ❌ | `default` | 采样模式 |
| `temperature` | FLOAT | ❌ | `1.0` | 温度参数 |
| `top_p` | FLOAT | ❌ | `1.0` | Top-P 参数 |
| `max_tokens` | INT | ❌ | `4096` | 最大输出 token 数 |
| `stream` | BOOLEAN | ❌ | `False` | 是否流式输出 |
| `extendParams` | STRING | ❌ | `""` | 扩展参数 JSON，可覆盖任意请求字段 |
| `timeout` | INT | ❌ | `60` | 超时时间（秒） |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `text` | STRING | 文本响应 |
| `image` | IMAGE | 图片张量 |
| `video` | VIDEO | 视频信息（URL 列表） |
| `audio` | AUDIO | 音频信息（URL 列表） |
| `has_image` | BOOLEAN | 是否包含图片 |
| `has_video` | BOOLEAN | 是否包含视频 |
| `has_audio` | BOOLEAN | 是否包含音频 |
| `request` | STRING | 请求 JSON |
| `response` | STRING | 响应 JSON |

---

### LLM 响应图片解析 (`LLMResponseImageParser`)

从 LLM API 的 JSON 响应中解析并下载图片。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `response` | STRING | ✅ | — | LLM API 返回的 JSON 响应字符串 |
| `timeout` | INT | ❌ | `60` | 下载图片的超时时间（秒） |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `text` | STRING | 提取的文本内容 |
| `image` | IMAGE | 解析出的图片张量 |
| `has_image` | BOOLEAN | 是否包含图片 |

---

### LLM 响应智能解析 (`LLMResponseSmartParser`)

从 LLM API 的 JSON 响应中智能解析文本、图片、视频、音频。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `response` | STRING | ✅ | — | LLM API 返回的 JSON 响应字符串 |
| `output_mode` | 枚举 | ❌ | `auto` | 输出模式 |
| `timeout` | INT | ❌ | `60` | 下载图片的超时时间（秒） |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `text` | STRING | 提取的文本内容 |
| `image` | IMAGE | 解析出的图片张量 |
| `video` | VIDEO | 视频信息 |
| `audio` | AUDIO | 音频信息 |
| `has_image` | BOOLEAN | 是否包含图片 |
| `has_video` | BOOLEAN | 是否包含视频 |
| `has_audio` | BOOLEAN | 是否包含音频 |

---

## 工具节点

### 文本拼接 (`MaletteTextConcatenate`)

将多段文本按指定分隔符拼接为一个字符串。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `text1` | STRING | ✅ | `""` | 第一段文本 |
| `text2` ~ `text5` | STRING | ❌ | `""` | 第二至第五段文本 |
| `separator` | 枚举 | ❌ | `newline` | 分隔符类型：newline / comma / space / comma_space / semicolon / pipe / custom |
| `custom_separator` | STRING | ❌ | `""` | 自定义分隔符（当 separator=custom 时使用） |
| `prefix` | STRING | ❌ | `""` | 每段文本的前缀 |
| `suffix` | STRING | ❌ | `""` | 每段文本的后缀 |
| `skip_empty` | BOOLEAN | ❌ | `True` | 是否跳过空文本 |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `text` | STRING | 拼接后的文本 |

---

### 图片批次合并 (`MaletteImageBatchMerge`)

将多个图片或图片批次合并为一个批次张量。

**输入参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `image1` | IMAGE | ✅ | — | 第一个图片或图片批次 |
| `image2` ~ `image5` | IMAGE | ❌ | — | 第二至第五个图片或图片批次 |

**输出：**

| 输出 | 类型 | 说明 |
|---|---|---|
| `images` | IMAGE | 合并后的图片批次 |
| `count` | INT | 合并后的图片总数 |

---

## 字段路径语法

多个节点支持通过**字段路径**从 JSON 响应中提取嵌套值，语法如下：

| 路径 | 说明 | 示例 JSON | 结果 |
|---|---|---|---|
| `status` | 顶层字段 | `{"status": "ok"}` | `"ok"` |
| `data.status` | 嵌套字段 | `{"data": {"status": "done"}}` | `"done"` |
| `items[0].id` | 数组索引 | `{"items": [{"id": 1}]}` | `1` |
| `data.results[2].value` | 混合路径 | — | 对应嵌套值 |

---

## License

MIT