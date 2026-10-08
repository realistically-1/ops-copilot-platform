# ops-copilot-platform

线上故障诊断服务，当前版本 v0.2。接收微服务告警，返回故障根因判断、证据和处理建议。当前日志及服务信息来自静态样本，不代表真实生产数据；诊断需要配置可用的模型 API。

## 环境要求

Python 3.12+，可访问支持 Chat Completions、`response_format: json_object` 的 OpenAI-compatible API。容器启动需要 Docker。

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

也可使用 Conda：

```bash
conda create --prefix .venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -e '.[dev]'
```

## 环境变量

```bash
cp .env.example .env
```

在 `.env` 设置 `LLM_BASE_URL`（含 `/v1` 等供应商要求的路径）、`LLM_API_KEY`、`LLM_MODEL`。例如 OpenAI 的地址为 `https://api.openai.com/v1`；其他兼容供应商请填入其提供的地址及模型名称。其他可配置项见 `.env.example`。未配置 key 时服务可启动，诊断请求返回 503。

## 启动

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

接口文档：<http://127.0.0.1:8000/docs>。`/healthz` 为进程健康检查；`/readyz` 检查模型配置、目录、日志服务初始化情况和诊断容量，不测试供应商网络可用性。

服务使用单进程启动。诊断记录仅在该进程内保留，默认最多 500 条、结束后保留 1 小时；容量满时优先移除最早结束的记录。重启后记录清空。

Docker：

```bash
docker build -f docker/Dockerfile -t ops-copilot-platform:0.2 .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 ops-copilot-platform:0.2
```

## 测试

测试不使用真实模型 API，无需配置 key。

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## API 调用示例

以下时间覆盖当前静态日志样本：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/diagnosis \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: local-check-001' \
  -d '{
    "service": "payment-service",
    "environment": "prod",
    "alert": "HTTP 5xx error rate > 15%",
    "timestamp": "2026-10-08T04:05:00Z"
  }'
```

响应包含 `diagnosis_id`、`service`、`environment`、`created_at`、`execution` 和 `result`。`result` 包含 `root_cause`、`confidence`、`summary`、`severity`、`evidence`、`recommended_actions`、`data_gaps`、`affected_components`、`assessment` 和 `timeline`。错误响应为 `{"error":{"code":"...","message":"...","request_id":"...","details":[]}}`。

诊断请求可选 `options`，例如：

```json
{"lookback_minutes": 15, "lookahead_minutes": 5, "keywords": ["timeout"], "log_levels": ["ERROR"]}
```

时间选项限定查询窗口；关键词和级别提供初始调查提示。成功响应的 `Location` 指向诊断记录。已创建记录的失败请求会在错误对象中返回 `diagnosis_id`。未找到、已过期或重启前的记录返回 404。

## API 列表与查询示例

| 方法 | 路径 | 使用方式 |
| --- | --- | --- |
| POST | `/api/v1/diagnosis` | 提交告警并等待诊断完成 |
| GET | `/api/v1/diagnoses` | 按 service、environment、status、创建时间筛选，支持 offset/limit |
| GET | `/api/v1/diagnoses/{diagnosis_id}` | 查询状态、结果及执行信息 |
| GET | `/api/v1/services` | 按 environment、owner、keyword 筛选，支持 offset/limit |
| GET | `/api/v1/services/{service_name}` | 查询目录信息，可指定 environment |
| GET | `/api/v1/services/{service_name}/dependencies` | 查询指定 environment 的依赖登记信息 |
| POST | `/api/v1/logs/search` | 查询静态日志样本 |

```bash
curl 'http://127.0.0.1:8000/api/v1/diagnoses?status=succeeded&limit=10'
curl 'http://127.0.0.1:8000/api/v1/diagnoses/<diagnosis_id>'
curl 'http://127.0.0.1:8000/api/v1/services/payment-service?environment=prod'
curl -X POST http://127.0.0.1:8000/api/v1/logs/search \
  -H 'Content-Type: application/json' \
  -d '{
    "service": "payment-service",
    "environment": "prod",
    "start": "2026-10-08T03:50:00Z",
    "end": "2026-10-08T04:10:00Z",
    "keywords": ["database", "timeout"],
    "levels": ["ERROR"],
    "match_mode": "all",
    "order": "desc",
    "limit": 20
  }'
```

日志时间区间包含 start、不包含 end，最长 3 小时。`match_mode` 支持 `any` / `all`，关键词按子串匹配。统计、模式和属性分组仅覆盖返回的日志；`total` 表示过滤后的全部匹配数量。
