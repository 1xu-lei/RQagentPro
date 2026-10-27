# RQagentPro — 衡阳燃气 AI 智能客服系统 v3.1

基于 **DeepSeek + FAISS + BM25** 的 RAG 智能客服，采用 **ReAct Agent 架构**，支持流式对话、知识图谱、调用链追踪。

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue 3 + Naive UI + SSE 流式渲染 |
| 后端框架 | FastAPI + Uvicorn (异步) |
| LLM 引擎 | DeepSeek V4 Flash (Function Calling) |
| 嵌入模型 | BGE-base-zh-v1.5 + PCA 256d 降维 |
| 向量检索 | FAISS (内积搜索) |
| 关键词检索 | BM25 + jieba 分词 |
| 融合排序 | RRF (Reciprocal Rank Fusion) |
| 数据存储 | SQLite (单文件, FTS5 全文索引) |
| 知识图谱 | NetworkX + vis.js 可视化 |
| 调用链追踪 | 自研 contextvars Span 追踪 + Langfuse 兼容 |

## 架构

```
Interface (FastAPI) → Application (ChatService) → Domain (Agent/Tools/Retriever) → Infrastructure (LLM/Embedding/Storage)
```

```
用户问题
  → 安全网关 (关键词 + 标签双层防御)
  → ReAct Agent (DeepSeek 推理 → 工具调用 → 观察 → 再推理)
  → 混合检索 (FAISS 256d + BM25 + RRF)
  → 法规追溯 + 知识图谱
  → 流式输出 (SSE 逐 token)
```

## 快速开始

### 1. 环境要求
- Python >= 3.10
- 4GB+ 内存 (BGE 模型 ~400MB)

### 2. 安装
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

### 4. 准备模型和数据
```bash
# 下载 BGE 模型到 models/bge-base-zh-v1.5/
# 知识库自动从 data/kb.sqlite3 加载（首次自动迁移自 JSON）
```

### 5. 启动
```bash
python main.py
```

服务启动后访问:
- 对话界面: http://localhost:8001
- 调用链追踪: http://localhost:8001/api/traces
- 知识图谱: http://localhost:8001/graph
- API 文档: http://localhost:8001/docs

## 项目结构

```
RQagentPro/
├── main.py                         # 启动入口
├── requirements.txt                # Python 依赖
├── .env / .env.example / .gitignore
│
├── config/                         # 配置
│   ├── config.yaml                 #   主配置 (LLM / Agent / 检索 / 安全 / 会话)
│   └── prompts/                    #   提示词模板
│       ├── system.yaml             #     Agent 系统提示词
│       ├── intent.yaml             #     意图分类 Few-Shot 示例
│       └── safety.yaml             #     安全关键词 / 模板 / 过滤规则
│
├── src/                            # 核心源码 (分层架构)
│   ├── interface/                  #   接口层 — FastAPI 路由 / 中间件 / 请求模型
│   │   ├── api.py                  #     应用工厂：路由注册、启动事件、健康检查
│   │   ├── routes/                 #     chat / session / knowledge / company
│   │   ├── schemas/                #     Pydantic 模型
│   │   └── middleware/             #     错误处理 / 请求日志 (含 Span 追踪入口)
│   ├── application/                #   应用层 — 业务编排
│   │   ├── chat_service.py         #     对话编排：安全预检 → Agent → 校验
│   │   └── session_service.py      #     会话管理
│   ├── domain/                     #   领域层 — 核心业务逻辑
│   │   ├── taxonomy.py             #     标签体系 (12域×65标签，唯一真相源)
│   │   ├── agent/                  #     ReAct Agent 引擎 + Prompt 管理
│   │   ├── tools/                  #     7个 Agent 工具 (安全/检索/分类/法规/信息/推荐/转人工)
│   │   ├── retriever/              #     FAISS + BM25 + RRF 混合检索 + 查询改写
│   │   ├── safety/                 #     安全网关 (关键词 + 标签双层防御)
│   │   ├── knowledge_graph/        #     NetworkX 知识图谱查询
│   │   └── validator/              #     输出校验
│   └── infrastructure/             #   基础设施层
│       ├── llm/                    #     DeepSeek 异步客户端 (httpx + 指数退避 + SSE 流式)
│       ├── embedding/              #     BGE 编码 (PyTorch + PCA 256d + LRU 缓存)
│       ├── storage/                #     SQLite 知识库存储 + JSON 文件存储
│       ├── config/                 #     YAML 配置加载
│       ├── logging/                #     结构化日志 (JSON / Console 双模式)
│       └── tracing/                #     Span 调用链追踪 (contextvars + 树形渲染)
│
├── data/                           # 数据
│   ├── kb.sqlite3                  #   知识库 (1204条，SQLite WAL模式)
│   ├── indexes/                    #   FAISS + BM25 索引
│   ├── graph_export.json           #   知识图谱数据 (2513节点/6736边)
│   ├── graph_viz.html              #   知识图谱可视化 (vis.js)
│   ├── augmented_queries.json      #   BGE 增强查询
│   ├── logs/ & traces/             #   运行时日志 + 调用链 Trace
│   └── 企业*.docx/png              #   企业资料 (company 路由使用)
│
├── web/                            # 前端 (Vue3 + Naive UI + SSE 流式)
├── eval/                           # 评估系统 (5项指标)
├── models/bge-base-zh-v1.5/        # BGE 模型 + PCA 矩阵
├── tests/                          # 集成测试 (12项)
│
├── README.md                       # 项目概述
└── 迭代升级.md                      # 12项关键改进记录 (v2.0→v3.1)
```

## 核心特性

- **双层安全防御**: 关键词匹配 + 8 个安全敏感标签（仅即时生命危险场景），召回率 100%，精确率 100%
- **混合检索**: FAISS 向量 + BM25 关键词 + RRF 融合 + 意图域加权
- **PCA 降维**: BGE 768d → 256d，方差保留 96.2%，检索质量零损失
- **流式输出**: SSE 逐 token 推送，Agent 思考过程实时可见
- **知识图谱**: 2513 节点/6736 边，支持法规追溯和关联推荐
- **调用链追踪**: 每次请求的完整 Agent 决策树可视化 (`/api/traces`)
- **标签体系对齐**: 12域×65子标签，编码规范 (BZ-01/JF-01...)，旧域名自动映射

## API 端点

| 端点 | 方法 | 说明 |
|------|:--:|------|
| `/` | GET | 前端对话界面 |
| `/api/v1/chat` | POST | 非流式对话 |
| `/api/v1/chat/stream` | POST | SSE 流式对话 |
| `/api/v1/sessions` | GET | 历史会话列表 |
| `/api/v1/history/{id}` | GET | 会话详情 |
| `/api/v1/tags` | GET | 标签体系树 |
| `/api/v1/graph/summary` | GET | 知识图谱概览 |
| `/api/v1/graph/regulation/{fg}` | GET | 法规追溯 |
| `/api/v1/company/profile` | GET | 公司概述 |
| `/api/v1/company/culture` | GET | 企业文化 |
| `/api/health` | GET | 健康检查 |
| `/api/debug` | GET | 组件诊断 |
| `/api/traces` | GET | 调用链追踪面板 |
| `/api/traces/json` | GET | 调用链 JSON |
| `/graph` | GET | 知识图谱可视化 |
| `/docs` | GET | Swagger API 文档 |

## 评估指标

### v3.1 当前 (1204条KB, 259条测试集)

| 指标 | 结果 | 目标 | 状态 |
|------|:----:|:----:|:--:|
| 意图分类准确率 | **80%** | ≥85% | 接近 |
| 安全检测召回率 | **100%** | ≥95% | ✅ |
| 检索 Top-1 命中 | **81%** | ≥60% | ✅ |
| 检索 Top-3 召回 | **94%** | ≥80% | ✅ |
| 自动应答率 | **96%** | ≥85% | ✅ |

### 对比 v2.0 基准 (1003条KB, 303条测试集)

| 指标 | v2.0 | v3.1 | 提升 |
|------|:----:|:----:|:--:|
| 意图分类准确率 (各类别平均) | 56% | **80%** | +24pp |
| 检索 Top-1 命中率 | 73% | **81%** | +8pp |
| 自动应答率 | 84% | **96%** | +12pp |
| 投诉建议分类准确率 | 42% | **86%** | +44pp |

> 详细迭代记录见 [`迭代升级.md`](迭代升级.md)

## 文档

- [README.md](README.md) — 项目概述
- [迭代升级.md](迭代升级.md) — 11 项关键改进的完整记录
