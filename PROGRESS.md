# CreditBench RAG — 项目进展

> 最后更新：2026-02-25

---

## 数据全景

### OneDrive 数据目录
`/Users/wxn/Library/CloudStorage/OneDrive-共享的库-NationalUniversityofSingapore/Li Shuping - Data/`

| 文件 | 大小 | 说明 |
|------|------|------|
| `Company Information.xlsx` | — | 29,118 家公司基本信息 |
| `Credit Events.xlsx` | — | 40,936 条信用事件（违约/降级等，1981–2025） |
| `Macros.xlsx` | — | 宏观经济指标（1988–2025） |
| `risk_indicators.csv` | — | 公司风险指标（DTD/流动性等，1988–2003） |
| `mktcap.parquet` | — | 日频市值/价格/成交量（1968–2025，778万行） |
| `earnings_call_transcripts/SP1500_earnings_call_transcripts.parquet` | — | S&P1500 财报电话会议（2005–2026，521万行，1446家公司）|
| `earnings_call_transcripts/Credit_ECC_combined_renamed.parquet` | — | 信用事件公司的财报会议（2005–2026，372万行，3133家公司）|
| `earnings_call_transcripts/credit_transcript_missing.csv` | — | 补充缺失公司（1.75万行，8家公司）|
| `foundamentals/` (12个 parquet) | ~2.5 GB | 财务基本面数据（BS/IS/CF/CR等） |

---

## 已完成

### 基础设施
- [x] PostgreSQL 15 + pgvector Docker 容器 (`ghcr.io/xwang049/creditbench-db:latest`)
- [x] SQLAlchemy ORM 模型（`src/db/models.py`）
- [x] 数据库 session 管理（`src/db/session.py`）
- [x] 统一数据更新脚本（`scripts/update_data.py`）

### A — Fundamentals（本地 DuckDB 直查）
- [x] `src/db/duckdb_query.py` — DuckDB 读取 OneDrive parquet，支持 Azure Blob 和本地两种模式
- [x] 12 张 fundamentals 表注册为 DuckDB view（af/bs/cf/cr2/ind1–4/is\_/sard\_bs/sard\_cf/sard\_is）
- [x] 支持任意 SQL 查询，无需导入数据库

**字段文档**：`foundamentals/Field Descriptions.xlsx`（约 1.1GB cr2 表是最大的）

### B — Market Data（PostgreSQL）
- [x] `market_data` 表：778万行，1968-01-02 ~ 2025-06-30
- [x] `src/ingestion/load_market_data.py` — 增量加载（按 trading_date 去重）
- [x] UniqueConstraint `(u3_company_number, trading_date)`

### 静态结构化数据（PostgreSQL）

| 表 | 行数 | 时间范围 | 说明 |
|----|------|----------|------|
| `companies` | 29,118 | — | 公司基本信息 + 行业分类 |
| `industry_mapping` | 64 | — | BICS 行业层级 |
| `credit_events` | 40,936 | 1981–2025 | 违约/降级/破产等 |
| `macro_us` | 9,428 | 1990–2025 | 美股指数等 |
| `macro_bond_yields` | 9,252 | 1990–2025 | 美债收益率 |
| `macro_fx` | 12,868 | 1988–2025 | 汇率 |
| `macro_commodities` | 9,241 | 1990–2025 | 大宗商品 |
| `risk_indicators` | 1,048,575 | 1988–2003 | DTD/流动性/sigma 等（11,866家公司）|

- [x] 所有表增量加载（重复运行安全）
- [x] `scripts/update_data.py --static / --credit-events / --market-data`

### C — Transcript 向量数据库（进行中）
- [x] `transcript_chunks` 表设计完成（768维 vector，pgvector HNSW 索引）
- [x] `src/ingestion/load_transcripts.py` — 三源合并增量加载
  - 同时读取 SP1500 parquet、Credit_ECC parquet、missing CSV
  - 以 `TRANSCRIPTCOMPONENTID` 为全局唯一键，自动跨源去重
  - `u3_company_number` 通过 `ID_BB_COMPANY → companies.id_bb_company` 自动映射
  - 跳过 Operator 发言和 <50 字符短文本
  - Gemini `gemini-embedding-001` 768维 embedding
  - 自动 429 限速重试，支持 `--year-from` / `--year-to` 过滤
- [x] `src/rag/transcript_retriever.py` — 语义检索，支持按公司/年份/季度/发言人类型过滤
- [x] HNSW 索引 `(m=16, ef_construction=64)`
- [ ] **Embedding 尚未全量运行**（当前 462 行，测试数据）

**三源合并后数据规模：**
| 项目 | 数量 |
|------|------|
| 合并后唯一段落 | 844万 |
| 过滤后需 embed | **645万行** |
| 覆盖公司数 | ~4,500+ |
| 时间范围 | 2005–2026 |

### RAG 核心
- [x] `src/rag/sql_retriever.py` — Text-to-SQL RAG（基于结构化数据）
- [x] `src/rag/chain.py` — LangChain RAG chain
- [x] `src/rag/transcript_retriever.py` — Transcript 语义检索
- [x] `src/api/main.py` — FastAPI REST API

---

## TODO

### 立即需要做

- [ ] **运行 transcript embedding**
  - 决定年份范围（推荐先跑 2020–2026，约 117 万行，4-5小时）
  - 命令：`nohup python scripts/update_data.py --transcripts --year-from 2020 > transcripts.log 2>&1 &`
  - 全量（399万行）需 14-16 小时，可断点续跑

### 数据层

- [ ] **Fundamentals 探索**：尚未决定如何使用 12 张 parquet（cr2 最大 1.1GB）
  - 目前只有 DuckDB 直查，没有导入 PostgreSQL
  - 是否需要部分字段导入？用于风险建模？
- [ ] **risk_indicators 时间范围**：当前数据只到 2003 年，是否有更新的版本？
- [ ] **companies 只有美国**：`country_name` 都是 US，是否预期如此？

### RAG / API 层

- [ ] **Transcript RAG 集成**：将 `transcript_retriever` 接入 `chain.py` 和 API
- [ ] **混合检索**：结合 SQL 结构化查询 + transcript 语义搜索
- [ ] **API 端点**：为 transcript 搜索添加 `/search/transcripts` 端点

### 工程

- [ ] **更新 Docker 镜像**：将新表结构（`market_data`, `transcript_chunks`）同步到 `ghcr.io/xwang049/creditbench-db:latest`
- [ ] **更新数据库快照**：`data/creditbench_dump_*.sql` 已过期（2026-02-15）
- [ ] **清理旧代码**：`src/rag/embeddings.py` 全部是死代码，可删除
- [ ] **`src/rag/retriever.py`** 和 `chain.py` 引用了已删除的 `CreditEventEmbedding`，需检查

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据库 | PostgreSQL 15 + pgvector |
| ORM | SQLAlchemy 2.0 |
| 大文件查询 | DuckDB（直读 parquet） |
| Embedding 模型 | Gemini `gemini-embedding-001`（768维） |
| LLM | Claude（Anthropic API） |
| API | FastAPI |
| 数据读取 | pandas + pyarrow |
| 云存储（可选）| Azure Blob Storage |

## 关键配置（.env）

```
DATABASE_URL=postgresql://creditbench:creditbench@localhost:5432/creditbench
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...              # Gemini embedding
DATA_DIR=.../Li Shuping - Data  # OneDrive 挂载路径
AZURE_STORAGE_ACCOUNT=...       # 可选，用于 fundamentals 远程访问
```
