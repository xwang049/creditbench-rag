# CreditBench — 数据源全览

> 最后更新：2026-02-26

所有原始数据均来自 OneDrive 挂载目录：
```
/Users/wxn/Library/CloudStorage/OneDrive-共享的库-NationalUniversityofSingapore/Li Shuping - Data/
```
下文简记为 `$DATA_DIR`。

---

## 数据流向总览

| 源文件 | 去向 | 方式 |
|--------|------|------|
| `Company Information.xlsx` | PostgreSQL `companies` + `industry_mapping` | 全量加载 |
| `Credit Events.xlsx` | PostgreSQL `credit_events` | 增量加载 |
| `Macros.xlsx` | PostgreSQL `macro_us` / `macro_bond_yields` / `macro_fx` / `macro_commodities` | 全量加载 |
| `risk_indicators.csv` | PostgreSQL `risk_indicators` | 全量加载 |
| `mktcap.parquet` | PostgreSQL `market_data` | 增量加载（按日期去重） |
| `foundamentals/*.parquet` (12个) | DuckDB 直查（不导入 PostgreSQL） | **A — 直查** |
| `earnings_call_transcripts/*.parquet + *.csv` (3个) | PostgreSQL `transcript_chunks` + 768维向量 | **C — 向量** |

---

## A — Fundamentals（DuckDB 直查）

**路径：** `$DATA_DIR/foundamentals/`

12 个 parquet 文件，通过 DuckDB 注册为视图，支持任意 SQL 查询，**不导入 PostgreSQL**。

| 文件名 | 视图名 | 行数 | 主要内容 |
|--------|--------|------|---------|
| `af.parquet` | `af` | — | Annual Fundamentals |
| `bs.parquet` | `bs` | — | Balance Sheet |
| `cf.parquet` | `cf` | — | Cash Flow |
| `cr2.parquet` | `cr2` | ~970万 | Credit Ratios（最大，421列，~1.1GB） |
| `ind1.parquet` | `ind1` | — | Industry data 1 |
| `ind2.parquet` | `ind2` | — | Industry data 2 |
| `ind3.parquet` | `ind3` | — | Industry data 3 |
| `ind4.parquet` | `ind4` | — | Industry data 4 |
| `is_.parquet` | `is_` | — | Income Statement |
| `sard_bs.parquet` | `sard_bs` | — | SARD Balance Sheet |
| `sard_cf.parquet` | `sard_cf` | — | SARD Cash Flow |
| `sard_is.parquet` | `sard_is` | — | SARD Income Statement |

**公共 ID 字段（所有表共享）：**
- `SECURITY` — 证券标识符
- `TICKER` — 股票代码
- `ID_BB_COMPANY` — Bloomberg 公司 ID（可与 companies 表关联）
- `FISCAL_YEAR_PERIOD` — 财年期（e.g. `2023A`）

**字段文档：** `$DATA_DIR/foundamentals/Field Descriptions.xlsx`

**访问方式：** `src/db/duckdb_query.py`，支持本地 OneDrive 和 Azure Blob Storage 两种模式。

---

## B — 静态结构化数据（PostgreSQL）

### 1. Company Information.xlsx → `companies` + `industry_mapping`

**路径：** `$DATA_DIR/Company Information.xlsx`
**行数：** 29,118 家公司

**字段：**

| 源列名 | DB 列名 | 类型 | 说明 |
|--------|---------|------|------|
| `U3_COMPANY_NUMBER` | `u3_company_number` | INTEGER PK | 主键，内部公司编号 |
| `ID_BB_UNIQUE` | `id_bb_unique` | TEXT | Bloomberg 唯一标识 |
| `ID_BB_COMPANY` | `id_bb_company` | INTEGER | Bloomberg 公司 ID（外部关联键） |
| `Ticker` | `ticker` | TEXT | 股票代码 |
| `Company_name` | `company_name` | TEXT | 公司名称 |
| `Country_name` | `country_name` | TEXT | 国家（当前数据全为 US） |
| `SECURITY_TYPE` | `security_type` | TEXT | 证券类型 |
| `MARKET_STATUS` | `market_status` | TEXT | 市场状态 |
| `Prime_exchange` | `prime_exchange` | TEXT | 主要交易所 |
| `Domicile` | `domicile` | TEXT | 注册地 |
| `INDUSTRY_SECTOR_NUM` | `industry_sector_num` | INTEGER | 行业板块编号（→ industry_mapping） |
| `INDUSTRY_GROUP_NUM` | `industry_group_num` | INTEGER | 行业组编号 |
| `INDUSTRY_SUBGROUP_NUM` | `industry_subgroup_num` | INTEGER | 行业子组编号 |
| `id_isin` | `id_isin` | TEXT | ISIN 编号 |
| `ID_cusip` | `id_cusip` | TEXT | CUSIP 编号 |

Excel 第二个 sheet 为行业层级映射 → `industry_mapping` 表（64行）。

---

### 2. Credit Events.xlsx → `credit_events`

**路径：** `$DATA_DIR/Credit Events.xlsx`
**行数：** 40,936 条
**时间范围：** 1981–2025

**字段：**

| 源列名 | DB 列名 | 类型 | 说明 |
|--------|---------|------|------|
| `U3_COMPANY_NUMBER` | `u3_company_number` | INTEGER FK→companies | 公司编号 |
| `EVENT_DATE` | `event_date` | DATE | 事件发生日期 |
| `EVENT_TYPE` | `event_type` | TEXT/INTEGER | 事件类型（数值编码，54 种唯一值） |
| `EVENT_DESCRIPTION` | `event_description` | TEXT | 事件描述 |
| `RATING_BEFORE` | `rating_before` | TEXT | 事件前评级 |
| `RATING_AFTER` | `rating_after` | TEXT | 事件后评级 |
| `SOURCE` | `source` | TEXT | 数据来源 |

**注：** `event_type` 是数值编码（如 54、55、72、97 等），需参考 Bloomberg 文档解读。

---

### 3. Macros.xlsx → 4 张 macro 表

**路径：** `$DATA_DIR/Macros.xlsx`

| Sheet 名 | DB 表 | 行数 | 内容 |
|----------|-------|------|------|
| `Commodities` | `macro_commodities` | 9,241 | 大宗商品价格（1990–2025，73列） |
| `Gov Bond Yield` | `macro_bond_yields` | 9,252 | 各期限美债收益率（1990–2025，11列） |
| `US Macro / 其他` | `macro_us` | ~9,428 | 美股指数、CPI、失业率等（1990–2025） |
| `Fx Rate` | `macro_fx` | 12,868 | 主要货币对美元汇率（1988–2025，65列） |

所有 macro 表以 `date` 列为时间索引，宽表结构（每列一个指标）。

---

### 4. risk_indicators.csv → `risk_indicators`

**路径：** `$DATA_DIR/risk_indicators.csv`
**行数：** 1,048,575
**时间范围：** 1988–2003（⚠️ 仅到 2003 年）

**字段（14列）：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `Company_Number` | INTEGER | = `u3_company_number × 1000 + suffix`（整除 1000 得 u3） |
| `u3_company_number` | INTEGER FK→companies | 关联公司（导入时计算） |
| `Date` | DATE | 数据日期（月度频率） |
| `DTD` | FLOAT | Distance to Default |
| `Liquidity` | FLOAT | 流动性指标 |
| `Sigma` | FLOAT | 波动率 |
| `NetDebt_MktVal` | FLOAT | 净负债/市值 |
| `ShortTermLiab_MktVal` | FLOAT | 短期负债/市值 |
| 其他风险指标... | FLOAT | 共约 10 个风险比率 |

**关联公式：** `u3_company_number = Company_Number // 1000`
**覆盖公司数：** 11,866 家

---

### 5. mktcap.parquet → `market_data`

**路径：** `$DATA_DIR/mktcap.parquet`
**总行数：** 76,837,227 行（1968–2025，28,274 家公司）
**DB 当前行数：** ~778 万行（1,169 家公司，从快照导入，不完整）

**字段（6列）：**

| 源列名 | DB 列名 | 类型 | 说明 |
|--------|---------|------|------|
| `U3_company_number` | `u3_company_number` | INTEGER FK→companies | 公司编号 |
| `trading_date` | `trading_date` | DATE | 交易日期（唯一约束组合键） |
| `last_price` | `last_price` | FLOAT | 收盘价 |
| `shares_outstanding` | `shares_outstanding` | FLOAT | 流通股数 |
| `volume` | `volume` | FLOAT | 成交量 |
| `marketcap` | `marketcap` | FLOAT | 市值 |

**唯一约束：** `(u3_company_number, trading_date)`
**增量加载：** 按 `trading_date` 跳过已有记录（当前 DB 最新日期为基准）

⚠️ **已知问题：** DB 中只有 1,169 家公司（快照数据），而 parquet 有 28,274 家。需要全量重新加载以覆盖全部公司。

---

## C — Transcript 向量数据库（PostgreSQL + pgvector）

### 三个数据源合并 → `transcript_chunks`

**路径：** `$DATA_DIR/earnings_call_transcripts/`

| 文件 | 行数 | 列结构 | 公司数 | 说明 |
|------|------|--------|--------|------|
| `SP1500_earnings_call_transcripts.parquet` | 521万 | 未命名列 `_COL_*` | ~1,500 | S&P1500 成分股财报电话会议 |
| `Credit_ECC_combined_renamed.parquet` | 372万 | 命名列 | 3,133 | 信用事件公司财报会议 |
| `credit_transcript_missing.csv` | 17,513 | 命名列 | 8 | 补充缺失公司 |

**三源合并后：** 844万唯一段落（按 `TRANSCRIPTCOMPONENTID` 去重），过滤后约 **645万行** 需要 embedding。

---

**列映射（命名列文件统一映射）：**

| 源列名 | 标准化名 | DB 列名 | 说明 |
|--------|----------|---------|------|
| `TRANSCRIPTCOMPONENTID` | `paragraph_id` | `paragraph_id` BIGINT PK | 全局唯一段落 ID |
| `ID_BB_COMPANY` | `bb_company_id` | — | 用于查找 u3_company_number |
| `MOSTIMPORTANTDATEUTC` | `call_start` | `call_date` | 电话会议日期 |
| `FISCALYEAR` | `year` | `year` | 财年 |
| `FISCALQUARTER` | `quarter` | `quarter` | 季度（1–4） |
| `HEADLINE` | `event_title` | `event_title` | 会议标题 |
| `TRANSCRIPTCOMPONENTTYPENAME` | `utterance_type` | `utterance_type` | 发言类型 |
| `TRANSCRIPTPERSONNAME` | `speaker_name` | `speaker_name` | 发言人姓名 |
| `SPEAKERTYPENAME` | `speaker_type` | `speaker_type` | 发言人类型（Analyst/Executive/Operator） |
| `COMPONENTTEXT` | `text_content` | `text_content` | 发言正文（embedding 对象） |
| `TITLE` | `speaker_title` | `speaker_title` | 发言人职位 |

**SP1500 parquet 列名对应（`_COL_*` → 标准名）：**

| `_COL_*` | 标准名 |
|----------|--------|
| `_COL_17` | `paragraph_id` |
| `_COL_1` | `bb_company_id` |
| `_COL_7` | `call_start` |
| `_COL_9` | `year` |
| `_COL_10` | `quarter` |
| `_COL_6` | `event_title` |
| `_COL_13` | `utterance_type` |
| `_COL_14` | `speaker_name` |
| `_COL_15` | `speaker_type` |
| `_COL_18` | `text_content` |
| `_COL_19` | `speaker_title` |
| `_COL_20` | `u3_company_number` |

---

**`transcript_chunks` 表结构：**

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | BIGINT PK | 自增主键 |
| `paragraph_id` | BIGINT UNIQUE | 源数据唯一段落 ID |
| `u3_company_number` | INTEGER FK→companies | 公司编号（可空，部分无法映射） |
| `call_date` | DATE | 电话会议日期 |
| `year` | SMALLINT | 财年 |
| `quarter` | SMALLINT | 季度 |
| `event_title` | TEXT | 会议标题（截断至 500 字符） |
| `speaker_type` | VARCHAR(50) | 发言人类型 |
| `utterance_type` | VARCHAR(80) | 发言类型 |
| `speaker_name` | VARCHAR(200) | 发言人姓名 |
| `speaker_title` | VARCHAR(200) | 发言人职位 |
| `text_content` | TEXT | 发言正文 |
| `embedding` | VECTOR(768) | Gemini embedding 向量 |
| `embedding_model` | VARCHAR(100) | `gemini-embedding-001` |
| `embedded_at` | TIMESTAMP | embedding 时间 |

**过滤规则：**
- 跳过 `speaker_type = 'Operator'`（主持人报幕）
- 跳过 `len(text_content) < 50`（"Yes."、"Thank you." 等短句）

**Embedding 模型：** `gemini-embedding-001`（Google Gemini），768 维，免费层 100 RPM
**HNSW 索引：** `m=16, ef_construction=64`，余弦距离（`<=>` 运算符）

**公司 ID 关联：**
- SP1500 parquet 直接提供 `u3_company_number`
- ECC parquet 和 CSV 通过 `ID_BB_COMPANY → companies.id_bb_company` 查表映射

---

## 关键 ID 字段关联

```
companies.u3_company_number
    ↑ FK
credit_events.u3_company_number
risk_indicators.u3_company_number  (= Company_Number // 1000)
market_data.u3_company_number
transcript_chunks.u3_company_number

companies.id_bb_company
    ↑ 关联
transcript_chunks (ECC/CSV源) via ID_BB_COMPANY → bb_to_u3 map
fundamentals.ID_BB_COMPANY (DuckDB 直查时用于关联)
```

---

## 数据质量备注

| 问题 | 详情 |
|------|------|
| `risk_indicators` 时间截止 | 只到 2003 年，是否有更新版本待确认 |
| `market_data` 不完整 | DB 中只有 1,169 家公司（快照），parquet 有 28,274 家 |
| `companies` 仅 US | `country_name` 全为美国公司，是否预期如此待确认 |
| `event_type` 编码 | `credit_events.event_type` 为数值编码，需 Bloomberg 文档解读 |
| Transcript embedding 进度 | 当前 462 行（测试），全量 2020+ 约 170 万行，正在后台运行 |
