# CreditBench Experiment Guide

## 1. 数据来源结构

### PostgreSQL（Docker 容器）— 结构化数据

| 表名 | 内容 | 字段示例 |
|------|------|---------|
| `companies` | 公司主数据 | `u3_company_number=12345, ticker="ENRN US", company_name="Enron Corp", market_status="DLST", country_name="United States"` |
| `credit_events` | 违约 / 破产事件 | `u3=12345, announcement_date=2001-12-02, action_name="Bankruptcy Filing", subcategory="Chapter 11"` |
| `macro_us` | 宏观指标（月频） | `date=2020-06-01, sp500=3100, vix=28.1, unemployment=13.3, cpi=257` |
| `macro_bond_yields` | 美债收益率曲线 | `data_date=2020-06-01, us_2y=0.17, us_10y=0.65, us_30y=1.41` |
| `market_data` | 日频股价 / 市值 | `u3=12345, trading_date=2020-06-01, last_price=4.2, marketcap=500M, volume=1.2M` |
| `risk_indicators` | Merton 模型指标（**仅覆盖 1988–2003**） | `u3=12345, year=2001, month=6, dtd=1.2, sigma=0.45, m2b=0.8` |
| `transcript_chunks` | 财报电话会议文字 + embedding | `u3=12345, call_date=2001-10-16, speaker_name="Jeff Skilling", text_content="..."` |

### DuckDB（本地 Parquet）— 财务报表

| 文件/视图 | 内容 | 字段示例 |
|------|------|---------|
| `is_` | 利润表（Income Statement） | `FISCAL_YEAR_PERIOD="2020A", SALES_REV_TURN=100B, EBIT=1.5B, NET_INCOME=980M, IS_DILUTED_EPS=1.12` |
| `bs` | 资产负债表（Balance Sheet） | `FISCAL_YEAR_PERIOD="2020A", BS_TOT_ASSET=65B, BS_CASH_NEAR_CASH_ITEM=1.2B, BS_LT_BORROW=12B` |
| `cf` | 现金流量表（Cash Flow） | `FISCAL_YEAR_PERIOD="2020A", CF_CASH_FROM_OPER=2.1B, CF_CAP_EXPEND_INC_FIX_ASSET=0.8B` |

> Parquet 文件路径：`data/foundamentals/*.parquet`（本地）或 Azure Blob Storage（配置 `.env` 后自动切换）

---

## 2. 实验入口 → 出口

  credit_events 表                                                                                                                                             
    WHERE action_name IN ('Default Corp Action', 'Bankruptcy Filing')                                                                                          
    DISTINCT ON u3_company_number   ← 每家公司只取最早那次事件                                                                                                 
    ORDER BY announcement_date    

  companies 表
    WHERE market_status = 'ACTV'        ← 目前还活跃
      AND country_name = 'United States'
      AND u3 NOT IN (SELECT u3 FROM credit_events)  ← 从未有过任何 credit event
    ORDER BY RANDOM()                   ← 随机打乱
  TODO：embedding for scripts
  
```
入口: python -m src.benchmark.run_experiment [参数]
         │
         ▼
   build_cases()          ← 从 PostgreSQL 找所有违约事件，构建 Group 1 + Group 2
         │                    Group 1: credit_events 里 Default Corp Action / Bankruptcy Filing
         │                    Group 2: market_status=ACTV、无任何 credit_event 的美国公司
         ▼
   fetch_all()            ← 按 cutoff_date 从 PostgreSQL + DuckDB 取历史数据
         │                    as_of = cutoff_date，严格不泄露未来信息
         ▼
   build_prompt()         ← 匿名化（公司名/ticker → Company-{u3}）
         │                    + 年份模糊（2020-06-01 → Y0-06-01，保留 VIX/收益率数值）
         │                    + 拼成完整 LLM prompt
         ▼
   predict()              ← 调 Claude API，解析 JSON 输出
         │                    输出字段: probability_pct, confidence, key_risk_factors,
         │                              protective_factors, reasoning 
         #MCP
         ▼
   写入 cases.jsonl       ← 每条 case 一行，实时 append（支持 --resume 断点续跑）
         │
         ▼
   evaluate()             ← AUC-ROC、Accuracy@50、混淆矩阵、各组平均概率

出口: results/exp_h{horizon}_lb{lookback}/
        ├── cases.jsonl           ← 每行一个 case 的完整记录
        └── results_summary.json  ← 汇总指标
```

**`cases.jsonl` 每行格式：**
```json
{
  "u3_company_number": 12345,
  "company_name": "Enron Corporation",
  "cutoff_date": "2001-06-01",
  "event_date": "2001-12-02",
  "label": true,
  "group": "default",
  "prediction": {
    "probability_pct": 73,
    "confidence": "medium",
    "key_risk_factors": ["negative operating cash flow", "high leverage"],
    "protective_factors": ["diversified revenue"],
    "reasoning": "..."
  },
  "prompt_chars": 3420
}
```

**`results_summary.json` 格式：**
```json
{
  "n_total": 200,
  "n_default": 100,
  "n_control": 100,
  "n_parsed": 195,
  "n_error": 5,
  "auc_roc": 0.72,
  "accuracy_at_50": 0.68,
  "avg_prob_default_group": 61.3,
  "avg_prob_control_group": 38.7,
  "confusion_matrix": {"TP": 62, "TN": 71, "FP": 29, "FN": 33},
  "config": {
    "prediction_horizon_months": 6,
    "lookback_months": 12,
    "llm_model": "claude-sonnet-4-6"
  }
}
```

---

## 3. 运行指令

### 前置条件
```bash
# 1. 启动 PostgreSQL
docker start creditbench-postgres

# 2. .env 里需要有：
#    ANTHROPIC_API_KEY=sk-ant-...
#    DATABASE_URL=postgresql://creditbench:creditbench@localhost:5432/creditbench
#    DATA_DIR=./data   （parquet 文件目录，可选）
```

### 常用指令

**① Dry-run（验证数据能跑通，不调 LLM）**

Dry-run 会对前 3 个 case 真正跑 `fetch_all()` + `build_prompt()`，确认数据管道无误。

```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --n-default 10 --n-control 10 \
  --dry-run
```

**② 小批量测试（真实调 LLM，快速验证）**
```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --n-default 10 --n-control 10 \
  --output results/test_run
```

**③ 正式实验（默认参数）**
```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --horizon 6 \
  --lookback 12 \
  --n-default 100 \
  --n-control 100 \
  --output results/exp_baseline
```

**④ 断点续跑（网络中断后继续）**
```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --horizon 6 --n-default 100 --n-control 100 \
  --output results/exp_baseline \
  --resume
```

**⑤ 不用 Parquet（本地没有 fundamentals 文件时）**
```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --n-default 100 --n-control 100 \
  --no-fundamentals
```

**⑥ 加 debug 日志（看每步发生什么）**
```bash
.venv/bin/python -m src.benchmark.run_experiment \
  --n-default 5 --n-control 5 -v
```

### 所有参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--horizon` | `6` | 预测窗口（月）：LLM 预测 cutoff 后 n 个月内是否违约 |
| `--lookback` | `12` | 历史回溯窗口（月）：LLM 能看到的数据范围 |
| `--n-default` | `100` | Group 1 最大案例数（违约公司） |
| `--n-control` | `100` | Group 2 最大案例数（对照公司） |
| `--model` | `claude-sonnet-4-6` | Claude 模型 ID |
| `--output` | `results/exp_h{horizon}_lb{lookback}` | 输出目录 |
| `--no-fundamentals` | off | 跳过 DuckDB parquet（本地无文件时使用） |
| `--resume` | off | 断点续跑，跳过 `cases.jsonl` 里已有的 case |
| `--dry-run` | off | 只建 cases、取数，不调 LLM |
| `--seed` | `42` | 随机种子（控制 Group 2 采样的可重复性） |
| `-v` | off | 启用 DEBUG 日志 |

### 只看评估结果（已有 JSONL）
```bash
.venv/bin/python -m src.benchmark.evaluator results/exp_baseline/cases.jsonl
```
