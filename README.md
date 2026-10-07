# risk_tool — 本地投资组合市场风险计算工具

一个**纯本地**的市场风险计算工具：所有价格序列、持仓、模型参数与报告
只来自本地 CSV 文件或代码生成的数据。**不访问**交易所、行情 API、
数据库或任何外部服务。

## 安装与依赖

- Python ≥ 3.10，依赖 `numpy`、`pandas`（测试需要 `pytest`）。
- 无其他依赖；正态分布分位数使用标准库 `statistics.NormalDist`。

## 快速开始

```bash
# 1) 生成示例数据（固定种子，纯本地代码生成）
python scripts/generate_sample_data.py        # 写入 data/prices.csv, data/holdings.csv

# 2) 运行风险计算（全部指标在终端输出，无图表窗口）
python -m risk_tool.cli \
    --prices data/prices.csv --holdings data/holdings.csv \
    --confidence 0.99 --window 250 --seed 42 \
    --shock '*=-0.10' --shock GAMMA=0.05

# 3) 运行自动化测试（同样只在终端输出）
python -m pytest tests/ -v -s
```

## 输入格式

`prices.csv`：第一列为日期，其余每列一个资产的价格。

```csv
date,ALPHA,BETA,GAMMA
2023-01-02,100.12,99.87,100.31
```

`holdings.csv`：`asset,quantity` 两列；数量为负表示空头。

```csv
asset,quantity
ALPHA,120
BETA,40
GAMMA,-30
```

## 风险定义与符号约定

- **收益率**：简单收益 `r_t = P_t/P_{t-1} - 1`（`--returns simple`，默认）
  或对数收益 `r_t = ln(P_t/P_{t-1})`（`--returns log`）。
- **组合收益**：`r_p,t = Σ_i w_i · r_i,t`，权重 `w_i = q_i·P_i / V`
  按最新价格计算（短窗口风险计量的标准线性近似）。
- **符号约定**：VaR 与 ES 以**正数表示亏损金额**（货币单位）。
  99% VaR = 972 表示：以 99% 置信度，单期亏损不超过 972。
- **Historical VaR**：`VaR = -quantile(r_p, 1-c) · V`（线性插值分位数）。
- **Expected Shortfall (ES/CVaR)**：`ES = -E[r_p | r_p ≤ quantile] · V`，
  即超过 VaR 的平均亏损。
- **参数法 VaR**：假设组合收益服从正态分布，
  `VaR = (z_c·σ_p − μ_p) · V`，其中 `σ_p = √(w′Σw)`，`μ_p = w′μ`。
- **Monte Carlo**：从多元正态 `N(μ, Σ)` 抽取 `--simulations` 个收益情景，
  对组合 P&L 取经验分位数。`--seed` 固定时结果**完全可复现**。
- **压力测试**：`--shock ASSET=PCT` 对单个资产施加比例价格冲击；
  `--shock '*=PCT'` 对所有资产同时冲击；单资产条目覆盖 `*`。
  损益 `PnL_i = q_i · P_i · shock_i`（盈利为正）。

## 缺失数据规则

通过 `--missing` 选择，在计算收益率**之前**作用于价格面板：

- `drop`（默认）：任何资产含缺失值的日期整行删除，保持截面对齐；
  收益率跨越被删日期计算。
- `ffill`：缺失价格前向填充；序列开头仍无观测的行被删除。

价格为非正数（≤0）会直接报错，因为对数/比例收益无定义。

## 模型假设

1. 持有期内持仓数量不变（buy-and-hold），权重按最新价格固定。
2. 参数法与 Monte Carlo 假设资产收益联合正态，μ、Σ 由观察窗口内
   历史收益估计。
3. 历史模拟法不假设分布，但假设历史窗口的收益分布对未来有代表性。
4. 所有指标为单期（与价格序列同频率）风险度量，不做时间平方根外推。

## 错误处理

以下情况返回明确错误（CLI 退出码 2，库抛出对应异常）：

| 情况 | 异常 |
|---|---|
| 样本不足（< 2 个收益观测，或 MC < 100 次模拟） | `InsufficientDataError` |
| 协方差矩阵非对称 / 非半正定 / 含 NaN / 维度不匹配 | `CovarianceError` |
| 组合最新市值为零 | `ZeroValueError` |
| 置信水平不在开区间 (0, 1) | `InvalidConfidenceError` |
| 数据问题（空面板、非正价格、持仓与价格不匹配等） | `DataError` |

## 测试

```bash
python -m pytest tests/ -v -s
```

覆盖：单资产与已知收益序列、多个相关资产、historical VaR/ES、
参数法 VaR、Monte Carlo 固定种子确定性（及与参数法的收敛一致性）、
压力测试（同时/分别冲击）、非法置信水平、样本不足、协方差异常、
零价值组合、缺失数据规则，以及 CLI 端到端冒烟测试。
所有输出均在终端打印，不依赖任何图表窗口。

## 项目结构

```
risk_tool/
  cli.py        # 终端入口（python -m risk_tool.cli）
  io.py         # 本地 CSV 加载（唯一数据来源）
  returns.py    # 简单/对数收益率 + 缺失数据规则
  portfolio.py  # 持仓、市值、权重、组合收益
  risk.py       # historical / parametric / Monte Carlo VaR & ES
  stress.py     # 压力情景
  errors.py     # 领域异常
scripts/generate_sample_data.py  # 固定种子的示例数据生成
tests/          # pytest 自动化测试
```
