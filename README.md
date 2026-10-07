# Local Portfolio Market Risk Tool

一个纯本地的投资组合市场风险计算工具。**所有价格序列、持仓、模型参数和报告
均来自本地 CSV 文件或代码生成的合成数据**，不访问交易所、行情 API、数据库
或任何外部服务。

## 功能

- 多资产持仓与历史价格（本地 CSV）
- 简单收益率 / 对数收益率，带明确的缺失数据清洗规则
- Historical Simulation VaR 与 Expected Shortfall（可配置置信水平与观察窗口）
- 参数法（variance-covariance）VaR，基于资产协方差矩阵与组合权重
- Monte Carlo 风险估计（固定随机种子时结果完全可复现）
- 基础压力情景：资产价格按给定比例同时（`ALL`）或分别变化，输出组合损益
- 对样本不足、协方差异常、零价值组合、非法置信水平返回明确错误

## 安装与运行

依赖：Python 3.10+，`numpy`、`pandas`（测试需 `pytest`）。

```bash
# 生成本地示例数据（纯代码生成的合成数据）
python scripts/generate_sample_data.py

# 计算风险指标（结果打印到终端，无图表窗口）
python -m risk_tool --prices data/prices.csv --positions data/positions.csv \
    --confidence 0.99 --window 250 --return-type simple \
    --mc-sims 10000 --seed 42 --stress data/stress.csv
```

### CSV 格式

`prices.csv`：`date` 列 + 每个资产一列价格。
`positions.csv`：`asset,quantity`（数量可正可负，负值表示空头）。
`stress.csv`：`asset,shock`，`shock` 为相对变动（如 `-0.20` 表示 -20%），
`asset` 取 `ALL` 时表示对所有未被单独列出的资产施加该冲击。

## 风险定义与符号约定

- **收益率**：简单收益率 `r_t = P_t / P_{t-1} - 1`；对数收益率
  `r_t = ln(P_t / P_{t-1})`。价格必须严格为正。
- **组合 P&L**：`PnL_t = Σ_i V_i · r_{i,t}`，其中 `V_i` 为按最新价格计算的
  各资产持仓市值（带符号）。
- **权重**：`w_i = V_i / Σ_j |V_j|`（相对总绝对市值，即 gross exposure），
  多空组合下波动率与 VaR 均有定义。
- **VaR**：以**正数表示损失**。历史法 `VaR = -quantile(PnL, 1 - c)`；
  参数法 `VaR = (z_c · σ_p - μ_p) · V_gross`，其中
  `σ_p = sqrt(wᵀΣw)`，`z_c` 为标准正态分布的 `c` 分位数。
- **Expected Shortfall (ES)**：突破 VaR 条件下损失分布的均值，同样以正数
  表示损失，且恒有 `ES ≥ VaR`。

## 缺失数据处理规则

按以下顺序应用于价格数据：

1. 按日期排序，重复日期保留最后一条；
2. 缺失价格前向填充（沿用最近的有效价格）；
3. 填充后仍含 NaN 的行（即资产首个有效价格之前的起始缺口）整行删除，
   保证剩余每一行所有资产价格齐全。

## 模型假设

- 持仓在观察期内保持不变（静态组合）。
- 参数法与 Monte Carlo 均假设资产收益率服从（多元）正态分布，均值与协方差
  由历史样本估计（样本协方差，`ddof=1`）。
- Monte Carlo 通过协方差矩阵特征分解生成相关正态情景；传入固定 `--seed`
  时结果完全可复现。
- 历史模拟不假设任何分布，直接使用历史 P&L 的经验分位数。
- 压力测试为确定性情景：`PnL_i = V_i · shock_i`，未指定冲击的资产视为不变。

## 错误处理

| 异常 | 触发条件 |
| --- | --- |
| `InsufficientDataError` | 收益率/窗口样本不足（少于 2 个观测） |
| `CovarianceError` | 协方差矩阵非方阵、不对称或非半正定 |
| `ZeroValuePortfolioError` | 组合总绝对市值为零 |
| `InvalidConfidenceLevelError` | 置信水平不在开区间 (0, 1) 内 |
| `MissingDataError` | CSV 缺列、文件不存在或资产无价格数据 |

## 测试

```bash
python -m pytest tests/ -s -q
```

`-s` 让各测试把计算出的风险指标直接打印到终端（不依赖任何图表窗口）。
覆盖场景：单资产、多个相关资产、已知收益序列、historical VaR、ES、
Monte Carlo 确定性（固定种子两次运行结果一致）、压力测试、缺失数据规则
以及各类非法输入。

## 目录结构

```
risk_tool/            # 核心包
  data.py             # CSV 加载与缺失数据清洗
  returns.py          # 简单/对数收益率
  portfolio.py        # 持仓、市值、权重、P&L
  historical.py       # 历史模拟 VaR / ES
  parametric.py       # 参数法 VaR、组合波动率
  montecarlo.py       # Monte Carlo VaR / ES（可复现）
  stress.py           # 压力情景
  errors.py           # 领域异常
  __main__.py         # 终端 CLI
scripts/generate_sample_data.py   # 本地合成数据生成
tests/test_risk_tool.py           # 自动化测试
data/                             # 生成的示例 CSV
```
