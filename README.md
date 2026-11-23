# 企业 KOL 智能识别、风控与营销预算分配系统

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41%2B-FF4B4B?logo=streamlit&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.x-2C6E49)
![Plotly](https://img.shields.io/badge/Plotly-WebGL-3F4F75?logo=plotly&logoColor=white)

> 用图论与社交网络分析（SNA），在**投放之前**回答两个问题：
> 谁是真 KOL？谁的流量是假的、还会暴雷？

传统 KOL 选号看粉丝数、互动量，这些指标都能刷。本项目换一个视角：**把账号之间的互动关系看成一张图**，用中心性衡量真实影响力，用拓扑结构识别作弊团伙，用跨期度数衰减预警"掉粉暴雷"，最后把预算按真实影响力分配。

---

## 功能一览

| 模块 | 做法 |
|---|---|
| **KOL 识别** | 度中心性 / 中介中心性 / 特征向量中心性 / 聚类系数四维刻画，加权打分排序 |
| **互刷群识别** | 高度抱团（聚类系数高）+ 不在关键路径上（中介中心性趋低） |
| **机器粉识别** | 有向图互惠性 ≈ 0（只被关注、不回关）+ 自身不获任何权重传导 |
| **暴雷预警** | 按时间窗口切分，度数衰减率 > 15% 判定为掉粉 |
| **预算分配** | 剔除异常节点后取 Top 10，按综合得分比例切分总预算 |
| **双轨界面** | Tab1 新品牌冷启动（内置模拟器，可复现作弊注入）｜Tab2 老品牌真实 CSV 分析 |

---

## 快速开始

```bash
git clone https://github.com/<你的用户名>/KOL-Decision-System.git
cd KOL-Decision-System

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

---

## 项目结构

```
KOL-Decision-System/
├── app.py                  # 主应用（单文件：算法 + 双 Tab UI + 可视化）
├── convert.py              # 数据转换：SNAP 原始 txt -> 系统可读 CSV
├── requirements.txt
├── data/
│   ├── facebook_combined.txt    # SNAP ego-Facebook 原始边表（4,039 节点 / 88,234 边）
│   └── real_facebook_data.csv   # 转换后的测试数据（source,target）
└── README.md
```

`convert.py` 按脚本所在位置解析路径，在任意目录下执行都不会找不到文件：

```bash
python convert.py            # 默认 data/facebook_combined.txt -> data/real_facebook_data.csv
python convert.py --input xxx.txt --output data/yyy.csv
```

---

## 算法设计

### 1. 图构建与中心性

`analyze_graph(edges_df, total_budget, is_directed, time_col)` 是唯一的核心入口，返回固定列序：

```
['Node', 'Degree', 'Betweenness', 'Eigenvector', 'Clustering', 'Status', 'Score', 'Budget']
```

- `is_directed=True` → `nx.DiGraph`（能看到"只出不进"），否则 `nx.Graph`
- 节点数 > `BETWEEN_SAMPLE_N`（默认 400）时，中介中心性自动切换为 **k 采样近似**，1000 节点全流程 < 1s，避免 `O(n³)` 卡死页面

### 2. 三道风控规则

| 规则 | 判定条件 | 标签 |
|---|---|---|
| 互刷群 | `Clustering > 0.80` **且** `Betweenness < 0.01` **且** `Degree ≥ 4` | 互刷水军 |
| 机器粉 | 有向图 **且** `互惠性 < 0.05` **且** `Eigenvector < 1e-3` | 机器粉 |
| 暴雷 | `(期间1度数 − 期间2度数) / 期间1度数 > 0.15` **且** `期间1度数 ≥ 8` **且** 净流失 ≥ 3 | ⚠️ 高风险暴雷 |

互惠性 = `|后继 ∩ 前驱| / 度数`，即"双向边占该节点全部连接边"的比例。

三个门槛（`POD_MIN_DEG`、`RISK_MIN_DEG`、`RISK_MIN_LOSS`）是**抗小样本误判**用的：如果不加它们，"只有 2 个互相认识的邻居"的普通小号会因聚类系数 = 1.0、中介中心性 = 0 被判成水军；"昨天 1 个粉丝、今天 0 个"的账号会被当成掉粉 100%。

### 3. 综合打分与预算

```
Score = 0.3 · norm(Degree) + 0.4 · norm(Betweenness) + 0.3 · norm(Eigenvector)
norm(x) = x / max(x)        # 注意：不是 Min-Max
```

这里刻意用 **Max 归一化**而非 Min-Max：星型网络里 99 个叶子节点度数完全相同，Min-Max 会把它们全部压成 0，导致预算表只剩 1 行。Max 归一化不改变排序，但保留了长尾节点的区分度。

判定为水军 / 暴雷的节点 **Score 强制归零**，不参与分配。剩余正常节点取 Top 10，按得分占比切分预算：

```
Budget_i = total_budget × Score_i / Σ Score(Top10)
```

### 4. 可视化降维（三层防护）

真实社交图动辄上千节点、上万条边，直接扔给浏览器必崩：

1. **节点抽样**：只画度数 Top 150
2. **边抽样**：硬上限 5,000 条 + 自适应软上限（每人 8~12 条），按 `deg(u) + deg(v)` 降序保留骨架
3. **WebGL 渲染**：`go.Scattergl` 画在 canvas 上，DOM 元素从 **6,089 个 `<path>` 降到 61 个**，帧间隔从 372ms 降到 56ms

布局对抗稠密图的两处补丁：分连通分量单独布局后紧凑拼装（避免两个社群被推成两极、主体缩成小斑点）、2%~98% 稳健包围盒等比归一 + 轻量宽高比补偿。

---

## 阈值参数表

全部集中在 `app.py` 顶部常量区，改数字即可调参，无需翻业务代码：

| 参数 | 默认 | 含义 |
|---|---|---|
| `CLUSTER_TH` / `BETWEEN_TH` | 0.80 / 0.01 | 互刷群判定线 |
| `POD_MIN_DEG` | 4 | 互刷群最小度数，置 0 即回到纯公式 |
| `RECIP_TH` / `EIGEN_TH` | 0.05 / 1e-3 | 机器粉判定线 |
| `RISK_DROP_TH` | 0.15 | 掉粉率阈值 |
| `RISK_MIN_DEG` / `RISK_MIN_LOSS` | 8 / 3 | 暴雷判定的小样本门槛 |
| `BETWEEN_SAMPLE_N` | 400 | 超过多少节点启用中介中心性采样 |
| `VIZ_MAX_NODES` / `VIZ_MAX_EDGES` | 150 / 5000 | 绘图抽样上限 |

---

## 输入数据格式

Tab2 上传的 CSV 至少需要两列，列名会自动识别常见变体（`source/target`、`from/to` 等）：

```csv
source,target,time_period
KOL_A,KOL_B,2024-01
KOL_B,KOL_C,2024-01
```

- 有 `time_period` 列 → 自动启用**暴雷风控**（按期间切分统计度数）
- 无时间列 → 只做水军过滤与打分排序

上传后强制执行四步清洗，防止内存溢出：

```
dropna → 去自环 → drop_duplicates
      → 端点合并 value_counts() 取 Top-800 核心节点
      → 只保留两端都在核心圈层内的边
```

页面会提示实际效果，例如 `原始数据 88,234 条边，清洗后降维至 xx 条边，核心节点 800 个`。

---

## 内置测试数据

Tab2 勾选「载入内置示例数据」即可零依赖跑通全链路。示例数据刻意注入了三类作弊模式，用于验证规则是否命中：

| 注入模式 | 构造方式 | 期望识别 |
|---|---|---|
| 互刷群 | 20 节点全连接，两期都刷 | 20 个「互刷水军」 |
| 买粉大V | 1 个核心节点引出 20 条单向边 | 20 个「机器粉」（假粉侧） |
| 掉粉暴雷 | 期间1 有 40 个粉丝互动，期间2 仅剩 1 个 | 1 个「⚠️ 高风险暴雷」 |

Tab1 的模拟器同样可复现：阶段 1 星型 100 节点 → 阶段 2 增至 500 并裂变 8 个社群 → 阶段 3 精确 1000 节点，其中强制注入 20 节点全连接互刷群 + 1 个买粉大V。

---

## ⚠️ 使用注意（踩过的坑）

**1. 无向图请务必关闭「按有向图分析」**

ego-Facebook 是无向好友图。若勾选有向模式，绝大多数路径断裂 → 中介中心性几乎全为 0 → 命中「聚类高 + 中介低」规则 → **800 个节点里 793 个被判成水军**，图全变灰、预算表近乎为空。取消后恢复正常（590 正常 / 210 水军）。

**2. 稠密真实社群上，水军规则先天偏严**

Facebook ego 网络平均聚类系数本就在 0.6 左右，稠密图里 782/800 个节点的中介中心性都 < 0.01。因此 26% 的判定率不代表真有 26% 水军。真实业务里建议把 `BETWEEN_TH` 改成**相对阈值**（如 `0.1 × 最大中介中心性`），或对无向稠密图只看得分排序。页面在水军占比 > 40% 时会自动弹出提示条，并提供一键切换按钮。

**3. 中介中心性是主要耗时项**

88k 边数据上传 → 出图约 9.9s，其中 6.3s 花在 `betweenness_centrality(k=400)`。想更快，把 `BETWEEN_SAMPLE_N` 调到 200，耗时约减半，Top10 排名基本不变。

**4. 模拟数据不启用暴雷风控**

Tab1 未传入 `time_col`：冷启动是持续增长场景，按期间切片会把正常核心 KOL 误判成"掉粉"。暴雷检测只在 Tab2 传入时间列时生效。

---

## 数据集来源

测试数据来自 Stanford SNAP 实验室公开数据集 **ego-Facebook**：

> J. McAuley and J. Leskovec. *Learning to Discover Social Circles in Ego Networks.* NIPS, 2012.
> https://snap.stanford.edu/data/ego-Facebook.html

原始数据为匿名化的 Facebook 好友关系图，含 4,039 个节点、88,234 条无向边。

代码可自由参考与修改；`data/` 目录内的数据版权归 Stanford SNAP 所有，仅限研究 / 教学用途。
