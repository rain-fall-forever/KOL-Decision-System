# -*- coding: utf-8 -*-
"""
企业 KOL 智能识别、风控与营销预算分配系统
========================================================
双轨制应用：
  Tab 1  新品牌冷启动模拟（内置图生成器，星型 -> 社群 -> 千节点 + 作弊注入）
  Tab 2  老品牌真实数据分析（CSV 上传 + 强制清洗降维）

核心：analyze_graph() —— 图构建 / 中心性 / 防作弊（互惠性）/ 水军过滤 / 暴雷风控 / 预算分配
绘图：统一使用 plotly.graph_objects（不使用 pyvis，避免 iframe 白屏）

运行：streamlit run app.py
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import time

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale


# ---- 兼容新旧 Streamlit：新版用 width="stretch"，旧版用 use_container_width ----
def _stretch_kwargs(fn) -> Dict[str, object]:
    try:
        import inspect
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


W_DF = _stretch_kwargs(st.dataframe)
W_PLOT = _stretch_kwargs(st.plotly_chart)

# Plotly 前端配置：关掉不必要的工具栏按钮（每个按钮都是一次 DOM/JS 初始化）
PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d",
                               "toggleSpikelines", "hoverClosestCartesian",
                               "hoverCompareCartesian"],
}


# ---- 稳定的元素 key + 表格渲染兜底（旧版 Streamlit 的 st.dataframe 不支持 key） ----
def show_df(df: pd.DataFrame, key: str, **kwargs) -> None:
    """带唯一 key 的表格渲染；不支持 key 的旧版本自动降级，不会抛 TypeError。"""
    kwargs.setdefault("hide_index", True)
    kwargs.update(W_DF)
    try:
        st.dataframe(df, key=key, **kwargs)
    except TypeError:
        st.dataframe(df, **kwargs)


def _inject_css() -> None:
    """
    样式注入改为 st.html。

    【关键】不要用 st.markdown(html, unsafe_allow_html=True) 注入 <style>：
    浏览器会把 <style> 标签提升到 <head>，React 卸载该节点时子节点已不在原父节点下，
    于是抛出 `NotFoundError: Failed to execute 'removeChild' on 'Node'`。
    st.html 走独立容器渲染，不参与 Markdown 的 DOM 差异比对。
    """
    css = """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """
    try:
        st.html(css)          # Streamlit >= 1.33
    except Exception:
        pass                  # 老版本放弃美化，绝不用 unsafe_allow_html 兜底


# ==============================================================================
# 0. 全局常量与阈值（集中管理，便于业务调参）
# ==============================================================================
EPS = 1e-9                  # 浮点零阈值
CLUSTER_TH = 0.80           # 互刷群：聚类系数下限
BETWEEN_TH = 0.01           # 互刷群：中介中心性上限
POD_MIN_DEG = 4             # 互刷群：最小度数（度数<4 时三角关系极易凑满，易误判；置 0 即回到纯公式）
RECIP_TH = 0.05             # 买粉号：互惠性上限
EIGEN_TH = 1e-3             # 买粉号：特征向量中心性趋近于 0
RISK_DROP_TH = 0.15         # 暴雷：度数衰减率阈值（>15%）
RISK_MIN_DEG = 8            # 暴雷：期间1 最小度数（样本量门槛，避免长尾噪声）
RISK_MIN_LOSS = 3           # 暴雷：期间1->2 至少净流失的边数（防小样本比率放大）
BETWEEN_SAMPLE_N = 400      # 节点数超过该值时，中介中心性改用 k 采样近似（防卡死）
VIZ_MAX_NODES = 150         # 网络图最多绘制的节点数（度数 Top-N）
VIZ_MAX_EDGES = 5000        # 网络图最多绘制的边数硬上限（超过则按端点度数和抽样）
VIZ_EDGE_PER_NODE = 12      # 自适应软上限：每个可见节点最多保留 12 条边（150 节点 ≈ 1800 条）
VIZ_DENSE_EDGE_PER_NODE = 8 # 稠密图（真实社交网络）更激进：每人 8 条边，只看骨架
VIZ_AUTO_OFF_EDGES = 200000 # 清洗后边数超过该值才默认不绘图（仅极端数据量会触发）

STATUS_OK = "正常"
STATUS_POD = "互刷水军"
STATUS_BOT = "机器粉"
STATUS_RISK = "⚠️ 高风险暴雷"
WATER_ARMY = {STATUS_POD, STATUS_BOT}

RESULT_COLS = ["Node", "Degree", "Betweenness", "Eigenvector",
               "Clustering", "Status", "Score", "Budget"]


# ==============================================================================
# 1. 通用工具函数
# ==============================================================================
def _empty_result() -> pd.DataFrame:
    """返回空的标准化结果表，保证下游列结构永远一致（防 KeyError 崩溃）。"""
    return pd.DataFrame(columns=RESULT_COLS)


def _normalize(series: pd.Series) -> pd.Series:
    """
    Max 归一化（x / max）：结果落在 [0, 1]，max 恒为 1。

    为什么不用 Min-Max：星型网络里 99 个叶子度数完全相同，Min-Max 会把它们
    全部压成 0，导致 Top10 只剩 1 行、预算表失效。Max 归一化对这种“同质长尾”
    更稳健，同时不改变排序，符合“归一化”口径。
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    s = s.clip(lower=0.0)
    hi = float(s.max()) if len(s) else 0.0
    if not np.isfinite(hi) or hi < EPS:
        return pd.Series(0.0, index=s.index)
    return s / hi


def _sort_periods(values: pd.Series) -> List:
    """对时间列取值排序：优先数值序，退化为自然序，再退化为字符串序。"""
    vals = pd.Series(values.dropna().unique())
    if vals.empty:
        return []
    try:
        numeric = pd.to_numeric(vals, errors="raise")
        return vals.iloc[numeric.argsort(kind="mergesort")].tolist()
    except Exception:
        pass
    try:
        return sorted(vals.tolist())
    except Exception:
        return sorted(vals.astype(str).tolist())


def _period_degree(sub: pd.DataFrame) -> Dict[str, int]:
    """
    统计某一时间窗口内每个节点的“期间度数”。
    说明：按边出现次数计数（有向图 = 出度+入度），非累计口径，
         这样才能捕捉“掉粉 / 连接衰减”的真实信号。
    """
    c: Counter = Counter()
    if sub.empty:
        return dict(c)
    c.update(sub["source"].astype(str))
    c.update(sub["target"].astype(str))
    return dict(c)


def build_graph(edges_df: pd.DataFrame, is_directed: bool = False) -> nx.Graph:
    """根据边表构建图，自动去自环；构建失败返回空图而非抛异常。"""
    G = nx.DiGraph() if is_directed else nx.Graph()
    try:
        if edges_df is None or edges_df.empty:
            return G
        pairs = edges_df[["source", "target"]].astype(str)
        G.add_edges_from(zip(pairs["source"], pairs["target"]))
        G.remove_edges_from(list(nx.selfloop_edges(G)))
    except Exception:
        return nx.DiGraph() if is_directed else nx.Graph()
    return G


# ==============================================================================
# 2. 预算分配（抽出来，便于“算法缓存 + 预算实时联动”）
# ==============================================================================
def _allocate_budget(res: pd.DataFrame, total_budget: float) -> pd.DataFrame:
    """取得分最高的 10 个真实节点，按得分比例分配总预算。"""
    res["Budget"] = 0.0
    try:
        if total_budget is None or total_budget <= 0 or res.empty:
            return res
        candidates = res[res["Status"] == STATUS_OK].head(10)
        if candidates.empty:
            return res
        total_score = float(candidates["Score"].sum())
        if total_score > EPS:
            alloc = total_budget * candidates["Score"] / total_score
        else:
            alloc = pd.Series(total_budget / len(candidates), index=candidates.index)
        res.loc[candidates.index, "Budget"] = alloc.values
    except Exception:
        pass
    return res


# ==============================================================================
# 3. 全局核心算法模块
# ==============================================================================
def analyze_graph(edges_df: pd.DataFrame,
                  total_budget: float = 0.0,
                  is_directed: bool = False,
                  time_col: Optional[str] = None) -> pd.DataFrame:
    """
    图分析主函数。

    参数
    ----
    edges_df    : 至少包含 source / target 两列
    total_budget: 待分配总预算（元）
    is_directed : True 构建 DiGraph（可算互惠性、识别买粉号）
    time_col    : 时间列名，存在则启用“暴雷风控”

    返回
    ----
    DataFrame['Node','Degree','Betweenness','Eigenvector','Clustering','Status','Score','Budget']
    """
    # ---------- 3.0 入参校验（任何异常都返回空标准表，绝不让页面崩） ----------
    if edges_df is None or not isinstance(edges_df, pd.DataFrame) or edges_df.empty:
        return _empty_result()
    if not {"source", "target"}.issubset(set(edges_df.columns)):
        return _empty_result()

    try:
        use_time = bool(time_col) and time_col in edges_df.columns
        keep_cols = ["source", "target"] + ([time_col] if use_time else [])
        df = edges_df[keep_cols].copy()
        df["source"] = df["source"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()
        df = df[(df["source"] != "") & (df["target"] != "")]
        df = df[df["source"] != df["target"]]
        if df.empty:
            return _empty_result()
    except Exception:
        return _empty_result()

    # ---------- 3.1 构建图 ----------
    try:
        G = nx.DiGraph() if is_directed else nx.Graph()
        G.add_edges_from(zip(df["source"], df["target"]))
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        if G.number_of_nodes() == 0:
            return _empty_result()
    except Exception:
        return _empty_result()

    nodes = list(G.nodes())
    n = len(nodes)

    # ---------- 3.2 中心性计算（逐项 try-except，单项失败降级为 0 而非整体崩溃） ----------
    # 2.1 度数（原始度数，用于展示与节点大小）
    try:
        degree = {nd: int(G.degree(nd)) for nd in nodes}
    except Exception:
        degree = {nd: 0 for nd in nodes}

    # 2.2 中介中心性（大图用 k 采样近似，O(k*m)，防止 1000+ 节点卡死页面）
    betweenness: Dict[str, float] = {}
    try:
        if n > 300:
            k = min(BETWEEN_SAMPLE_N, n)
            betweenness = nx.betweenness_centrality(G, k=k, seed=42, normalized=True)
        else:
            betweenness = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        try:
            betweenness = {nd: 0.0 for nd in nodes}
        except Exception:
            return _empty_result()

    # 2.3 特征向量中心性（幂迭代可能不收敛，多级降级）
    eigen: Dict[str, float] = {}
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except Exception:
        try:
            eigen = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eigen = {nd: 0.0 for nd in nodes}

    # 2.4 聚类系数（networkx 对 DiGraph 计算的是有向聚类系数，双向完全图仍为 1.0）
    try:
        clustering = nx.clustering(G)
    except Exception:
        clustering = {nd: 0.0 for nd in nodes}

    # ---------- 3.3 防作弊：互惠性（仅 DiGraph） ----------
    # 互惠性 = 双向边节点数 / 总度数
    reciprocity: Dict[str, float] = {nd: 1.0 for nd in nodes}
    if is_directed:
        for nd in nodes:
            try:
                deg = G.degree(nd)
                if not deg:
                    reciprocity[nd] = 0.0
                    continue
                mutual = len(set(G.successors(nd)) & set(G.predecessors(nd)))
                reciprocity[nd] = mutual / deg
            except Exception:
                reciprocity[nd] = 0.0

    # ---------- 3.4 暴雷风控：期间1 vs 期间2 度数衰减 ----------
    risk_nodes: set = set()
    if use_time:
        try:
            tmp = df.dropna(subset=[time_col]).copy()
            tmp["source"] = tmp["source"].astype(str)
            tmp["target"] = tmp["target"].astype(str)
            periods = _sort_periods(tmp[time_col])
            if len(periods) >= 2:
                split = max(1, len(periods) // 2)
                early_vals, late_vals = periods[:split], periods[split:]
                d1 = _period_degree(tmp[tmp[time_col].isin(early_vals)])
                d2 = _period_degree(tmp[tmp[time_col].isin(late_vals)])
                for nd, deg1 in d1.items():
                    if deg1 < RISK_MIN_DEG:
                        continue
                    deg2 = d2.get(nd, 0)
                    if (deg1 - deg2) < RISK_MIN_LOSS:
                        continue
                    if (deg1 - deg2) / deg1 > RISK_DROP_TH:
                        risk_nodes.add(nd)
        except Exception:
            risk_nodes = set()

    # ---------- 3.5 组装结果 + 打标签 ----------
    try:
        res = pd.DataFrame({
            "Node": [str(nd) for nd in nodes],
            "Degree": [degree.get(nd, 0) for nd in nodes],
            "Betweenness": [float(betweenness.get(nd, 0.0)) for nd in nodes],
            "Eigenvector": [float(eigen.get(nd, 0.0)) for nd in nodes],
            "Clustering": [float(clustering.get(nd, 0.0)) for nd in nodes],
        })
    except Exception:
        return _empty_result()

    status_list = []
    for nd in nodes:
        clu = float(clustering.get(nd, 0.0))
        btw = float(betweenness.get(nd, 0.0))
        eig = float(eigen.get(nd, 0.0))
        rec = float(reciprocity.get(nd, 1.0))
        deg = int(degree.get(nd, 0))

        status = STATUS_OK
        # 规则A｜互刷群：高度抱团（聚类系数高）且不在关键路径上
        if clu > CLUSTER_TH and btw < BETWEEN_TH and deg >= POD_MIN_DEG:
            status = STATUS_POD
        # 规则B｜买粉号：只出不进（互惠性≈0）且自身不获任何权重
        elif is_directed and rec < RECIP_TH and eig < EIGEN_TH:
            status = STATUS_BOT
        # 规则C｜暴雷：仅在未被判定为水军时覆盖标签，保证水军人数统计不失真
        if nd in risk_nodes and status == STATUS_OK:
            status = STATUS_RISK
        status_list.append(status)
    res["Status"] = status_list

    # ---------- 3.6 综合打分：0.3 度 + 0.4 中介 + 0.3 特征向量 ----------
    try:
        res["Score"] = (0.3 * _normalize(res["Degree"])
                        + 0.4 * _normalize(res["Betweenness"])
                        + 0.3 * _normalize(res["Eigenvector"]))
    except Exception:
        res["Score"] = 0.0

    # 作弊 / 风险节点得分强制归零
    res.loc[res["Status"] != STATUS_OK, "Score"] = 0.0
    res["Score"] = res["Score"].fillna(0.0)

    # ---------- 3.7 排序 + 预算分配 ----------
    res = res.sort_values(["Score", "Degree"], ascending=[False, False]).reset_index(drop=True)
    res = _allocate_budget(res, total_budget)

    return res[RESULT_COLS]


# ==============================================================================
# 4. Tab1：新品牌冷启动 —— 图数据模拟器
# ==============================================================================
def generate_simulation(step: int, seed: int = 42) -> pd.DataFrame:
    """
    生成 3 个演化阶段的有向边表（含 period 列）。

    step=1 : 100 节点，星型（1 个核心 + 99 个叶子，双向关注）
    step=2 : +400 节点 -> 500，形成 8 个小社群 + 跨界桥接
    step=3 : +500 节点 -> 1000，其中强制注入
              · 20 节点封闭互刷群（内部全连接）
              · 1 个买粉大V + 20 条单向假粉边
    """
    rng = random.Random(seed * 100 + step)
    directed_edges: List[Tuple[str, str, int]] = []   # 单向边
    bidir_pairs: set = set()                          # 双向边（去重）
    max_step = int(step)

    def link(a: str, b: str, period: int) -> None:
        """双向关注（互惠），带 period 标记。"""
        if a == b or period > max_step:
            return
        bidir_pairs.add((min(a, b), max(a, b), period))

    def one_way(a: str, b: str, period: int) -> None:
        """单向边（用于买粉）。"""
        if a == b or period > max_step:
            return
        directed_edges.append((a, b, period))

    # ---------------- 阶段 1：100 节点星型网络 ----------------
    hub = "H_头部KOL"
    leaves = [f"T1_{i:03d}" for i in range(1, 100)]
    for lf in leaves:
        link(hub, lf, 1)

    # ---------------- 阶段 2：扩张至 500 节点，社群化 + 跨界 ----------------
    communities: Dict[int, List[str]] = {}
    if max_step >= 2:
        for c in range(8):
            members = [f"T2_C{c}_{i:02d}" for i in range(1, 51)]
            communities[c] = members
            # 社群内：环 + 随机双向边（制造高聚类的小世界结构）
            for i, m in enumerate(members):
                link(m, members[(i + 1) % len(members)], 2)
                for _ in range(2):
                    link(m, rng.choice(members), 2)
            # 社群领袖挂到核心 KOL，形成全局中心
            link(members[0], hub, 2)
        # 跨界连接：相邻社群之间架桥，制造“破圈”通道
        for c in range(8):
            for _ in range(3):
                a = rng.choice(communities[c])
                b = rng.choice(communities[(c + 1) % 8])
                link(a, b, 2)

    # ---------------- 阶段 3：扩张至 1000 节点 + 作弊注入 ----------------
    if max_step >= 3:
        fake_clique = [f"FAKE_POD_{i:02d}" for i in range(20)]        # 20 节点互刷群
        buy_core = "V_买粉大V"                                        # 买粉号核心
        fake_fans = [f"FAKE_FAN_{i:02d}" for i in range(20)]          # 20 个假粉

        # 1000 - 500 - 20 - 21 = 459 个正常新增节点
        new_normal = 459
        pool = [f"T3_N{i:03d}" for i in range(1, new_normal + 1)]
        for idx, nd in enumerate(pool):
            c = rng.choice(list(communities.keys()))
            # 每个新号挂 4 条社群内边：邻域足够稠密，避免“2 个互相认识的邻居”
            # 被聚类系数误判成互刷（也符合真实用户的关注广度）
            for _ in range(4):
                link(nd, rng.choice(communities[c]), 3)
            if idx % 20 == 0:  # 每 20 个节点制造一条跨社群边，模拟破圈传播
                link(nd, rng.choice(communities[(c + 3) % 8]), 3)

        # ---- 作弊注入 1：20 节点封闭互刷群（内部全连接，且不与外界相连）----
        for i in range(len(fake_clique)):
            for j in range(i + 1, len(fake_clique)):
                link(fake_clique[i], fake_clique[j], 3)

        # ---- 作弊注入 2：买粉大V（核心 1 条 -> 20 条单向边，零互惠、零被引用）----
        for fan in fake_fans:
            one_way(buy_core, fan, 3)

    # ---------------- 落表：双向边展开为两条有向边 ----------------
    rows: List[Tuple[str, str, int]] = []
    for a, b, p in bidir_pairs:
        rows.append((a, b, p))
        rows.append((b, a, p))
    rows.extend(directed_edges)

    return pd.DataFrame(rows, columns=["source", "target", "period"])


# ==============================================================================
# 5. 可视化：Plotly 网络图（Top-N 抽样，绝不整图渲染）
# ==============================================================================
def sample_edges_for_viz(sub: nx.Graph,
                         max_edges: int = VIZ_MAX_EDGES,
                         degree_map: Optional[Dict[str, int]] = None) -> List[Tuple[str, str]]:
    """
    边抽样（Edge Sampling）：边数超过 max_edges 时，按"端点度数和"降序只保留前 max_edges 条。

    为什么按端点度数和排序：一条边的两个端点度数越高，它越处于网络主干骨架上；
    保留它们等于保住核心拓扑（社群桥接、头部 KOL 之间的连接），被丢弃的只是长尾细枝，
    对"谁是关键 KOL"的判断几乎没有影响，但前端 SVG 图元数量会降一个量级。

    任何异常都退化为返回全部边，绝不因为抽样逻辑把图搞崩。
    """
    try:
        edges = list(sub.edges())
        if max_edges is None or max_edges <= 0 or len(edges) <= max_edges:
            return edges
        deg = degree_map or dict(sub.degree())
        scored = sorted(
            edges,
            key=lambda e: deg.get(str(e[0]), 0) + deg.get(str(e[1]), 0),
            reverse=True,
        )
        return scored[:max_edges]
    except Exception:
        try:
            return list(sub.edges())
        except Exception:
            return []


def _gl_or_svg(kind: str, **kwargs) -> go.BaseTraceType:
    """
    优先用 WebGL trace（Scattergl），失败退回 SVG（Scatter）。

    为什么必须 WebGL：go.Scatter 走 SVG，Plotly 会把每条折线/每个点都变成 DOM 节点。
    实测 ego-Facebook 的 Top-150 子图（5000 条边）一次性插入了 6089 个 <path>，
    浏览器主线程要 300~800ms 才能画完；而预算滑块每动一格就整页重跑、整图重画，
    叠几次就变成"此页面没有响应"。Scattergl 把整层压到 <canvas> 里，DOM 只有个位数。
    """
    try:
        return getattr(go, f"{kind}gl")(**kwargs)
    except Exception:
        return getattr(go, kind)(**kwargs)


def _lines_trace(ex, ey) -> go.BaseTraceType:
    """边层：单条 trace + None 分隔，WebGL 渲染。"""
    return _gl_or_svg(
        "Scatter", x=ex, y=ey, mode="lines", hoverinfo="skip",
        line=dict(width=0.6, color="rgba(148,163,184,0.42)"),
        showlegend=False,
    )


def _nodes_trace(xs, ys, sizes, colors, texts, hovers) -> go.BaseTraceType:
    """节点层：散点 + Top10 标签，WebGL 渲染。"""
    return _gl_or_svg(
        "Scatter", x=xs, y=ys, mode="markers+text",
        text=texts, textposition="top center",
        textfont=dict(size=9, color="#334155"),
        marker=dict(size=sizes, color=colors,
                    line=dict(width=0.6, color="rgba(255,255,255,0.9)")),
        hovertext=hovers, hoverinfo="text",
        showlegend=False,
    )


def _components(G: nx.Graph) -> List[set]:
    """取连通分量：DiGraph 必须用 weak 版本 —— connected_components 对有向图会直接抛
    NetworkXNotImplemented，异常若被外层吞掉就会出现"布局没生效但也不报错"的静默退化。"""
    try:
        if G.is_directed():
            return sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        return sorted(nx.connected_components(G), key=len, reverse=True)
    except Exception:
        return [set(G.nodes())]


def _layout_subgraph(sub: nx.Graph, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    按连通分量分别布局、再紧凑拼装。

    直接对整张子图跑 spring_layout 时，互不相连的社群会被斥力推到极远：
    实测 ego-Facebook 的 Top-150 子图是「123 人 + 27 人」两个社群，
    spring_layout 把它们甩到画布两端，各自缩成一团小斑点，中间一大片空白。
    这里先给每个分量单独布局并按规模缩放，再横向紧凑拼装，主拓扑才能填满画布。
    """
    comps = _components(sub)
    big = [c for c in comps if len(c) > 1][:4]
    singles = [n for c in comps if len(c) == 1 for n in c]

    pos: Dict[str, Tuple[float, float]] = {}
    if not big:
        return {n: (rng_float(i), rng_float(i + 3)) for i, n in enumerate(sub.nodes())}

    max_n = max(len(c) for c in big)
    radii = [0.62 * float(np.sqrt(len(c) / max_n)) for c in big]

    local_maps: List[Dict[str, Tuple[float, float]]] = []
    for comp, rad in zip(big, radii):
        sc = sub.subgraph(comp)
        lp = nx.spring_layout(sc, seed=seed, k=1.7 / np.sqrt(max(len(comp), 2)), iterations=70)
        arr = np.array([lp[n] for n in comp], dtype=float)
        arr -= arr.mean(axis=0)
        scale = float(np.max(np.abs(arr)))
        arr = arr / (scale if scale > EPS else 1.0) * rad
        local_maps.append({n: (float(a[0]), float(a[1])) for n, a in zip(comp, arr)})

    if len(big) == 1:
        pos.update(local_maps[0])
    else:                                  # 多社群：横向排列，间距按半径紧凑留白
        gap = 0.14
        width = sum(2 * r for r in radii) + gap * (len(radii) - 1)
        cursor = -width / 2
        for rad, lp in zip(radii, local_maps):
            cx = cursor + rad
            for nd, (px, py) in lp.items():
                pos[nd] = (px + cx, py)
            cursor += 2 * rad + gap

    # 孤立点（子图内度数为 0，边被抽样掉了）：沿底部摆成一行，不参与主体缩放
    for i, nd in enumerate(singles):
        fx = (i + 1) / (len(singles) + 1)
        pos[nd] = (-1.0 + 2.0 * fx, -1.05)
    return pos


def build_network_figure(G: nx.Graph,
                         res: pd.DataFrame,
                         max_nodes: int = VIZ_MAX_NODES,
                         title: str = "",
                         max_edges: int = VIZ_MAX_EDGES) -> Optional[go.Figure]:
    """仅抽取度数 Top-N 节点 + Top-M 边绘图：节点大小随度数、颜色随得分（黄->红）、水军灰、风险红。"""
    try:
        if G is None or G.number_of_nodes() == 0 or res is None or res.empty:
            return None

        # 5.1 节点采样：按度数取 Top-N，避免 spring_layout O(n^2) 卡死页面
        ranked = res.sort_values("Degree", ascending=False)["Node"].tolist()
        keep = [nd for nd in ranked if G.has_node(nd)][:max_nodes]
        if not keep:
            return None
        sub = G.subgraph(keep)

        # 5.2 布局：分社群布局 + 紧凑拼装（失败降级为环形布局）
        try:
            pos = _layout_subgraph(sub)
        except Exception:
            try:
                pos = nx.spring_layout(sub, seed=42, k=1.6 / np.sqrt(max(len(keep), 2)), iterations=80)
            except Exception:
                try:
                    pos = nx.circular_layout(sub)
                except Exception:
                    pos = {nd: (rng_float(i), rng_float(i + 1)) for i, nd in enumerate(sub.nodes())}

        # 5.2b 视图正视化：用 2%~98% 分位的稳健包围盒（等比、取两条轴较大跨度）居中缩放，
        #      把个别外挂节点夹到画布边缘，避免它把 150 个节点压成角落小团；
        #      必须等比缩放，否则两个社群会被拉成对角两极。
        #      统计分位时排除度数为 0 的孤立点（它们被统一摆在底部），否则会撑大包围盒。
        try:
            core = [nd for nd in pos if sub.degree(nd) > 0] or list(pos)
            px = np.array([pos[nd][0] for nd in core], dtype=float)
            py = np.array([pos[nd][1] for nd in core], dtype=float)
            x_lo, x_hi = np.percentile(px, [2, 98])
            y_lo, y_hi = np.percentile(py, [2, 98])
            span = float(max(x_hi - x_lo, y_hi - y_lo))
            if span > EPS:
                cx, cy = float((x_lo + x_hi) / 2), float((y_lo + y_hi) / 2)
                pos = {
                    nd: (float((p[0] - cx) / span * 2),
                         float((p[1] - cy) / span * 2))
                    for nd, p in pos.items()
                }

                # 宽高比补偿：画布约 1.75:1，而链路型 / 多社群图的横纵跨度往往差 2~4 倍，
                # 等比缩放会让内容变成一条扁带、上下大片留白。这里只把较短的一轴
                # 轻度拉伸（上限 1.45 倍，肉眼几乎察觉不到形变），让图铺满画布。
                try:
                    xs_ = np.array([p[0] for p in pos.values()], dtype=float)
                    ys_ = np.array([p[1] for p in pos.values()], dtype=float)
                    sx, sy = float(np.ptp(xs_)), float(np.ptp(ys_))
                    if sx > EPS and sy > EPS:
                        ratio = sx / sy
                        if ratio > 1.75:
                            k = min(1.45, ratio / 1.75)
                            pos = {nd: (p[0], p[1] * k) for nd, p in pos.items()}
                        elif ratio < 1 / 1.75:
                            k = min(1.45, 1.75 / ratio)
                            pos = {nd: (p[0] * k, p[1]) for nd, p in pos.items()}
                except Exception:
                    pass

                pos = {nd: (float(np.clip(p[0], -1.12, 1.12)),
                            float(np.clip(p[1], -1.12, 1.12)))
                       for nd, p in pos.items()}
        except Exception:
            pass

        info = res.set_index("Node")
        max_deg = max(float(info.loc[keep, "Degree"].max()), 1.0)
        max_score = max(float(res["Score"].max()), EPS)

        # 5.3 边抽样：Top-150 节点之间仍可能有上万条边（ego-Facebook 就有 7k+），
        #     直接铺上万个 SVG 图元会让浏览器假死，因此做两层裁剪：
        #     ① 硬上限 VIZ_MAX_EDGES 条；② 自适应软上限 = 每节点 12 条（稠密图 6 条）。
        #     ego-Facebook 这类"人均 30+ 边"的稠密图，150 节点 × 6 = 900 条边就够看清骨架了。
        all_edges = sample_edges_for_viz(sub, max_edges=None)      # 抽样前的边数（用于提示）
        degree_map = dict(zip(info.index.astype(str), info["Degree"].astype(int).tolist()))
        avg_deg = (2.0 * len(all_edges) / max(len(keep), 1))
        per_node = VIZ_EDGE_PER_NODE if avg_deg <= 20 else VIZ_DENSE_EDGE_PER_NODE
        edge_cap = min(max_edges or VIZ_MAX_EDGES, max(300, per_node * len(keep)))
        viz_edges = sample_edges_for_viz(sub, max_edges=edge_cap, degree_map=degree_map)

        # 节点半径按"子图真实稠密度"分档：用抽样前的 avg_deg，不能用抽样后的
        # （抽样后人均 8 条边会被误判成稀疏图，于是 150 个 40px 圆糊成一个实心球）
        if avg_deg <= 20:
            size_base, size_amp = 10, 32
        elif avg_deg <= 60:
            size_base, size_amp = 6, 14
        else:
            size_base, size_amp = 4, 9
        dense = avg_deg > 60
        sampled = len(viz_edges) < len(all_edges)

        # 只画"抽样后仍有连线"的节点：孤立点（邻居都不在 Top150、或边被抽样掉）画出来
        # 只是一颗悬空的点，会被误读成"孤立账号"，统一不画并在标题里说明数量
        drawn = [nd for nd in sub.nodes() if sub.degree(nd) > 0] or list(sub.nodes())

        ex, ey = [], []
        for u, v in viz_edges:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                ex += [x0, x1, None]
                ey += [y0, y1, None]

        fig = go.Figure()
        fig.add_trace(_lines_trace(ex, ey))

        # 抽样说明写进图标题（单行，避免 <br><sup> 双行标题与图重叠），
        # 详情由调用方在图下用 st.caption 补充
        if sampled:
            head = title if title else "网络关系图"
            title = (f"{head}｜⚠️ 边抽样 {len(viz_edges):,}/{len(all_edges):,} 条"
                     f"（按端点度数和保留核心骨架）")
        hidden = sub.number_of_nodes() - len(drawn)
        if hidden > 0:
            title = f"{title}｜{hidden} 个无可见连线的节点未绘制" if title else f"{hidden} 个无可见连线的节点未绘制"

        # 5.4 节点样式：灰（水军）/ 红（暴雷）/ 黄->红渐变（得分）
        xs, ys, sizes, colors, texts, hovers = [], [], [], [], [], []
        # 稠密图里 30 个标签会糊成一团，只标 Top5 + 少量异常节点
        n_label = 5 if dense else 10
        label_set = set(res.head(n_label)["Node"].tolist()) | \
                    set(res[res["Status"] != STATUS_OK]["Node"].tolist()[:8 if dense else 20])

        for nd in drawn:
            x, y = pos.get(nd, (0.0, 0.0))
            row = info.loc[nd]
            deg = float(row["Degree"])
            score = float(row["Score"])
            status = str(row["Status"])

            size = size_base + size_amp * (deg / max_deg) ** 0.5
            if status in WATER_ARMY:
                color = "#94a3b8"                                  # 水军强制灰
            elif status == STATUS_RISK:
                color = "#dc2626"                                  # 暴雷强制红
            else:
                ratio = min(max(score / max_score, 0.0), 1.0) ** 0.5
                color = sample_colorscale("YlOrRd", 0.12 + 0.83 * ratio)[0]

            xs.append(x); ys.append(y); sizes.append(size); colors.append(color)
            texts.append(str(nd) if nd in label_set else "")
            hovers.append(
                f"<b>{nd}</b><br>状态：{status}"
                f"<br>度数：{deg:.0f}"
                f"<br>中介中心性：{float(row['Betweenness']):.4f}"
                f"<br>特征向量：{float(row['Eigenvector']):.4f}"
                f"<br>聚类系数：{float(row['Clustering']):.3f}"
                f"<br>综合得分：{score:.4f}"
                f"<br>预算：¥{float(row['Budget']):,.0f}"
            )

        fig.add_trace(_nodes_trace(xs, ys, sizes, colors, texts, hovers))

        fig.update_layout(
            title=dict(text=title, x=0.01, font=dict(size=14, color="#0f172a")),
            height=620,
            margin=dict(l=10, r=10, t=52, b=10),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            hovermode="closest",
            dragmode="pan",
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
        )
        return fig
    except Exception:
        return None


def rng_float(i: int) -> float:
    """布局兜底用的确定性伪随机坐标。"""
    return ((i * 37) % 100) / 100.0


# ==============================================================================
# 6. 结果渲染：指标卡 + Top10 预算表 + 预算柱状图
# ==============================================================================
def _force_undirected() -> None:
    """按钮回调：把「按有向图分析」关掉（回调在脚本重跑前执行，改控件状态是安全的）。"""
    st.session_state["t2_directed"] = False


def render_metrics(res: pd.DataFrame, total_budget: float, n_edges: int, n_raw: Optional[int] = None):
    """顶部 KPI：总节点数 / 剔除水军数 / 高风险数 / 待分配总预算。"""
    total_nodes = 0 if res is None or res.empty else len(res)
    water = 0 if res is None or res.empty else int(res["Status"].isin(WATER_ARMY).sum())
    risk = 0 if res is None or res.empty else int((res["Status"] == STATUS_RISK).sum())
    allocated = 0.0 if res is None or res.empty else float(res["Budget"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总节点数", f"{total_nodes:,}", help="当前图中共有多少个可识别的 KOL / 账号节点")
    c2.metric("剔除水军数", f"{water:,}",
              help=f"互刷水军 + 机器粉合计，占全图 {water / total_nodes * 100:.1f}%" if total_nodes else None)
    c3.metric("高风险暴雷", f"{risk:,}", help="度数在期间2衰减超过 15% 的节点")
    c4.metric("待分配总预算", f"¥{allocated:,.0f}",
              help=f"预算池 ¥{total_budget:,.0f}，已按得分比例分配到 Top10 真实节点")


def render_budget_table(res: pd.DataFrame, total_budget: float, key_prefix: str = "t1"):
    """Top10 预算分配表 + 预算柱状图。key_prefix 保证两个 Tab 的元素 key 全局唯一。"""
    if res is None or res.empty:
        st.warning("没有可展示的节点数据。")
        return
    show = res[res["Budget"] > 0].copy()
    if show.empty:
        st.warning("未筛选出可投放的真实节点（全部被判定为水军 / 高风险，或得分为 0）。")
        return

    show.insert(0, "排名", range(1, len(show) + 1))
    budget_num = show["Budget"].astype(float).tolist()   # 柱状图用数值副本
    show["预算占比"] = (show["Budget"] / max(total_budget, EPS) * 100).round(2).astype(str) + "%"
    show["Degree"] = show["Degree"].astype(int)
    show["Betweenness"] = show["Betweenness"].round(4)
    show["Eigenvector"] = show["Eigenvector"].round(4)
    show["Clustering"] = show["Clustering"].round(3)
    show["Score"] = show["Score"].round(4)
    show["Budget"] = show["Budget"].map(lambda v: f"¥{v:,.0f}")

    st.markdown("##### 💰 Top 10 营销预算分配")
    show_df(
        show[["排名", "Node", "Degree", "Betweenness", "Eigenvector",
              "Clustering", "Score", "Budget", "预算占比"]],
        key=f"{key_prefix}_budget_tbl",
    )

    try:
        bar_df = show.iloc[::-1]
        # 节点 ID 可能是"纯数字字符串"（SNAP 数据集就是 2543 / 2266 这种），
        # Plotly 会自动把它推断成数值轴 → 10 根柱子被压成 10 条横线、纵轴变成 0~2500 坐标轴。
        # 因此必须显式声明 yaxis.type="category"（并把标签统一成 str）。
        labels = bar_df["Node"].astype(str).tolist()
        fig = go.Figure(go.Bar(
            x=budget_num[::-1],
            y=labels,
            orientation="h",
            marker=dict(color="#f97316"),
            text=bar_df["预算占比"].tolist(),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>预算 ¥%{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="预算分配（元）", x=0.01, font=dict(size=13, color="#0f172a")),
            height=max(280, 34 * len(bar_df) + 90),
            margin=dict(l=10, r=70, t=40, b=30),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            xaxis=dict(title="预算（元）", gridcolor="#e2e8f0",
                       rangemode="tozero", separatethousands=True),
            yaxis=dict(type="category", tickfont=dict(size=11), gridcolor="#f1f5f9"),
            bargap=0.28,
            showlegend=False,
        )
        st.plotly_chart(fig, key=f"{key_prefix}_budget_bar", config=PLOT_CONFIG, **W_PLOT)
    except Exception:
        st.info("预算柱状图渲染失败，表格数据不受影响。")


def render_status_detail(res: pd.DataFrame, key_prefix: str = "t1"):
    """水军 / 风控明细，便于人工复核。"""
    if res is None or res.empty:
        return
    bad = res[res["Status"] != STATUS_OK].copy()
    # 注意：expander 的 label 不做动态拼接（label 变化会让整个容器重新挂载），
    # 数量放进内部占位符里渲染，DOM 结构因此保持稳定。
    with st.expander("🔍 异常节点复核清单", expanded=False):
        slot = st.empty()
        with slot.container():
            if bad.empty:
                st.success("本次扫描未发现水军或高风险节点。")
                return
            st.caption(f"共 {len(bad)} 个异常节点，按度数降序展示（最多 200 行）")
            bad["Clustering"] = bad["Clustering"].round(3)
            bad["Betweenness"] = bad["Betweenness"].round(4)
            bad["Eigenvector"] = bad["Eigenvector"].round(5)
            show_df(
                bad[["Node", "Status", "Degree", "Clustering",
                     "Betweenness", "Eigenvector"]].head(200),
                key=f"{key_prefix}_bad_tbl",
            )


# ==============================================================================
# 7. 带缓存的分析入口（图计算很贵，缓存“指标”，预算单独实时分配）
# ==============================================================================
@st.cache_data(show_spinner=False, max_entries=8)
def _cached_core(edges_df: pd.DataFrame, is_directed: bool, time_col: Optional[str]) -> pd.DataFrame:
    return analyze_graph(edges_df, 0.0, is_directed=is_directed, time_col=time_col)


def run_analysis(edges_df: pd.DataFrame, total_budget: float,
                 is_directed: bool, time_col: Optional[str]) -> pd.DataFrame:
    """算法缓存（预算=0）+ 实时预算分配，拖动预算滑块不会重算中心性。"""
    core = _cached_core(edges_df, is_directed, time_col)
    if core is None or core.empty:
        return _empty_result()
    res = core.copy()
    return _allocate_budget(res, total_budget)


# ==============================================================================
# 8. Tab2：真实数据读取与清洗（严格防内存溢出）
# ==============================================================================
def load_csv(file) -> pd.DataFrame:
    """多编码尝试读取 CSV，全部失败时抛业务异常由上层 try 捕获。"""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            break
    raise ValueError(f"CSV 解析失败：{last_err}")


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """自动识别 source / target / time 列（兼容中英文常见命名）。"""
    lower = {str(c).strip().lower(): c for c in df.columns}
    src_keys = ["source", "src", "from", "from_node", "源", "源节点", "起点", "用户a", "user_a"]
    tgt_keys = ["target", "tgt", "to", "to_node", "目标", "目标节点", "终点", "用户b", "user_b"]
    time_keys = ["time_period", "period", "time", "date", "month", "期间", "时间", "时间窗口", "时间段"]

    src = next((lower[k] for k in src_keys if k in lower), None)
    tgt = next((lower[k] for k in tgt_keys if k in lower), None)
    tcol = next((lower[k] for k in time_keys if k in lower), None)
    return src, tgt, tcol


def clean_edges(df: pd.DataFrame, src: str, tgt: str, tcol: Optional[str],
                max_nodes: int = 800) -> Tuple[pd.DataFrame, int, int]:
    """
    强制清洗 + 降维（顺序不可调换）：
      1) dropna -> 2) 去自环 -> 3) 去重
      4) 端点合并 value_counts 取 Top-800 核心节点
      5) 只保留两端都在核心集合内的边
    返回 (清洗后边表, 原始边数, 核心节点数)
    """
    raw_n = len(df)
    cols = [src, tgt] + ([tcol] if tcol else [])
    d = df[cols].copy()
    d[src] = d[src].astype(str).str.strip()
    d[tgt] = d[tgt].astype(str).str.strip()

    # 1) 缺失值
    d = d.dropna(subset=[src, tgt])
    d = d[(d[src] != "") & (d[tgt] != "") & (d[src].str.lower() != "nan") & (d[tgt].str.lower() != "nan")]
    # 2) 自环边
    d = d[d[src] != d[tgt]]
    # 3) 重复边
    d = d.drop_duplicates(subset=cols)

    # 4) Pandas 级预降维：合并端点做度数统计，锁定核心圈层
    deg_rank = pd.concat([d[src], d[tgt]], ignore_index=True).value_counts()
    core = set(deg_rank.head(max_nodes).index)

    # 5) 双向过滤：任一端点在长尾直接丢弃
    d = d[d[src].isin(core) & d[tgt].isin(core)]
    return d.reset_index(drop=True), raw_n, len(core)


# ==============================================================================
# 9. 内置示例数据（让 Tab2 无需外部文件也能跑通全链路）
# ==============================================================================
def make_sample_edges() -> pd.DataFrame:
    """
    构造带 time_period 的示例边表：
      · 12 个正常社群（关系结构两期稳定，只随时间自然波动）
      · 20 节点封闭互刷群
      · 1 个买粉大V（20 条单向假粉边）
      · 1 个掉粉暴雷号（期间1 有 40 个真实互动，期间2 只剩 2 个）
    结构先建一次再打期次戳，避免随机噪声把“稳定节点”误判成掉粉。
    """
    rng = random.Random(7)
    periods = ["2024-01", "2024-02"]

    # ---- 1) 正常社群骨架（只生成一次，两期都存在）----
    communities: Dict[int, List[str]] = {}
    base_edges: List[Tuple[str, str]] = []
    for c in range(12):
        members = [f"KOL_C{c}_{i:02d}" for i in range(1, 26)]
        communities[c] = members
        for i, m in enumerate(members):
            base_edges.append((m, members[(i + 1) % len(members)]))
            base_edges.append((m, members[rng.randrange(len(members))]))
    # 相邻社群架桥，保证全图连通（否则特征向量中心性会被最大连通片垄断）
    for c in range(12):
        for _ in range(5):
            a = rng.choice(communities[c])
            b = rng.choice(communities[(c + 1) % 12])
            if a != b:
                base_edges.append((a, b))

    rows: List[Tuple[str, str, str]] = []
    for p in periods:
        for u, v in base_edges:
            if u != v:
                rows.append((u, v, p))   # 双向关注
                rows.append((v, u, p))

    # ---- 2) 互刷群：20 节点内部全连接，两期都刷 ----
    clique = [f"FAKE_POD_{i:02d}" for i in range(20)]
    for p in periods:
        for i in range(20):
            for j in range(i + 1, 20):
                rows.append((clique[i], clique[j], p))
                rows.append((clique[j], clique[i], p))

    # ---- 3) 买粉大V：只有出边，没有回关（互惠性 = 0）----
    for p in periods:
        for i in range(20):
            rows.append(("V_买粉大V", f"FAKE_FAN_{i:02d}", p))

    # ---- 4) 掉粉暴雷号：真实社群成员，期间1 被 40 人互动，期间2 只剩 2 人 ----
    champ = "KOL_C5_01"
    others = [m for c in communities for m in communities[c] if m != champ]
    fans = rng.sample(others, 40)
    for f in fans:
        rows.append((f, champ, periods[0]))
    for f in fans[:2]:
        rows.append((f, champ, periods[1]))

    return pd.DataFrame(rows, columns=["source", "target", "time_period"])


# ==============================================================================
# 10. 主程序
# ==============================================================================
def main() -> None:
    st.set_page_config(page_title="KOL风控决策系统", layout="wide",
                       initial_sidebar_state="expanded")

    _inject_css()
    st.title("🛰️ 企业 KOL 智能识别、风控与营销预算分配系统")
    st.caption("图论 + 社交网络分析（SNA）驱动：中心性识别真 KOL · 互惠性拆穿假流量 · 度数衰减预警暴雷")

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.header("🎛️ 控制台")

        st.subheader("🚀 冷启动模拟参数")
        step = st.slider("时间轴（演化阶段）", 1, 3, 3, 1,
                         help="1=百人星型起盘；2=社群化扩张破圈；3=千节点并注入作弊流量")
        budget_wan = st.slider("总预算（万元）", 10, 100, 50, 5)
        total_budget = float(budget_wan) * 10000.0

        st.divider()
        st.subheader("🏢 真实数据参数")
        uploaded = st.file_uploader(
            "上传 CSV（必含 source / target；含 time_period 自动启用暴雷风控）",
            type=["csv"], accept_multiple_files=False,
        )
        if "t2_directed" not in st.session_state:
            st.session_state["t2_directed"] = True
        directed_real = st.checkbox("按有向图分析（启用互刷 / 买粉检测）", key="t2_directed",
                                    help="关闭则退化为无向图，互惠性规则不生效。"
                                         "无向好友图（如 SNAP ego-Facebook）请务必关闭，否则会被大面积误判为机器粉。")

    tab1, tab2 = st.tabs(["🚀 新品牌冷启动模拟", "🏢 老品牌真实数据分析"])

    # ============================ Tab 1 ============================
    with tab1:
        st.markdown(f"#### 演化阶段 {step}｜预算池 ¥{total_budget:,.0f}")
        # 固定占位符：整个动态区域只挂在一个稳定父节点下，
        # 切换 Tab / 换时间轴时 React 只替换子节点，不会去 removeChild 错误的父节点
        body1 = st.empty()
        with body1.container():
            try:
                # 生成 -> 快照（保留当前阶段及以前的边）
                edges_all = generate_simulation(step, seed=42)
                edges_snap = edges_all[edges_all["period"] <= step][["source", "target", "period"]].copy()

                res = run_analysis(edges_snap, total_budget, is_directed=True, time_col=None)
                render_metrics(res, total_budget, len(edges_snap))

                if res.empty:
                    st.error("模拟图谱构建失败，请切换时间轴后重试。")
                else:
                    G = build_graph(edges_snap, is_directed=True)
                    c_l, c_r = st.columns([1, 3])
                    with c_l:
                        st.markdown("**图谱概览**")
                        st.write(f"- 节点数：**{G.number_of_nodes():,}**")
                        st.write(f"- 边数：**{G.number_of_edges():,}**")
                        st.write("- 图类型：**有向图 DiGraph**")
                        st.write(f"- 绘图抽样：**Top {VIZ_MAX_NODES} 度数节点**")
                        st.caption("节点大小∝度数；颜色：灰=水军，黄→红=得分，红=暴雷")
                    with c_r:
                        # 图表放在占位符里渲染，key 全局唯一且不随时间轴变化
                        chart1 = st.empty()
                        with chart1.container():
                            fig = build_network_figure(
                                G, res, VIZ_MAX_NODES,
                                title=f"KOL 关系网络（阶段 {step}｜抽样可视化）")
                            if fig is not None:
                                st.plotly_chart(fig, key="t1_network", config=PLOT_CONFIG, **W_PLOT)
                            else:
                                st.warning("网络图渲染失败（已降级），下方表格与指标不受影响。")

                    render_budget_table(res, total_budget, key_prefix="t1")
                    render_status_detail(res, key_prefix="t1")

                    st.caption("提示：本页为模拟冷启动场景，未传入时间列，故不触发“暴雷风控”；"
                               "在「老品牌真实数据分析」中上传含 time_period 的数据即可启用。")
            except Exception as e:
                st.error(f"模拟数据生成或计算异常：{type(e).__name__} - {e}")

    # ============================ Tab 2 ============================
    with tab2:
        st.markdown("#### 真实投放数据诊断")
        use_sample = st.checkbox("没有数据？载入内置示例数据先跑通流程", value=False)

        # 状态提示单独占位：success/info/error 反复替换也不会打乱同层兄弟节点
        msg_slot = st.empty()
        df_raw: Optional[pd.DataFrame] = None
        with msg_slot.container():
            try:
                if use_sample:
                    df_raw = make_sample_edges()
                    st.success(f"已载入示例数据：{len(df_raw):,} 条边（含 time_period）")
                elif uploaded is not None:
                    df_raw = load_csv(uploaded)
                    st.success(f"文件读取成功：{len(df_raw):,} 行 × {len(df_raw.columns)} 列")
            except Exception as e:
                st.error(f"文件读取失败：{e}")
                df_raw = None

        body2 = st.empty()
        with body2.container():
            if df_raw is None:
                st.info("请在左侧上传 CSV，或勾选上方「载入内置示例数据」。"
                        "文件需包含 `source`、`target` 两列，若含 `time_period` 将自动启用暴雷风控。")
                with st.expander("📄 数据格式示例", expanded=False):
                    st.code("source,target,time_period\nKOL_A,KOL_B,2024-01\nKOL_B,KOL_C,2024-02",
                            language="csv")
            else:
                # ---------- 列识别 ----------
                src, tgt, tcol = detect_columns(df_raw)
                if not src or not tgt:
                    st.error(f"未识别到 source / target 列。当前列名：{list(df_raw.columns)[:12]}")
                else:
                    st.caption(f"列识别结果：source=`{src}`，target=`{tgt}`，"
                               f"时间列=`{tcol}`" + ("（已启用暴雷风控）" if tcol else "（未启用暴雷风控）"))

                    # ---------- 清洗 + 强制降维 ----------
                    edges = pd.DataFrame()
                    try:
                        edges, raw_n, core_n = clean_edges(df_raw, src, tgt, tcol, max_nodes=800)
                        st.info(f"原始数据 {raw_n:,} 条边，清洗后降维至 **{len(edges):,}** 条边，"
                                f"核心节点 **{core_n:,}** 个（Top-800 圈层过滤）")
                        if edges.empty:
                            st.warning("清洗后无有效边，请检查数据是否全为自环 / 重复 / 空值。")
                    except Exception as e:
                        st.error(f"数据清洗失败：{type(e).__name__} - {e}")

                    if not edges.empty:
                        # ---------- 稠密数据保护：先让页面可用，再决定是否画图 ----------
                        # 边数超过 VIZ_AUTO_OFF_EDGES 时默认不自动绘图：整页重跑一次就重画一次图，
                        # 浏览器主线程被 SVG 图元塞满就是"此页面没有响应"的成因。
                        if "t2_render_graph" not in st.session_state:
                            st.session_state["t2_render_graph"] = len(edges) <= VIZ_AUTO_OFF_EDGES
                        render_graph = st.toggle(
                            "🕸️ 渲染网络关系图", key="t2_render_graph",
                            help=("绘图只渲染 Top150 节点 / 约 1200 条骨架边（WebGL），通常无需关闭。"
                                  "如果页面仍偏慢，关掉它：KPI 与预算表不受影响。"))
                        if not render_graph:
                            st.caption(f"已跳过绘图（清洗后 {len(edges):,} 条边）。"
                                       f"KPI 与预算分配不受影响，需要看图时打开上方开关即可。")

                        try:
                            t0 = time.perf_counter()
                            with st.spinner("图计算中（中介中心性采用采样近似，请稍候）..."):
                                res2 = run_analysis(edges, total_budget,
                                                    is_directed=directed_real, time_col=tcol)
                            t_analysis = time.perf_counter() - t0
                            render_metrics(res2, total_budget, len(edges))

                            # ---------- 误判哨兵：水军占比过高时给出解释和一键切换 ----------
                            try:
                                if not res2.empty:
                                    n_all = len(res2)
                                    n_water = int(res2["Status"].isin(WATER_ARMY).sum())
                                    if n_all and n_water / n_all > 0.4:
                                        msg = (f"本次有 **{n_water:,}/{n_all:,}**（{n_water / n_all:.0%}）"
                                               f"个节点被判为水军。真实社交网络的平均聚类系数本身就高，"
                                               f"「聚类>0.8 且 中介<0.01」在这类稠密无向图上会大面积误判。")
                                        if directed_real:
                                            msg += "你当前勾选了「按有向图分析」，无向好友关系数据请先关闭它。"
                                            st.warning(msg)
                                            # 必须用 on_click 回调改控件状态：
                                            # 在脚本里直接写 st.session_state["t2_directed"] 会触发
                                            # "cannot be modified after the widget is instantiated" 异常
                                            st.button("↩️ 改为无向图重新分析", key="t2_fix_directed",
                                                      on_click=_force_undirected)
                                        else:
                                            msg += ("可在文件顶部调高 `CLUSTER_TH`、或把 `BETWEEN_TH` 改成"
                                                    "相对阈值（如 0.1×最大中介中心性）后再试。")
                                            st.warning(msg)
                            except Exception:
                                pass

                            if res2.empty:
                                st.error("图分析未返回结果，请检查数据质量。")
                            else:
                                if render_graph:
                                    G2 = build_graph(edges, is_directed=directed_real)
                                    chart2 = st.empty()
                                    with chart2.container():
                                        t1 = time.perf_counter()
                                        fig2 = build_network_figure(
                                            G2, res2, VIZ_MAX_NODES,
                                            title=(f"真实数据网络（{G2.number_of_nodes():,} 节点 / "
                                                   f"{G2.number_of_edges():,} 边，抽样 Top{VIZ_MAX_NODES}）"))
                                        t_fig = time.perf_counter() - t1
                                        if fig2 is not None:
                                            st.plotly_chart(fig2, key="t2_network",
                                                            config=PLOT_CONFIG, **W_PLOT)
                                            st.caption(f"耗时：图计算 {t_analysis:.1f}s（首次，已缓存）"
                                                       f"｜绘图构建 {t_fig:.2f}s｜WebGL 渲染")
                                        else:
                                            st.warning("网络图渲染失败（已降级），下方表格不受影响。")

                                render_budget_table(res2, total_budget, key_prefix="t2")
                                render_status_detail(res2, key_prefix="t2")
                        except Exception as e:
                            st.error(f"图计算失败：{type(e).__name__} - {e}")

    st.divider()
    st.caption("算法口径：综合得分 = 0.3×归一化度 + 0.4×归一化中介 + 0.3×归一化特征向量；"
               "水军与暴雷节点得分强制归零，不参与预算分配。")


if __name__ == "__main__":
    main()
