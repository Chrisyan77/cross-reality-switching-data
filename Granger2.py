import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from infomap import Infomap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False       # 正常显示负号

# 忽略未来警告
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")

# ---------------------------
# 数据读取与预处理
xlsx = pd.ExcelFile('d:/desktop/27份数据源.xlsx')
sheet_names = xlsx.sheet_names

chinese_mapping = {
    'SocialPresence': '社会感知',
    'ImmersionFlow': '沉浸感与心流',
    'Presence': '临场感',
    'CognitiveLoad': '认知负荷',
    'Satisfaction': '愉悦度',
    'Anxiety': '焦虑与挫败感',
    'SwitchingSmoothness': '切换流畅度',
    'Comfort': '感官与生理舒适度',
    'Usability': '可用性',
    'NeedsFit': '需求匹配',
    'InteractionEfficiency': '交互效率'
}

subjects_data = []  # 存储每个受试者数据
misop_rates = []    # 误操作率
crash_rates = []    # 系统崩溃率

for sheet in sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.rename(columns={
        '时间': 'Time',
        '社会感知': 'SocialPresence',
        '沉浸感与心流': 'ImmersionFlow',
        '临场感': 'Presence',
        '认知负荷': 'CognitiveLoad',
        '愉悦度': 'Satisfaction',
        '焦虑与挫败感': 'Anxiety',
        '切换流畅度': 'SwitchingSmoothness',
        '感官与生理舒适度': 'Comfort',
        '可用性': 'Usability',
        '需求匹配': 'NeedsFit',
        '交互效率': 'InteractionEfficiency',
        '系统出错': 'SystemError',
        '任务表现': 'TaskPerformance',
        '误操作': 'Misoperation',
        '处于什么位置': 'Location',
        '心率': 'HeartRate'
    }, inplace=True)
    df.columns = df.columns.str.strip()
    if 'Time' in df.columns:
        df.set_index('Time', inplace=True)
    df.index = df.index.astype(int)
    perception_factors = ['SocialPresence','ImmersionFlow','Presence','CognitiveLoad',
                          'Satisfaction','Anxiety','SwitchingSmoothness','Comfort',
                          'Usability','NeedsFit','InteractionEfficiency']
    df[perception_factors] = df[perception_factors].interpolate(method='linear', limit_direction='both')
    df['HeartRate'] = df['HeartRate'].fillna(
        df['HeartRate'].rolling(window=3, min_periods=1, center=True).mean()
    )
    df['HeartRate'] = df['HeartRate'].bfill()
    df['HeartRate'] = df['HeartRate'].ffill()
    if 'Misoperation' in df.columns:
        df['Misoperation'] = df['Misoperation'].fillna(0)
    if 'SystemError' in df.columns:
        df['SystemError'] = df['SystemError'].fillna(0)
    total_misops = df['Misoperation'].sum()
    misop_rate = total_misops / len(df)
    misop_rates.append(misop_rate)
    total_crashes = df['SystemError'].sum()
    crash_rate = total_crashes / len(df)
    crash_rates.append(crash_rate)
    subjects_data.append(df)

misop_rates = np.array(misop_rates)
crash_rates = np.array(crash_rates)
print("误操作率描述性统计:", {"mean": misop_rates.mean(), "std": misop_rates.std(ddof=0)})
print("系统崩溃率描述性统计:", {"mean": crash_rates.mean(), "std": crash_rates.std(ddof=0)})

# 构建平均时间序列（所有受试者的45分钟数据平均）
time_index = subjects_data[0].index
avg_df = pd.DataFrame(0, index=time_index, columns=subjects_data[0].columns)
for df in subjects_data:
    avg_df += df
avg_df = avg_df / len(subjects_data)

# 提取感知因子
factor_cols = ['SocialPresence','ImmersionFlow','Presence','CognitiveLoad',
               'Satisfaction','Anxiety','SwitchingSmoothness','Comfort',
               'Usability','NeedsFit','InteractionEfficiency']
factors = factor_cols

# ---------------------------
# 数据去趋势：使用 HP 滤波器
# 选择平滑参数 λ，根据数据频率调整，45分钟数据可尝试较小的 λ 值
lamb_value = 30  # 你可以调整此值以获得理想的去趋势效果
from statsmodels.tsa.filters.hp_filter import hpfilter

avg_df_detrended = avg_df.copy()
for col in factor_cols:
    cycle, trend = hpfilter(avg_df[col], lamb=lamb_value)
    avg_df_detrended[col] = avg_df[col] - trend

# ---------------------------
# 1. Granger 因果检验
maxlag = 5
p_values = pd.DataFrame(1.0, index=factors, columns=factors)
best_lags = pd.DataFrame(0, index=factors, columns=factors)

for cause in factors:
    for effect in factors:
        if cause == effect:
            continue
        data_pair = avg_df_detrended[[effect, cause]].dropna()
        if data_pair.shape[0] < maxlag + 1:
            continue
        results = grangercausalitytests(data_pair, maxlag=maxlag, verbose=False)
        min_p = 1.0
        best_k = 0
        for lag, res in results.items():
            p_val = res[0]['ssr_ftest'][1]
            if p_val < min_p:
                min_p = p_val
                best_k = lag
        p_values.loc[cause, effect] = min_p
        best_lags.loc[cause, effect] = best_k

causal_matrix = pd.DataFrame(0, index=factors, columns=factors)
for cause in factors:
    for effect in factors:
        if cause != effect and p_values.loc[cause, effect] < 0.05:
            causal_matrix.loc[cause, effect] = 1

# ---------------------------
# 2. VAR 模型分析：提取正负方向
sign_dict = {}
for cause in factors:
    for effect in factors:
        if causal_matrix.loc[cause, effect] == 1:
            lag_order = int(best_lags.loc[cause, effect])
            if lag_order < 1:
                continue
            data_pair = avg_df_detrended[[effect, cause]].dropna().reset_index(drop=True)
            try:
                model = VAR(data_pair[[cause, effect]])
                var_result = model.fit(lag_order, ic='aic')
            except Exception as e:
                print(f"VAR拟合失败 {cause} -> {effect}: {e}")
                continue
            cause_index = 0
            effect_index = 1
            coef_sum = 0.0
            for i in range(var_result.k_ar):
                coef_matrix = var_result.coefs[i]
                coef_sum += coef_matrix[effect_index, cause_index]
            sign_dict[(cause, effect)] = 1 if coef_sum > 0 else -1

print("VAR系数符号字典:")
for k, v in sign_dict.items():
    print(f"{k[0]} -> {k[1]}: {'正向' if v==1 else '负向'}")

# ---------------------------
# 3. 构建因果网络（基于 Granger 和 VAR 结果）
G = nx.DiGraph()
G.add_nodes_from(factors)
for cause in factors:
    for effect in factors:
        if causal_matrix.loc[cause, effect] == 1:
            weight = -np.log10(p_values.loc[cause, effect])
            sign = sign_dict.get((cause, effect), 1)
            G.add_edge(cause, effect, weight=abs(weight), sign=sign)

# ---------------------------
# 弱边过滤：过滤掉权重低于阈值（例如0.5）的边
edge_threshold =3.5
filtered_edges = [(u, v, data) for u, v, data in G.edges(data=True) if data["weight"] >= edge_threshold]
G_filtered = nx.DiGraph()
G_filtered.add_nodes_from(G.nodes())
for u, v, data in filtered_edges:
    G_filtered.add_edge(u, v, **data)

# ---------------------------
# 4. 可视化热力图
# 4.1 Granger 因果热力图
strength_matrix = -np.log10(p_values)
strength_matrix[p_values >= 0.05] = 0
np.fill_diagonal(strength_matrix.values, np.nan)
plt.figure(figsize=(8,6))
sns.heatmap(strength_matrix, annot=False, cmap='YlOrRd', 
            xticklabels=factors, yticklabels=factors, mask=np.isnan(strength_matrix))
plt.title('Granger 因果热力图 (-log10 p)')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# 4.2 VAR 方向性热力图
direction_matrix = pd.DataFrame(0, index=factors, columns=factors).astype(float)
for (cause, effect), s in sign_dict.items():
    direction_matrix.loc[cause, effect] = s
np.fill_diagonal(direction_matrix.values, np.nan)
plt.figure(figsize=(8,6))
sns.heatmap(direction_matrix, annot=False, cmap='coolwarm', center=0,
            xticklabels=factors, yticklabels=factors, mask=np.isnan(direction_matrix))
plt.title('VAR 方向性热力图 (红=正向, 蓝=负向)')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------
# 5. 社团划分：使用 Infomap 进行社团检测（有向网络）
node_to_id = {node: i for i, node in enumerate(G_filtered.nodes())}
id_to_node = {i: node for node, i in node_to_id.items()}
im = Infomap(directed=True)
for u, v, data in G_filtered.edges(data=True):
    im.add_link(node_to_id[u], node_to_id[v], data["weight"])
im.run(two_level=True)
modules = im.get_modules()
infomap_communities = {}
for node_id, module in modules.items():
    node = id_to_node[node_id]
    infomap_communities.setdefault(module, []).append(node)
print("初始 Infomap 社团划分结果：")
for comm_id, nodes in infomap_communities.items():
    print(f"社团 {comm_id}: {nodes}")

# 目标：选取3个社团。如果社团数不为3，则选择节点数最多的3个
if len(infomap_communities) != 3:
    sorted_comms = sorted(infomap_communities.items(), key=lambda x: len(x[1]), reverse=True)
    selected_comms = {comm_id: nodes for comm_id, nodes in sorted_comms[:3]}
else:
    selected_comms = infomap_communities

print("\n选定的3个社团：")
for comm_id, nodes in selected_comms.items():
    print(f"社团 {comm_id}: {nodes}")

# ---------------------------
# 6. 核心因子选择：采用 PageRank 指标
pagerank = nx.pagerank(G_filtered, weight="weight")
core_nodes = {}
for comm_id, nodes in selected_comms.items():
    core = max(nodes, key=lambda node: pagerank[node])
    core_nodes[comm_id] = core
    print(f"社团 {comm_id} 核心因子: {core}")

# ---------------------------
# 7. 网络图可视化（优化版）
pr_values = list(pagerank.values())
min_pr, max_pr = min(pr_values), max(pr_values)
node_sizes = [np.interp(pagerank[node], [min_pr, max_pr], [300, 800]) for node in G_filtered.nodes()]

# 节点颜色：根据社团划分，不在选定社团的统一设为浅灰色
num_selected = len(selected_comms)
color_map = cm.get_cmap("tab10", num_selected)
node_color_map = {}
for idx, (comm_id, nodes) in enumerate(selected_comms.items()):
    for node in nodes:
        node_color_map[node] = color_map(idx)
for node in G_filtered.nodes():
    if node not in node_color_map:
        node_color_map[node] = "lightgray"

# 边宽度：根据权重映射到 [0.5, 6.0]
all_edge_weights = [data["weight"] for u, v, data in G_filtered.edges(data=True)]
min_linewidth, max_linewidth = 0.5, 6.0
edge_widths = [np.interp(data["weight"], [min(all_edge_weights), max(all_edge_weights)], 
                         [min_linewidth, max_linewidth]) for u, v, data in G_filtered.edges(data=True)]
# 边颜色：根据 sign 属性，正向为黑色，负向为红色
edge_colors = ["black" if data.get("sign",1)==1 else "red" for u, v, data in G_filtered.edges(data=True)]

pos = nx.kamada_kawai_layout(G_filtered)
plt.figure(figsize=(10,8))
nx.draw_networkx_nodes(G_filtered, pos, node_size=node_sizes, 
                       node_color=[node_color_map[node] for node in G_filtered.nodes()],
                       edgecolors="black", linewidths=1.5)
nx.draw_networkx_edges(G_filtered, pos, arrows=True, arrowstyle="->", arrowsize=15,
                       width=edge_widths, edge_color=edge_colors)
nx.draw_networkx_labels(G_filtered, pos, font_size=10, font_weight="bold", font_color="black")
plt.title("因果网络社团划分与核心因子 (正向边黑色，负向边红色)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

print("\n最终选定的社团及核心因子：")
for comm_id, nodes in selected_comms.items():
    print(f"社团 {comm_id}: {nodes}  -> 核心因子: {core_nodes[comm_id]}")
