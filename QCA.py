import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
# 假设 Excel 数据文件名为 data.xlsx，其中包含27个表单 (每个受试者一个表单)

# 载入清洗后的数据，并构造 QCA 案例数据：每个受试者取45分钟内各感知因子的平均值
subjects_data = pd.read_pickle("cleaned_data.pkl")
cases = []
for df in subjects_data:
    case = {}
    case['SocialPresence'] = df['SocialPresence'].mean()
    case['ImmersionFlow'] = df['ImmersionFlow'].mean()
    case['CognitiveLoad'] = df['CognitiveLoad'].mean()
    case['Usability'] = df['Usability'].mean()
    case['InteractionEfficiency'] = df['InteractionEfficiency'].mean()
    case['Anxiety'] = df['Anxiety'].mean()
    # 任务表现作为结果变量
    if df['TaskPerformance'].count() > 0:
        case['TaskPerformance'] = df['TaskPerformance'].mean()
    else:
        case['TaskPerformance'] = 0
    cases.append(case)
cases_df = pd.DataFrame(cases)

# 将数据标准化到 [0,1]
for col in ['SocialPresence','ImmersionFlow','CognitiveLoad','Usability','InteractionEfficiency','Anxiety','TaskPerformance']:
    min_val = cases_df[col].min()
    max_val = cases_df[col].max()
    if max_val > min_val:
        cases_df[col] = (cases_df[col] - min_val) / (max_val - min_val)
    else:
        cases_df[col] = 0.5

# 定义条件集合和结果
conditions = ['SocialPresence','ImmersionFlow','CognitiveLoad','Usability','InteractionEfficiency','Anxiety']
outcome = 'TaskPerformance'

# 二值化：阈值 0.5
binary_conditions = (cases_df[conditions] > 0.5).astype(int)
binary_outcome = (cases_df[outcome] > 0.5).astype(int)

# 用 pyeda 构造条件变量
cond_vars = exprvars('x', len(conditions))  # x0, x1, ..., x5

# 构造所有结果为1的案例对应的合取项
terms = []
for i in range(len(cases_df)):
    if binary_outcome.iloc[i] == 1:
        row = binary_conditions.iloc[i].values
        term = None
        for j, val in enumerate(row):
            literal = cond_vars[j] if val == 1 else ~cond_vars[j]
            term = literal if term is None else term & literal
        if term is not None:
            terms.append(term)


# 复杂解：所有项的或
complex_solution = Or(*terms)
complex_dnf = complex_solution.to_dnf()

print("复杂解：")
print(complex_solution)

# 简单解（最简解）：利用 espresso_exprs 简化
[parsimonious_solution] = espresso_exprs(complex_dnf)
print("\n简单解：")
print(parsimonious_solution)






# 中间解：例如假定 'SocialPresence' (x0) 为核心条件，要求必须出现
intermediate_expr = complex_solution & cond_vars[0]
# 将中间表达式转换为 DNF 格式后再进行简化
intermediate_expr_dnf = intermediate_expr.to_dnf()
[intermediate_solution] = espresso_exprs(intermediate_expr_dnf)

print("\n中间解（假定 'SocialPresence' 为核心）：")
print(intermediate_solution)

# 中文转换映射
mapping = {
    'x0': '社会感知',
    'x1': '沉浸感与心流',
    'x2': '认知负荷',
    'x3': '可用性',
    'x4': '交互效率',
    'x5': '焦虑与挫败感'
}
def translate_expr(expr):
    expr_str = str(expr)
    for eng, chi in mapping.items():
        expr_str = expr_str.replace(eng, chi)
    return expr_str

print("\n中文表达：")
print("复杂解：", translate_expr(complex_solution))
print("简单解：", translate_expr(parsimonious_solution))
print("中间解：", translate_expr(intermediate_solution))

# 构建 QCA 路径分析图（以有向图表示）
G_qca = nx.DiGraph()
outcome_node = "高任务表现"
G_qca.add_node(outcome_node, type='outcome')
for expr, sol in [ (translate_expr(complex_solution), "复杂解"),
                   (translate_expr(parsimonious_solution), "简单解"),
                   (translate_expr(intermediate_solution), "中间解") ]:
    G_qca.add_node(sol, type='combo', label=expr)
    G_qca.add_edge(sol, outcome_node)

pos = nx.spring_layout(G_qca, seed=42)
labels = {}
for node, attr in G_qca.nodes(data=True):
    if attr.get('type')=='combo':
        labels[node] = attr.get('label')
    else:
        labels[node] = node
plt.figure(figsize=(8,6))
nx.draw_networkx_nodes(G_qca, pos, node_color='lightyellow', node_size=1500)
nx.draw_networkx_edges(G_qca, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
nx.draw_networkx_labels(G_qca, pos, labels=labels, font_size=10)
plt.title("QCA 路径分析图（复杂、中间、简单解）")
plt.axis('off')
plt.show()