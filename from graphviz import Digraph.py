from graphviz import Digraph
from graphviz import Digraph

# 初始化流程图时指定中文字体
flow = Digraph('Research_Flow', 
              format='png',
              graph_attr={'rankdir':'TB', 'splines':'ortho'},
              node_attr={
                  'shape': 'box',
                  'style': 'rounded',
                  'fontname': 'SimHei'  # 👈 关键修改：指定中文字体
              },
              edge_attr={'fontname': 'SimHei'})  # 👈 边文字同样需要设置

# 设置全局编码（防止文件读写乱码）
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # 👈 根据实际安装路径修改
os.environ["LANG"] = "zh_CN.UTF-8"  # 👈 设置环境编码

# ========== 后续代码保持不变 ==========

# ========== 颜色编码分类 ==========
theory_color = '#B3CDE3'    # 理论研究-蓝色
experiment_color = '#CCEBC5' # 实验研究-绿色
modeling_color = '#FED9A6'   # 模型构建-橙色
practice_color = '#E6E6FA'   # 实践应用-紫色
meta_color = '#FBB4AE'       # 元分析-红色

# ========== 节点定义 ==========
# 阶段1: 研究起点
flow.node('start', '研究起点\n(现实需求: 扩展现实技术发展中的用户感知问题\n 理论需求: 跨现实系统过渡界面研究空白)', 
         color=theory_color, shape='ellipse')

# 阶段2: 理论建构
with flow.subgraph(name='theory') as t:
    t.attr(label='理论建构阶段', labelloc='t')
    t.node('lit_review', '文献研究\n(用户感知理论 | 过渡界面设计 | 量化方法)\n对应章节：第二章', color=theory_color)
    t.node('hypothesis', '研究假设提出\n(3.1.2)', color=theory_color)

# 阶段3: 实验研究
with flow.subgraph(name='experiment') as e:
    e.attr(label='实验研究阶段', labelloc='t')
    e.node('pre_exp', '''预实验设计 (3.1)
    ├ 指标筛选 (3.1.1)
    ├ 原型搭建 (3.1.4)
    └ 数据预收集''', color=experiment_color)
    e.node('formal_exp', '''正式实验 (3.2)
    ├ 对象选择
    ├ 数据采集
    └ 假设检验 (3.2.2)''', color=experiment_color)

# 阶段4: 模型构建
with flow.subgraph(name='model') as m:
    m.attr(label='模型构建阶段', labelloc='t')
    m.node('data_pre', '数据预处理 (4.1.1)', color=modeling_color)
    m.node('causal', '因果关系分析 (4.1.2)', color=modeling_color)
    m.node('network', '复杂网络建模 (4.2)', color=modeling_color)
    m.node('qca', 'QCA组态分析 (4.3)', color=modeling_color)

# 阶段5: 实践验证
with flow.subgraph(name='practice') as p:
    p.attr(label='实践验证阶段', labelloc='t')
    p.node('design', '设计转化 (5.1)\n交互流程 | 产品定义', color=practice_color)
    p.node('prototype', '原型迭代 (5.2)', color=practice_color)
    p.node('test', '''验证测试 (5.3)
    ├ 用户测试
    └ 效果验证''', color=practice_color)

# 阶段6: 研究闭环
flow.node('conclusion', '''结论与研究闭环 (第六章)
├ 理论贡献
├ 实践价值
└ 未来展望''', color=meta_color, shape='ellipse')

# ========== 边连接 ==========
# 主流程
flow.edges([
    ('start', 'lit_review'),
    ('lit_review', 'hypothesis'),
    ('hypothesis', 'pre_exp'),
    ('pre_exp', 'formal_exp'),
    ('formal_exp', 'data_pre'),
    ('data_pre', 'causal'),
    ('causal', 'network'),
    ('network', 'qca'),
    ('qca', 'design'),
    ('design', 'prototype'),
    ('prototype', 'test'),
    ('test', 'conclusion')
])

# 特殊连接
flow.edge('lit_review', 'pre_exp', label='理论支撑', style='dashed')
flow.edge('formal_exp', 'data_pre', label='数据输入', style='dashed')
flow.edge('qca', 'design', label='设计准则转化', style='dashed')
flow.edge('test', 'lit_review', label='实践反哺理论', style='dashed', constraint='false')

# ========== 输出设置 ==========
flow.render(filename='research_flow', cleanup=True, view=True)
print("流程图已生成：research_flow.png")