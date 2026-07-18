import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.stattools")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx


plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
# 假设 Excel 数据文件名为 data.xlsx，其中包含27个表单 (每个受试者一个表单)
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




# 准备存储每个受试者处理后的数据
subjects_data = []  # 将每个受试者的DataFrame存入列表
misop_rates = []    # 存储每个受试者的误操作率
crash_rates = []    # 存储每个受试者的系统崩溃率

for sheet in sheet_names:
    # 读取每个表单的数据
    df = pd.read_excel(xlsx, sheet_name=sheet)
    # 重命名列为英文，便于编程（保留必要的中文含义）
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

    print(df.columns.tolist())
    # 将时间列设置为索引，确保时间轴为1~45
    if 'Time' in df.columns:
        df.set_index('Time', inplace=True)
    # 将索引转换为整数（时间步）以防不是整数
    df.index = df.index.astype(int)

    # 缺失值处理：
    # 感知因子列使用线性插值填充 (包括SocialPresence等主观感知因子，以及InteractionEfficiency等)
    perception_factors = ['SocialPresence','ImmersionFlow','Presence','CognitiveLoad',
                          'Satisfaction','Anxiety','SwitchingSmoothness','Comfort',
                          'Usability','NeedsFit','InteractionEfficiency']
    # 如果系统稳定性(0)被认为是感知因子的一部分，可加入上面列表；这里假定不是
    df[perception_factors] = df[perception_factors].interpolate(method='linear', limit_direction='both')


#   print(df[perception_factors].head())

    # 心率数据使用滚动均值填充 (窗口=3)
    df['HeartRate'] = df['HeartRate'].fillna(
        df['HeartRate'].rolling(window=3, min_periods=1, center=True).mean()
    )
    # 滚动均值后可能在序列两端仍有NaN，用前向/后向填充确保无NaN
    df['HeartRate'] = df['HeartRate'].bfill()
    df['HeartRate'] = df['HeartRate'].ffill()

    # 误操作 和 系统稳定性 缺失值填充为0
    if 'Misoperation' in df.columns:
        df['Misoperation']= df['Misoperation'].fillna(0)
    if 'SystemError' in df.columns:
        df['SystemError'] = df['SystemError'].fillna(0)
    
    # 计算误操作率 和 系统崩溃率 并保存
    total_misops = df['Misoperation'].sum()
    misop_rate = total_misops / len(df)  # 误操作率 = 误操作次数/总时间点数
    misop_rates.append(misop_rate)
    total_crashes = df['SystemError'].sum()  # 假定SystemStability列1表示崩溃事件
    crash_rate = total_crashes / len(df)         # 系统崩溃率 = 崩溃次数/总时间点数
    crash_rates.append(crash_rate)

    # 将处理后的数据存入列表
    subjects_data.append(df)
with pd.ExcelWriter("d:/desktop/QCA.xlsx") as writer:
    for i, df in enumerate(subjects_data):
        df.to_excel(writer, sheet_name=f"Sheet{i+1}", index=False)