import pandas as pd
import numpy as np

# 设定文件路径（请修改为实际路径）
excel_path = 'd:/desktop/27份数据源.xlsx'
xlsx = pd.ExcelFile(excel_path)
sheet_names = xlsx.sheet_names

# 定义重命名映射和感知因子列表
rename_dict = {
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
}
# 感知因子列表
perception_factors = ['SocialPresence','ImmersionFlow','Presence','CognitiveLoad',
                      'Satisfaction','Anxiety','SwitchingSmoothness','Comfort',
                      'Usability','NeedsFit','InteractionEfficiency']

subjects_data = []  # 存储所有受试者的 DataFrame
misop_rates = []    # 误操作率
crash_rates = []    # 系统出错率

for sheet in sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    # 重命名列，注意直接赋值
    df = df.rename(columns=rename_dict)
    df.columns = df.columns.str.strip()
    
    # 添加受试者标识
    df['Subject'] = sheet
    
    # 将 Time 列设置为索引，确保时间轴为1~45
    if 'Time' in df.columns:
        df = df.set_index('Time')
    df.index = df.index.astype(int)
    
    # 对感知因子使用线性插值（limit_direction='both'避免两端NaN）
    df[perception_factors] = df[perception_factors].interpolate(method='linear', limit_direction='both')
    
    # 心率数据：滚动均值填充，并避免链式赋值
    df['HeartRate'] = df['HeartRate'].fillna(df['HeartRate'].rolling(window=3, min_periods=1, center=True).mean())
    df['HeartRate'] = df['HeartRate'].bfill()
    df['HeartRate'] = df['HeartRate'].ffill()
    
    # 误操作和系统出错填充为0
    df['Misoperation'] = df['Misoperation'].fillna(0)
    df['SystemError'] = df['SystemError'].fillna(0)
    
    # 计算误操作率和系统出错率（基于45个时间点）
    misop_rate = df['Misoperation'].sum() / len(df)
    crash_rate = df['SystemError'].sum() / len(df)
    misop_rates.append(misop_rate)
    crash_rates.append(crash_rate)
    
    subjects_data.append(df)

# 输出描述性统计信息（可选）
misop_rates = np.array(misop_rates)
crash_rates = np.array(crash_rates)
print("误操作率统计:", misop_rates)
print("系统出错率统计:", crash_rates)

# 保存所有受试者数据到一个 pickle 文件，以便后续模块调用
pd.to_pickle(subjects_data, "d:/desktop/cleaned_data.pkl")
print("数据清洗完毕，已保存为 cleaned_data.pkl")