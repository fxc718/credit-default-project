# src/02_data_clean.py
# 信贷风控项目：数据清洗脚本
# 输入：原始数据集 cs-training.csv
# 输出：清洗后的 clean_data.csv，供后续WOE分箱、建模使用
import pandas as pd

# 1. 读取原始数据
df = pd.read_csv("../data/cs-training.csv")
# 复制副本，保留原始df，所有清洗操作作用在df_clean
df_clean = df.copy()

# 2. 缺失值处理，新增缺失标记特征
# 家属数量：新增缺失标记，中位数填充
df_clean["Missing_Dependents"] = df_clean["NumberOfDependents"].isnull().astype(int)
df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].fillna(df_clean["NumberOfDependents"].median())

# 月收入：新增缺失标记，缺失填充0
df_clean["Income_Missing"] = df_clean["MonthlyIncome"].isnull().astype(int)
df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].fillna(0)

# 3. 逾期特征特殊编码清洗：96、98为缺失标记，替换为0
pastdue_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]
for col in pastdue_cols:
    df_clean.loc[df_clean[col] >= 96, col] = 0

# 4. 业务异常样本过滤：年龄小于21岁剔除
df_clean = df_clean[df_clean["age"] >= 21]

# 5. 打印清洗前后样本对比
print(f"原始数据集 shape: {df.shape}")
print(f"清洗后数据集 shape: {df_clean.shape}")
print("\n==== 清洗完成，保存清洗后数据 ====")

# 保存清洗数据集，给03脚本读取
df_clean.to_csv("../data/clean_data.csv", index=False)
print("文件已保存至 data/clean_data.csv")
