"""
EDA 探索性数据分析报告
1. 数据集共13万样本，标签SeriousDlqin2yrs存在严重类别不平衡，负样本（不违约）远多于违约样本。
建模不使用准确率，重点参考AUC、KS作为核心评估指标。

2. 缺失情况：
MonthlyIncome（月收入）缺失约20%；NumberOfDependents（家属数量）缺失约2.5%。
信贷场景缺失具备业务含义，不直接删除，计划新增缺失标记特征，保留风险信号。

3. 逾期相关特征存在特殊编码96、98，不属于真实逾期次数，属于数据异常，清洗阶段替换为0。

4. 存在年龄小于21岁的不合理样本，业务上不符合信贷准入条件，后续做样本过滤。

5. 特征相关性分析：不存在极强多重共线性，大部分特征可以保留，进入后续WOE分箱流程。

EDA总结：
数据集存在样本不平衡、部分字段缺失、特殊异常编码、不合理样本等问题；
整体特征区分能力尚可，经过清洗预处理后，可以开展后续特征工程与建模。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
print("当前工作目录：", os.getcwd())
# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据集
df = pd.read_csv("data/cs-training.csv")

# ========== 1.基础信息查看 ==========
print("数据集形状：", df.shape)
print("\n前5行数据：")
print(df.head())
print("\n数据类型：")
print(df.dtypes)
print("\n基础统计描述：")
print(df.describe())

# ========== 2.缺失值统计 ==========
missing = df.isnull().sum()
print("\n各字段缺失数量：")
print(missing[missing>0])
# 缺失占比计算
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "缺失数量": missing,
    "缺失占比(%)": missing_pct
})
print("\n缺失值详情表：")
print(missing_df[missing_df["缺失数量"]>0])

# ========== 3.异常值统计 ==========
# 探查年龄异常样本
print("\n====年龄特征探查====")
print("年龄统计：")
print(df['age'].describe())
# 统计小于21岁样本数量
age_err = df[df['age'] < 21]
print(f"\n年龄小于21的样本条数：{len(age_err)}")
print(age_err[['age','SeriousDlqin2yrs']].head())
# 探查逾期特征的特殊编码（96、98）
print("\n====逾期特征特殊值探查====")
pastdue_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]
for col in pastdue_cols:
    print(f"\n{col} 取值计数：")
    print(df[col].value_counts().sort_index().tail(5))

# ========== 4.标签分布（违约与否）==========
label_counts = df['SeriousDlqin2yrs'].value_counts()
print("\n标签分布：")
print(label_counts)
print(f"违约样本占比：{label_counts[1]/len(df):.4f}")

plt.figure(figsize=(6,4))
label_counts.plot(kind="bar")
plt.title("标签分布（1=严重违约）")
plt.ylabel("样本数量")
plt.savefig("output/label_dist.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========5.相关性热力图 ==========
corr = df.corr()
plt.figure(figsize=(10,8))
im = plt.imshow(corr, cmap="coolwarm")
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
plt.title("特征相关性热力图")
plt.savefig("output/corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

