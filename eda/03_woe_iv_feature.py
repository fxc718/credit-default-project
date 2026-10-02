# src/03_woe_iv_feature.py
# 信贷风控项目：WOE分箱 + IV特征筛选
#
# 设计要点：
# 1. 防数据泄露：先划分训练/测试集，仅在训练集拟合分箱边界和WOE，再transform测试集
# 2. WOE平滑：分子分母均加0.5，避免极端分箱导致除零
# 3. 分箱上限：自定义分箱上限使用 np.inf，防止极端值落入 NaN
# 4. 工程化输出：保存 WOE 映射字典和分箱边界到 models/，供 predict.py 加载复用
# 5. 下游对接：输出 train_woe.csv / test_woe.csv，04_model_train.py 直接读取


import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split


def calc_woe_iv(df, feature_col, target_col):
    """
    计算单个特征WOE与IV (基于分箱后的特征)
    注意：此函数必须只在训练集上调用计算，防止标签泄露
    """
    # 分组统计每组总样本、坏样本(违约)数量
    group_stat = df.groupby(feature_col)[target_col].agg(["count", "sum"])
    group_stat.columns = ["total", "bad"]
    group_stat["good"] = group_stat["total"] - group_stat["bad"]

    # 全局好坏样本总数
    total_bad = df[target_col].sum()
    total_good = len(df) - total_bad

    # 标准拉普拉斯平滑：分子分母都加0.5，防止分母为0
    group_stat["bad_pct"] = (group_stat["bad"] + 0.5) / (total_bad + 0.5)
    group_stat["good_pct"] = (group_stat["good"] + 0.5) / (total_good + 0.5)

    group_stat["WOE"] = np.log(group_stat["bad_pct"] / group_stat["good_pct"])
    group_stat["IV_i"] = (group_stat["bad_pct"] - group_stat["good_pct"]) * group_stat["WOE"]
    total_iv = group_stat["IV_i"].sum()

    return group_stat, total_iv


def fit_woe_binning(df_train, feature_col, target_col):
    """
    在训练集上拟合分箱边界和WOE映射
    返回：woe_mapping (字典), iv (float), bins (array/list 或 None)
    """
    nunique_val = df_train[feature_col].nunique()
    bins = None
    temp_bin_col = "bin_temp"

    # 1. 二值 / 离散少的特征，直接用原值分组，不做cut
    if nunique_val <= 5:
        df_train[temp_bin_col] = df_train[feature_col].astype(str)
    else:
        # 2. 逾期计数特征，手动自定义分箱：0单独一箱，1，2，3，>=4
        if feature_col.startswith("NumberOfTime") or feature_col == "NumberOfTimes90DaysLate":
            bins = [-1, 0, 1, 2, 3, np.inf]  # 修复上限100的隐患，使用inf
            df_train[temp_bin_col] = pd.cut(df_train[feature_col], bins=bins, right=True)
        else:
            # 3. 普通连续特征：等频5箱，并返回分箱边界
            df_train[temp_bin_col], bins = pd.qcut(
                df_train[feature_col], q=5, duplicates="drop", retbins=True
            )

    # 计算 WOE 和 IV
    woe_table, iv = calc_woe_iv(df_train, temp_bin_col, target_col)

    # 构造映射字典：分箱区间字符串 -> WOE值
    woe_mapping = {}
    for idx, row in woe_table.iterrows():
        woe_mapping[str(idx)] = row["WOE"]

    return woe_mapping, iv, bins, woe_table


def transform_woe(df, feature_col, woe_mapping, bins):
    """
    使用训练集学到的规则对数据集进行 transform
    """
    temp_bin_col = "bin_temp"

    # 使用训练集的边界进行分箱
    if bins is None:
        # 离散特征
        df[temp_bin_col] = df[feature_col].astype(str)
    else:
        if feature_col.startswith("NumberOfTime") or feature_col == "NumberOfTimes90DaysLate":
            df[temp_bin_col] = pd.cut(df[feature_col], bins=bins, right=True)
        else:
            df[temp_bin_col] = pd.cut(df[feature_col], bins=bins, right=True)

    # 将分箱结果转为字符串，并映射到 WOE 值
    df[feature_col + "_WOE"] = df[temp_bin_col].astype(str).map(woe_mapping)

    # 处理测试集可能出现的未知分箱（例如测试集有极端值超出训练集边界）
    # 填充为训练集该特征 WOE 的均值（或者 0）
    mean_woe = np.mean(list(woe_mapping.values()))
    df[feature_col + "_WOE"] = df[feature_col + "_WOE"].fillna(mean_woe)

    return df


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../output", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # 1. 读取清洗好的数据
    print("正在读取数据...")
    df_clean = pd.read_csv("../data/clean_data.csv")
    target = "SeriousDlqin2yrs"

    # 2. 关键防泄露：先划分训练集和测试集
    print("正在划分训练集和测试集 (先切分，再拟合)...")
    df_train, df_test = train_test_split(
        df_clean, test_size=0.2, random_state=42, stratify=df_clean[target]
    )
    # 重置索引，防止后续操作产生对齐bug
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 需要计算IV的特征列表
    feature_list = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        "Missing_Dependents",
        "Income_Missing"
    ]

    iv_summary = {}
    feature_woe_mappings = {}
    feature_bins = {}

    print("\n==== 开始计算各特征IV (仅在训练集上拟合) ====")

    for feat in feature_list:
        # 2.1 仅在训练集上拟合分箱与WOE
        woe_mapping, iv, bins, woe_table = fit_woe_binning(df_train, feat, target)
        iv_summary[feat] = iv

        # 2.2 保存映射规则
        feature_woe_mappings[feat] = woe_mapping
        feature_bins[feat] = bins

        print(f"\n【{feat}】 IV = {iv:.4f}")
        print(woe_table[["total", "bad", "good", "WOE", "IV_i"]])

        # 2.3 将规则应用到训练集和测试集 (Transform)
        df_train = transform_woe(df_train, feat, woe_mapping, bins)
        df_test = transform_woe(df_test, feat, woe_mapping, bins)

    # 3. 整理IV汇总表，按IV降序排序
    iv_df = pd.DataFrame(list(iv_summary.items()), columns=["feature", "IV"])
    iv_df = iv_df.sort_values("IV", ascending=False)
    print("\n===== 全部特征IV汇总表 =====")
    print(iv_df)

    # 4. 保存 IV 结果 (供报告使用)
    iv_df.to_csv("../output/iv_result.csv", index=False)
    print("\nIV结果已保存到 output/iv_result.csv")

    # 5. 保存 WOE 编码后的数据集 (剔除原始特征，保留 WOE 特征和 target)
    # 供 04_model_train.py
    woe_columns = [f + "_WOE" for f in feature_list]
    train_final = df_train[woe_columns + [target]]
    test_final = df_test[woe_columns + [target]]

    train_final.to_csv("../data/train_woe.csv", index=False)
    test_final.to_csv("../data/test_woe.csv", index=False)
    print("训练集和测试集的 WOE 编码文件已保存到 data/ 目录")

    # 6. 保存分箱边界和 WOE 映射字典，供 predict.py 线上预测使用
    joblib.dump(feature_woe_mappings, "../models/woe_mappings.pkl")
    joblib.dump(feature_bins, "../models/feature_bins.pkl")
    print("WOE映射字典和分箱边界已保存到 models/ 目录")

