# src/03_woe_iv_feature.py
# 信贷风控项目：WOE分箱 + IV特征筛选
# 输入：data/clean_data.csv 清洗完成的数据
# 输出：各特征IV值，筛选出有效特征
import pandas as pd
import numpy as np

def calc_woe_iv(df, feature_col, target_col):
    """
    计算单个特征WOE与IV
    df:数据集
    feature_col：分箱后的特征列
    target_col：标签 SeriousDlqin2yrs（1=违约，0=正常）
    """
    # 分组统计每组总样本、坏样本(违约)数量
    group_stat = df.groupby(feature_col)[target_col].agg(["count", "sum"])
    group_stat.columns = ["total", "bad"]
    group_stat["good"] = group_stat["total"] - group_stat["bad"]

    # 全局好坏样本总数
    total_bad = df[target_col].sum()
    total_good = len(df) - total_bad

    # +0.5平滑，防止分母为0
    group_stat["bad_pct"] = (group_stat["bad"] + 0.5) / total_bad
    group_stat["good_pct"] = (group_stat["good"] + 0.5) / total_good

    group_stat["WOE"] = np.log(group_stat["bad_pct"] / group_stat["good_pct"])
    group_stat["IV_i"] = (group_stat["bad_pct"] - group_stat["good_pct"]) * group_stat["WOE"]
    total_iv = group_stat["IV_i"].sum()
    return group_stat, total_iv


if __name__ == "__main__":
    # 读取清洗好的数据
    df_clean = pd.read_csv("../data/clean_data.csv")
    target = "SeriousDlqin2yrs"

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
    print("==== 开始计算各特征IV ====")

    for feat in feature_list:
        # --------------------------

        # --------------------------
        nunique_val = df_clean[feat].nunique()
        # 1. 二值 / 离散少的特征，直接用原值分组，不做qcut
        if nunique_val <= 5:
            df_clean["bin_temp"] = df_clean[feat].astype(str)
        else:
            # 2. 逾期计数特征，手动自定义分箱：0单独一箱，1，2，3，>=4
            if feat.startswith("NumberOfTime") or feat == "NumberOfTimes90DaysLate":
                df_clean["bin_temp"] = pd.cut(
                    df_clean[feat],
                    bins=[-1, 0, 1, 2, 3, 100],
                    right=True
                )
            else:
                # 3.普通连续特征：等频5箱
                df_clean["bin_temp"] = pd.qcut(df_clean[feat], q=5, duplicates="drop")

        woe_table, iv = calc_woe_iv(df_clean, "bin_temp", target)
        iv_summary[feat] = iv

        print(f"\n【{feat}】 IV = {iv:.4f}")
        print(woe_table)

    # 整理IV汇总表，按IV降序排序
    iv_df = pd.DataFrame(list(iv_summary.items()), columns=["feature", "IV"])
    iv_df = iv_df.sort_values("IV", ascending=False)
    print("\n===== 全部特征IV汇总表 =====")
    print(iv_df)

    # 保存IV结果，后面写报告用
    iv_df.to_csv("../output/iv_result.csv", index=False)
    print("\nIV结果已保存到 output/iv_result.csv")
