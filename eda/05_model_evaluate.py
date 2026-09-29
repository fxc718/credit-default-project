# 05_model_evaluate.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def calc_ks(y_true, y_pred_prob):
    """计算KS值"""
    df = pd.DataFrame({"y": y_true, "prob": y_pred_prob})
    df = df.sort_values("prob")
    df["good"] = 1 - df["y"]
    df["bad"] = df["y"]
    df["cum_good"] = df["good"].cumsum() / df["good"].sum()
    df["cum_bad"] = df["bad"].cumsum() / df["bad"].sum()
    df["ks"] = np.abs(df["cum_good"] - df["cum_bad"])
    return df["ks"].max()


def calculate_psi(base, compare, bins=10):
    """
    计算PSI群体稳定性指标
    base: 基准样本（训练集）
    compare: 对比样本（测试集）
    bins: 等频分箱数
    return: total_psi, 分箱详情df
    """
    q = np.linspace(0, 1, bins + 1)
    bin_edges = np.quantile(base, q)
    bin_edges = np.unique(bin_edges)

    base_counts = np.histogram(base, bins=bin_edges)[0]
    comp_counts = np.histogram(compare, bins=bin_edges)[0]

    base_pct = base_counts / base_counts.sum()
    comp_pct = comp_counts / comp_counts.sum()

    # 防止log(0)
    base_pct = np.where(base_pct == 0, 1e-6, base_pct)
    comp_pct = np.where(comp_pct == 0, 1e-6, comp_pct)

    psi_each = (comp_pct - base_pct) * np.log(comp_pct / base_pct)
    total_psi = psi_each.sum()

    df_bin = pd.DataFrame({
        "base_pct": base_pct,
        "compare_pct": comp_pct,
        "psi_bin": psi_each
    })
    return total_psi, df_bin


if __name__ == "__main__":
    # 读取清洗后的数据
    df = pd.read_csv("../data/clean_data.csv")
    target = "SeriousDlqin2yrs"

    # ===== IV>=0.02 筛选后的8个有效特征 =====
    feature_selected = [
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTimes90DaysLate",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "age",
        "DebtRatio",
        "NumberOfOpenCreditLinesAndLoans",
        "MonthlyIncome",
    ]

    X = df[feature_selected]
    y = df[target]

    # 划分训练/测试集，stratify分层保证正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集 {X_train.shape}, 测试集 {X_test.shape}")
    print(f"训练集违约占比 {y_train.mean():.4f}, 测试集违约占比 {y_test.mean():.4f}\n")

    # ========== 加载模型 ==========
    lr_model = joblib.load("lr_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")

    # ========== LR预测与指标 ==========
    lr_train_pred = lr_model.predict_proba(X_train.values)[:, 1]
    lr_test_pred = lr_model.predict_proba(X_test.values)[:, 1]

    lr_train_auc = roc_auc_score(y_train, lr_train_pred)
    lr_test_auc = roc_auc_score(y_test, lr_test_pred)
    lr_train_ks = calc_ks(y_train, lr_train_pred)
    lr_test_ks = calc_ks(y_test, lr_test_pred)

    # ========== XGB预测与指标 ==========
    xgb_train_pred = xgb_model.predict_proba(X_train.values)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test.values)[:, 1]

    xgb_train_auc = roc_auc_score(y_train, xgb_train_pred)
    xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)
    xgb_train_ks = calc_ks(y_train, xgb_train_pred)
    xgb_test_ks = calc_ks(y_test, xgb_test_pred)

    # ========== 汇总AUC、KS结果 ==========
    result = pd.DataFrame({
        "model": ["LR", "LR", "XGB", "XGB"],
        "dataset": ["train", "test", "train", "test"],
        "AUC": [lr_train_auc, lr_test_auc, xgb_train_auc, xgb_test_auc],
        "KS": [lr_train_ks, lr_test_ks, xgb_train_ks, xgb_test_ks]
    })
    print("===== 模型AUC & KS汇总 =====")
    print(result)
    result.to_csv("../output/train_test_compare.csv", index=False)
    print("\n训练/测试集指标保存完成\n")

    # ========== 计算各特征PSI ==========
    psi_result = []
    for col in feature_selected:
        psi_val, _ = calculate_psi(X_train[col].values, X_test[col].values, bins=10)
        psi_result.append({"feature": col, "PSI": psi_val})

    psi_df = pd.DataFrame(psi_result)
    print("===== 特征PSI结果（训练集VS测试集） =====")
    print(psi_df)
    psi_df.to_csv("../output/feature_psi.csv", index=False)
    print("\nPSI结果保存完成\n")

    # ========== XGB特征重要性（新增部分） ==========
    feature_importance = pd.DataFrame({
        "feature": feature_selected,
        "importance": xgb_model.feature_importances_
    })
    # 按重要性降序排序
    feature_importance = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    print("===== XGBoost特征重要性（降序） =====")
    print(feature_importance)
    feature_importance.to_csv("../output/xgb_feature_importance.csv", index=False)
    print("\nXGB特征重要性保存完成")
