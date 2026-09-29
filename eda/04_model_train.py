# src/04_model_train.py
# 信贷违约预测：逻辑回归(评分卡) + XGBoost，AUC/KS评估
# 特征：按 IV>=0.02 从 iv_result.csv 筛选的8个有效特征
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb


def calc_ks(y_true, y_pred_proba):
    """KS指标：衡量好坏样本分布的最大区分度"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return np.max(tpr - fpr)

if __name__ == "__main__":
    # 读取清洗后的数据
    df = pd.read_csv("../data/clean_data.csv")
    target = "SeriousDlqin2yrs"

    # ===== IV>=0.02 筛选后的8个有效特征 =====
    feature_selected = [
        "RevolvingUtilizationOfUnsecuredLines",      # IV=1.06
        "NumberOfTimes90DaysLate",                   # IV=0.83
        "NumberOfTime30-59DaysPastDueNotWorse",      # IV=0.70
        "NumberOfTime60-89DaysPastDueNotWorse",      # IV=0.55
        "age",                                        # IV=0.25
        "DebtRatio",                                  # IV=0.059
        "NumberOfOpenCreditLinesAndLoans",            # IV=0.048
        "MonthlyIncome",                              # IV=0.040
    ]

    X = df[feature_selected]
    y = df[target]

    # 划分训练/测试集：stratify分层保证正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集 {X_train.shape}，测试集 {X_test.shape}")
    print(f"训练集违约占比 {y_train.mean():.4f}，测试集违约占比 {y_test.mean():.4f}\n")

    # ===== 模型1：逻辑回归（评分卡） =====
    # 逻辑回归对特征量纲敏感，先标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("===== 逻辑回归(评分卡模型) =====")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    p_lr = lr.predict_proba(X_test_s)[:, 1]

    lr_auc = roc_auc_score(y_test, p_lr)
    lr_ks = calc_ks(y_test, p_lr)
    print(f"AUC: {lr_auc:.4f}   KS: {lr_ks:.4f}")
    print("混淆矩阵(阈值0.5):")
    print(confusion_matrix(y_test, p_lr > 0.5))

    # ===== 模型2：XGBoost =====
    print("\n===== XGBoost模型 =====")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,

        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)
    p_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    xgb_auc = roc_auc_score(y_test, p_xgb)
    xgb_ks = calc_ks(y_test, p_xgb)
    print(f"AUC: {xgb_auc:.4f}   KS: {xgb_ks:.4f}")
    print("混淆矩阵(阈值0.5):")
    print(confusion_matrix(y_test, p_xgb > 0.5))

    # XGB特征重要性(Gain)
    print("\n==== XGB特征重要性(Gain) ====")
    gain_imp = pd.DataFrame({
        "feature": feature_selected,
        "gain": xgb_clf.feature_importances_
    }).sort_values("gain", ascending=False)
    print(gain_imp)

    # 保存模型对比结果，供写报告用
    result = pd.DataFrame({
        "model": ["LogisticRegression", "XGBoost"],
        "AUC": [lr_auc, xgb_auc],
        "KS": [lr_ks, xgb_ks]
    })
    result.to_csv("../output/model_compare.csv", index=False)
    print("\n模型对比结果已保存到 ../output/model_compare.csv")

import joblib
# 保存模型
joblib.dump(lr, filename="lr_model.pkl")
joblib.dump(xgb_clf, filename="xgb_model.pkl")
print("模型保存完成")




