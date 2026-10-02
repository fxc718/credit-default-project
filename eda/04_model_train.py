# src/04_model_train.py
# 信贷风控项目：逻辑回归 + XGBoost 模型训练与评估
# 1. 直接读取 03 脚本产出的 WOE 编码数据集 (train_woe.csv / test_woe.csv)，避免重复切分导致数据不一致
# 2. WOE 特征已无量纲，逻辑回归无需再做 StandardScaler
# 3. 同时训练 LR 与 XGBoost 做对比，体现"可解释性 vs 预测能力"的业务权衡
# 4. 输出 AUC、KS、混淆矩阵，KS 是风控业务更看重的指标
# 5. 模型统一保存到 models/ 目录，供 05_evaluate.py 和 06_predict.py 加载

import pandas as pd
import numpy as np
import joblib
import os
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb



def calc_ks(y_true, y_prob):
    """
    计算 KS 值 (风控核心指标)
    KS = max(TPR - FPR)，反映模型区分好坏样本的最大能力
    经验值：KS > 0.3 合格，KS > 0.4 优秀
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def evaluate_model(model_name, y_true, y_prob, threshold=0.5):
    """
    统一评估函数，输出 AUC、KS、混淆矩阵
    """
    auc = roc_auc_score(y_true, y_prob)
    ks = calc_ks(y_true, y_prob)
    cm = confusion_matrix(y_true, y_prob > threshold)

    print(f"\n===== {model_name} 评估结果 =====")
    print(f"AUC: {auc:.4f}")
    print(f"KS : {ks:.4f}")
    print(f"混淆矩阵 (阈值={threshold}):")
    print(cm)

    return {"model": model_name, "auc": auc, "ks": ks}


if __name__ == "__main__":
    # 确保 models 目录存在
    os.makedirs("../models", exist_ok=True)

    # 1. 读取 03 脚本产出的 WOE 编码数据集
    print("正在读取 WOE 编码后的数据...")
    train_df = pd.read_csv("../data/train_woe.csv")
    test_df = pd.read_csv("../data/test_woe.csv")

    target = "SeriousDlqin2yrs"
    features = [col for col in train_df.columns if col != target]

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集违约率: {y_train.mean():.4f}, 测试集违约率: {y_test.mean():.4f}")

    results = []

    # =====================================================
    # 模型 1：逻辑回归 (评分卡模型)
    # WOE 编码后特征已是无量纲，无需 StandardScaler，系数即可解释
    # =====================================================
    print("\n===== 训练逻辑回归模型 (评分卡) =====")
    lr = LogisticRegression(
        max_iter=500,
        random_state=42,
        class_weight="balanced"  # 处理样本不平衡，提升违约样本召回
    )
    lr.fit(X_train, y_train)

    p_lr = lr.predict_proba(X_test)[:, 1]
    res_lr = evaluate_model("逻辑回归", y_test, p_lr)
    results.append(res_lr)

    # 保存逻辑回归模型 + 特征列表
    joblib.dump(lr, "../models/lr_model.pkl")
    joblib.dump(features, "../models/feature_columns.pkl")

    # =====================================================
    # 模型 2：XGBoost
    # =====================================================
    print("\n===== 训练 XGBoost 模型 =====")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_clf.fit(X_train, y_train)

    p_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    res_xgb = evaluate_model("XGBoost", y_test, p_xgb)
    results.append(res_xgb)

    # 保存 XGBoost 模型
    joblib.dump(xgb_clf, "../models/xgb_model.pkl")

    # =====================================================
    # 模型对比汇总
    # =====================================================
    print("\n===== 模型对比汇总 =====")
    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False))

    result_df.to_csv("../output/model_comparison.csv", index=False,encoding='utf-8-sig')
    print("\n模型对比结果已保存到 output/model_comparison.csv")




