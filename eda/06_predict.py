# src/06_predict.py
# 信贷风控项目：模型预测推理脚本
# 1. 严格遵循"训练时的规则不能变"原则：加载 03 保存的分箱边界和 WOE 映射
# 2. 加载 04 保存的模型和特征列表，保证特征顺序完全一致
# 3. 对新数据只做 transform，绝不重新 fit，避免数据泄露
# 4. 输出预测概率、违约标签（按最佳阈值切分）到 output/predictions.csv
#
# 【使用方法】
# 1. 将待预测的原始 csv 放到 data/ 目录，命名为 test_predict.csv
# 2. 确保字段与训练数据完全一致（缺少的字段会报错）
# 3. 运行本脚本，结果会保存到 output/predictions.csv

import pandas as pd
import numpy as np
import joblib
import os
import sys

# 待预测数据路径（使用者自行准备）
PREDICT_DATA_PATH = "../data/test_predict.csv"
OUTPUT_PATH = "../output/predictions.csv"


def load_assets():
    """加载所有训练时保存的规则和模型"""
    required_files = [
        "../models/woe_mappings.pkl",
        "../models/feature_bins.pkl",
        "../models/feature_columns.pkl",
        "../models/lr_model.pkl",
        "../models/xgb_model.pkl",
    ]

    # 检查文件是否存在
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ 缺少文件: {f}")
            print("请先按顺序运行 03_woe_iv_feature.py 和 04_model_train.py")
            sys.exit(1)

    woe_mappings = joblib.load("../models/woe_mappings.pkl")
    feature_bins = joblib.load("../models/feature_bins.pkl")
    feature_columns = joblib.load("../models/feature_columns.pkl")
    lr_model = joblib.load("../models/lr_model.pkl")
    xgb_model = joblib.load("../models/xgb_model.pkl")

    return woe_mappings, feature_bins, feature_columns, lr_model, xgb_model


def transform_single_feature(df, feature_col, woe_mapping, bins):
    """
    对单个特征应用训练时的分箱和 WOE 映射
    这里的所有逻辑必须和 03_woe_iv_feature.py 中的 transform_woe 完全一致
    """
    temp_bin_col = "bin_temp"

    # 判断特征类型，使用训练时的分箱规则
    if bins is None:
        # 离散特征，直接用原值
        df[temp_bin_col] = df[feature_col].astype(str)
    else:
        # 检查是否是逾期计数类特征
        if feature_col.startswith("NumberOfTime") or feature_col == "NumberOfTimes90DaysLate":
            df[temp_bin_col] = pd.cut(df[feature_col], bins=bins, right=True)
        else:
            df[temp_bin_col] = pd.cut(df[feature_col], bins=bins, right=True)

    # 映射到 WOE 值
    df[feature_col + "_WOE"] = df[temp_bin_col].astype(str).map(woe_mapping)

    # 处理未知分箱（测试集出现训练集没见过的值）
    mean_woe = np.mean(list(woe_mapping.values()))
    df[feature_col + "_WOE"] = df[feature_col + "_WOE"].fillna(mean_woe)

    return df


if __name__ == "__main__":
    # 1. 检查待预测数据是否存在
    if not os.path.exists(PREDICT_DATA_PATH):
        print(f"❌ 找不到待预测数据: {PREDICT_DATA_PATH}")
        print("请将待预测的 csv 文件放到 data/ 目录下，并命名为 test_predict.csv")
        sys.exit(1)

    # 2. 加载训练时的所有资产
    print("正在加载模型和特征工程规则...")
    woe_mappings, feature_bins, feature_columns, lr_model, xgb_model = load_assets()
    print(f"✅ 加载成功，模型使用的特征: {feature_columns}")

    # 3. 读取待预测数据
    print(f"\n正在读取待预测数据: {PREDICT_DATA_PATH}")
    df_predict = pd.read_csv(PREDICT_DATA_PATH)
    print(f"待预测样本数: {len(df_predict)}")

    # 4. 字段校验：确保待预测数据包含所有训练时用到的原始特征
    original_features = [col.replace("_WOE", "") for col in feature_columns]
    missing_cols = [col for col in original_features if col not in df_predict.columns]

    if missing_cols:
        print(f"❌ 待预测数据缺少以下字段: {missing_cols}")
        print("请确保字段名与训练数据完全一致")
        sys.exit(1)

    print("✅ 字段校验通过")

    # 5. 应用训练时的分箱和 WOE 映射
    print("\n正在应用 WOE 编码...")
    for feat in original_features:
        df_predict = transform_single_feature(
            df_predict, feat, woe_mappings[feat], feature_bins[feat]
        )

    # 6. 检查 WOE 特征是否全部生成
    missing_woe = [col for col in feature_columns if col not in df_predict.columns]
    if missing_woe:
        print(f"❌ WOE 编码后缺少特征: {missing_woe}")
        sys.exit(1)

    # 7. 按训练时的特征顺序排列
    X_predict = df_predict[feature_columns]

    # 8. 模型预测
    print("\n正在预测...")
    prob_lr = lr_model.predict_proba(X_predict)[:, 1]
    prob_xgb = xgb_model.predict_proba(X_predict)[:, 1]

    # 9. 使用最佳阈值（来自 05 评估结果）
    # 逻辑回归的最佳阈值是 0.4580（来自你的 05 输出）
    LR_BEST_THRESHOLD = 0.4580
    XGB_BEST_THRESHOLD = 0.0627

    label_lr = (prob_lr >= LR_BEST_THRESHOLD).astype(int)
    label_xgb = (prob_xgb >= XGB_BEST_THRESHOLD).astype(int)

    # 10. 构造输出结果
    result_df = pd.DataFrame({
        "lr_prob": prob_lr,
        "lr_prediction": label_lr,
        "xgb_prob": prob_xgb,
        "xgb_prediction": label_xgb,
    })

    # 11. 业务提示
    lr_bad_count = label_lr.sum()
    xgb_bad_count = label_xgb.sum()
    print(f"\n===== 预测结果 =====")
    print(f"逻辑回归预测违约用户: {lr_bad_count} 人 ({lr_bad_count / len(result_df) * 100:.2f}%)")
    print(f"XGBoost预测违约用户: {xgb_bad_count} 人 ({xgb_bad_count / len(result_df) * 100:.2f}%)")

    # 12. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ 预测结果已保存到: {OUTPUT_PATH}")
    print("\n业务解读：")
    print("  - lr_prob / xgb_prob: 预测违约概率 (0~1)")
    print("  - lr_prediction / xgb_prediction: 违约标签 (1=违约, 0=正常)")
    print("  - 可根据业务需求调整阈值，或对高分用户做人工复核")