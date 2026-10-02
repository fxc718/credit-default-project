# src/05_model_evaluate.py
# 信贷风控项目：模型评估与可视化
# 1. 加载 04 脚本训练好的模型和 03 脚本产出的测试集 WOE 数据
# 2. 绘制风控核心图表：ROC 曲线、KS 曲线、分数分布图、阈值权衡图
# 3. 计算 PSI (Population Stability Index)，衡量训练集与测试集分布稳定性
# 4. 所有图表保存到 output/charts/ 目录

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统使用黑体
plt.rcParams['axes.unicode_minus'] = False


def calc_ks(y_true, y_prob):
    """计算 KS 值"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def calc_psi(expected, actual, bins=10):
    """
    计算 PSI (群体稳定性指标)
    PSI < 0.1: 稳定性很好
    0.1 <= PSI < 0.25: 需要关注
    PSI >= 0.25: 稳定性差，模型需要重新训练
    """
    # 基于预期(训练集)分布进行分箱
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)

    # 统计各箱占比
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # 防止出现 0
    expected_percents = np.clip(expected_percents, 0.0001, 1)
    actual_percents = np.clip(actual_percents, 0.0001, 1)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value


def plot_roc_ks(y_true, y_prob, model_name, save_dir):
    """绘制 ROC 和 KS 曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ks_values = tpr - fpr
    max_ks_idx = np.argmax(ks_values)
    max_ks = ks_values[max_ks_idx]
    best_threshold = thresholds[max_ks_idx]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1. ROC 曲线
    ax[0].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', color='blue')
    ax[0].plot([0, 1], [0, 1], 'k--', label='随机猜测')
    ax[0].set_xlabel('False Positive Rate (FPR)')
    ax[0].set_ylabel('True Positive Rate (TPR)')
    ax[0].set_title(f'{model_name} - ROC Curve')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # 2. KS 曲线
    ax[1].plot(thresholds, tpr, label='TPR (累积好样本比例)', color='green')
    ax[1].plot(thresholds, fpr, label='FPR (累积坏样本比例)', color='red')
    ax[1].axvline(best_threshold, color='black', linestyle='--',
                  label=f'最佳阈值={best_threshold:.4f}\nmax KS={max_ks:.4f}')
    ax[1].set_xlabel('Probability Threshold')
    ax[1].set_ylabel('Cumulative Rate')
    ax[1].set_title(f'{model_name} - KS Curve')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_ROC_KS.png'), dpi=150)
    plt.close()

    return max_ks, best_threshold


def plot_score_distribution(y_true, y_prob, model_name, save_dir):
    """绘制违约与正常用户的分数分布图"""
    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_prob[y_true == 0], label='正常用户 (Label=0)', fill=True, color='green')
    sns.kdeplot(y_prob[y_true == 1], label='违约用户 (Label=1)', fill=True, color='red')
    plt.axvline(0.5, color='black', linestyle='--', label='默认阈值 0.5')
    plt.xlabel('预测违约概率')
    plt.ylabel('密度')
    plt.title(f'{model_name} - 预测概率分布图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_Score_Distribution.png'), dpi=150)
    plt.close()


def plot_threshold_tradeoff(y_true, y_prob, model_name, save_dir):
    """绘制阈值与通过率、坏账率的权衡图"""
    thresholds = np.linspace(0, 1, 100)
    pass_rates = []
    bad_rates = []
    total_bad = sum(y_true == 1)
    total_good = sum(y_true == 0)

    for t in thresholds:
        # 预测为坏样本（拒绝）
        rejected = y_prob >= t
        # 预测为好样本（通过）
        passed = ~rejected

        pass_rates.append(passed.sum() / len(y_true))
        # 在通过的客户中，实际坏样本的比例（坏账率）
        if passed.sum() > 0:
            bad_rates.append(y_true[passed].sum() / passed.sum())
        else:
            bad_rates.append(0)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, pass_rates, label='通过率 (Approval Rate)', color='blue')
    plt.plot(thresholds, bad_rates, label='坏账率 (Bad Rate)', color='red')
    plt.xlabel('阈值 (Threshold)')
    plt.ylabel('比率')
    plt.title(f'{model_name} - 阈值与业务指标权衡')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_Threshold_Tradeoff.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    # 1. 准备目录
    os.makedirs("../output/charts", exist_ok=True)

    # 2. 加载数据
    print("正在读取数据与模型...")
    train_df = pd.read_csv("../data/train_woe.csv")
    test_df = pd.read_csv("../data/test_woe.csv")
    target = "SeriousDlqin2yrs"

    features = joblib.load("../models/feature_columns.pkl")
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # 3. 加载模型
    lr_model = joblib.load("../models/lr_model.pkl")
    xgb_model = joblib.load("../models/xgb_model.pkl")

    # 4. 生成预测概率
    p_lr = lr_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # 5. 评估与绘图
    print("\n===== 开始生成图表 =====")

    print("正在评估逻辑回归...")
    lr_ks, lr_best_threshold = plot_roc_ks(y_test, p_lr, "LR", "../output/charts")
    plot_score_distribution(y_test, p_lr, "LR", "../output/charts")
    plot_threshold_tradeoff(y_test, p_lr, "LR", "../output/charts")

    print("正在评估 XGBoost...")
    xgb_ks, xgb_best_threshold = plot_roc_ks(y_test, p_xgb, "XGBoost", "../output/charts")
    plot_score_distribution(y_test, p_xgb, "XGBoost", "../output/charts")
    plot_threshold_tradeoff(y_test, p_xgb, "XGBoost", "../output/charts")

    # 6. PSI 稳定性计算 (基于预测概率)
    print("\n===== PSI 稳定性分析 (基于预测分数) =====")
    p_train_lr = lr_model.predict_proba(X_train)[:, 1]
    psi_lr = calc_psi(p_train_lr, p_lr)

    p_train_xgb = xgb_model.predict_proba(X_train)[:, 1]
    psi_xgb = calc_psi(p_train_xgb, p_xgb)

    print(
        f"逻辑回归 PSI: {psi_lr:.4f}  ({'稳定性好' if psi_lr < 0.1 else '需要关注' if psi_lr < 0.25 else '稳定性差'})")
    print(
        f"XGBoost PSI: {psi_xgb:.4f}  ({'稳定性好' if psi_xgb < 0.1 else '需要关注' if psi_xgb < 0.25 else '稳定性差'})")

    # 7. 输出最终总结
    print("\n===== 最终评估总结 =====")
    print(
        f"逻辑回归 - AUC: {roc_auc_score(y_test, p_lr):.4f}, KS: {lr_ks:.4f}, 最佳阈值: {lr_best_threshold:.4f}, PSI: {psi_lr:.4f}")
    print(
        f"XGBoost  - AUC: {roc_auc_score(y_test, p_xgb):.4f}, KS: {xgb_ks:.4f}, 最佳阈值: {xgb_best_threshold:.4f}, PSI: {psi_xgb:.4f}")

    print("\n所有图表已保存到 output/charts/ 目录")