import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from omics_model import BranchNet, UnifiedCascadeModel, CascadeDataset, get_hierarchy_info

# ================= 配置 =================
SEEDS = [6003, 4873, 3157, 1107, 9468]  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "E:/学习/论文2/战桐整理/代码和结果/代码和结果"
FEATURE_DIR_PATTERN = os.path.join(BASE_PATH, "vae_best_features_{seed}")
MODEL_DIR_PATTERN = os.path.join(BASE_PATH, "omics_model_{seed}")
AGGREGATED_OUTPUT = "./omics_model_results_5seed - 2"

os.makedirs(AGGREGATED_OUTPUT, exist_ok=True)

# ================= 工具函数 =================

def add_label(ax, label):
    """在子图左上角添加加粗标号"""
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='right')

def get_shortened_names(class_names, classifier_name):
    short_names = []
    is_final = (classifier_name == 'Unified Cascade' or len(class_names) == 6)
    
    if classifier_name == 'Subtype':
        mapping = {'Astrocytoma': 'Astro', 'GBM': 'GBM', 'Oligodendroglioma': 'Oligo'}
        for name in class_names:
            short_names.append(mapping.get(name, name))
        return short_names
    
    if not is_final:
        short_names = class_names[:]
    else:
        for name in class_names:
            n = str(name).lower()
            if 'astro' in n and '2' in n: short_names.append('Ast_G2')
            elif 'astro' in n and '3' in n: short_names.append('Ast_G3')
            elif 'astro' in n and '4' in n: short_names.append('Ast_G4')
            elif 'oligo' in n and '2' in n: short_names.append('Oli_G2')
            elif 'oligo' in n and '3' in n: short_names.append('Oli_G3')
            elif 'gbm' in n or 'glioblastoma' in n: short_names.append('GBM_G4')
            else: short_names.append(name)
    return short_names

# ================= 绘图函数 =================

def plot_roc_ax(ax, y_true, y_score, n_classes, class_names, title):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    display_names = get_shortened_names(class_names, title)
    
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, 
                label='{0} (AUC={1:0.3f})'.format(display_names[i], roc_auc[i]))

    ax.plot([-0.02, 1.02], [-0.02, 1.02], 'k--', lw=1.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('1 - Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.set_title(f"ROC: {title}", fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11.5, framealpha=0.9, edgecolor='lightgrey')
    ax.grid(alpha=0.3)

def plot_metrics_ax(ax, metrics_list, title):
    df = pd.DataFrame(metrics_list).melt(var_name="Metric", value_name="Score")
    sns.boxplot(data=df, x="Metric", y="Score", ax=ax, palette="Set3", width=0.5)
    sns.stripplot(data=df, x="Metric", y="Score", ax=ax, color="black", alpha=0.5, size=4)
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Metric: {title}", fontsize=15, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Test results', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)

# ================= 推理核心逻辑 =================

def collect_predictions(task, seeds):
    all_y_true, all_y_pred, all_y_score, seed_metrics = [], [], [], []
    
    for seed in seeds:
        f_dir = FEATURE_DIR_PATTERN.format(seed=seed)
        m_dir = MODEL_DIR_PATTERN.format(seed=seed)
        pth_path = os.path.join(m_dir, task['pth'])
        if not os.path.exists(pth_path): 
            continue

        l2i, hier, _ = get_hierarchy_info(f_dir)
        
        if task['name'] == "Unified Cascade":
            sub = BranchNet(128, 3)
            ast = BranchNet(128, 3)
            oli = BranchNet(128, 2)
            model = UnifiedCascadeModel(sub, ast, oli, hier).to(DEVICE)
        else:
            model = BranchNet(128, task['n_classes']).to(DEVICE)
            
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()

        test_ds = CascadeDataset(f_dir, 'test', l2i, hier, filter_sub=task['fs'], task=task['type'])
        loader = DataLoader(test_ds, batch_size=32)

        y_true_s, y_score_s = [], []
        with torch.no_grad():
            for x, y in loader:
                out = model(x.to(DEVICE))
                probs = torch.exp(out) if task['name'] == "Unified Cascade" else torch.softmax(out, dim=1)
                y_score_s.extend(probs.cpu().numpy())
                y_true_s.extend(y.numpy())

        if y_true_s:
            y_true_s, y_score_s = np.array(y_true_s), np.array(y_score_s)
            y_pred_s = np.argmax(y_score_s, axis=1)
            seed_metrics.append({
                'Accuracy': accuracy_score(y_true_s, y_pred_s),
                'Precision': precision_score(y_true_s, y_pred_s, average='macro', zero_division=0),
                'Recall': recall_score(y_true_s, y_pred_s, average='macro', zero_division=0),
                'F1': f1_score(y_true_s, y_pred_s, average='macro', zero_division=0)
            })
            all_y_true.append(y_true_s)
            all_y_pred.append(y_pred_s)
            all_y_score.append(y_score_s)
            
    if not all_y_true: 
        return None, None, None, None
    return (np.concatenate(all_y_true), np.concatenate(all_y_pred), 
            np.concatenate(all_y_score), seed_metrics)

# ================= 主函数 =================

def main():
    tasks = [
        {'name': 'Subtype', 'pth': 'best_subtype.pth', 'n_classes': 3, 
         'class_names': ["Astrocytoma", "Oligodendroglioma", "GBM"], 'fs': None, 'type': 'sub'},
        {'name': 'Astro Grade', 'pth': 'best_ast_grade.pth', 'n_classes': 3, 
         'class_names': ["Grade 2", "Grade 3", "Grade 4"], 'fs': 0, 'type': 'grade'},
        {'name': 'Oligo Grade', 'pth': 'best_oli_grade.pth', 'n_classes': 2, 
         'class_names': ["Grade 2", "Grade 3"], 'fs': 1, 'type': 'grade'},
        {'name': 'Unified Cascade', 'pth': 'best_unified_cascade.pth', 'n_classes': 6, 
         'class_names': None, 'fs': None, 'type': '6class'}
    ]
    
    _, _, names_6 = get_hierarchy_info(FEATURE_DIR_PATTERN.format(seed=SEEDS[0]))
    tasks[3]['class_names'] = names_6

    # ===== 布局改为 2行 × 4列 (横向拉宽) =====
    # 总共8个子图，每行4个，共2行
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 宽度放大，高度缩小

    # 面板标签：按从左到右、从上到下的顺序
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    # 将axes展平为一维，便于顺序索引
    axes_flat = axes.flatten()

    # 每个任务分配两个连续的子图位置：ROC 和 Metrics
    for i, task in enumerate(tasks):
        print(f">>> Analyzing Omics Task: {task['name']}...")
        y_true, y_pred, y_score, seed_metrics = collect_predictions(task, SEEDS)
        
        if y_true is not None:
            display_title = 'Final Classification' if task['name'] == 'Unified Cascade' else task['name']
            
            # ROC 子图 (位置 2*i)
            plot_roc_ax(axes_flat[2*i], y_true, y_score, task['n_classes'], 
                        task['class_names'], display_title)
            add_label(axes_flat[2*i], panel_labels[2*i])
            
            # Metrics 子图 (位置 2*i+1)
            plot_metrics_ax(axes_flat[2*i+1], seed_metrics, display_title)
            add_label(axes_flat[2*i+1], panel_labels[2*i+1])

    plt.tight_layout(h_pad=2.0, w_pad=3.0)  # 适当调整间距
    
    # 保存为新文件，避免覆盖
    save_path = os.path.join(AGGREGATED_OUTPUT, "omics_roc_metrics_only_2x4.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印绝对路径，方便查找
    abs_path = os.path.abspath(save_path)
    print(f"\n✅ 修改完成！新图已保存至: {abs_path}")
    print("   布局已改为 2行×4列 (横向)")

if __name__ == "__main__":
    main()