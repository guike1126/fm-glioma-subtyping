import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 导入本地模块
from img_model_loader import create_datasets, get_stage_loader
from train_img_model import DeepImageTransformer

# ================= 配置部分（绝对路径） =================
SEEDS = [6003, 4873, 3157, 1107, 9468]  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== 绝对路径 =====
BASE_DIR = r"E:\学习\论文2\战桐整理"

CSV_FILE = r"E:\学习\论文2\战桐整理\imputed_data_OS_filled.csv"
FEATURES_DIR = r"E:\学习\论文2\战桐整理\h5_files\h5_files"
SPLIT_DIR = r"E:\学习\论文2\战桐整理"

# 模型权重和输出图片的路径（在code1文件夹下）
OUTPUT_BASE = r"E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\outputs_img_model"
AGGREGATED_OUTPUT = r"E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\img_model_results_5seed"

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(AGGREGATED_OUTPUT, exist_ok=True)

DATASET_CACHE = {}

# ================= 工具函数 =================

def add_label(ax, label):
    """在子图左上角添加加粗标号"""
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='right')

def get_shortened_names(class_names, classifier_name):
    short_names = []
    is_final = (classifier_name == 'Final Classification' or len(class_names) == 6)
    
    if classifier_name == 'Subtype':
        mapping = {'Astrocytoma': 'Astro', 'GBM': 'GBM', 'Oligodendroglioma': 'Oligo'}
        for name in class_names:
            short_names.append(mapping.get(name, name))
        return short_names
    
    for name in class_names:
        if not is_final:
            short_names.append(name)
            continue
        n = str(name).lower()
        if 'astro' in n and '2' in n: short_names.append('Ast_G2')
        elif 'astro' in n and '3' in n: short_names.append('Ast_G3')
        elif 'astro' in n and '4' in n: short_names.append('Ast_G4')
        elif 'oligo' in n and '2' in n: short_names.append('Oli_G2')
        elif 'oligo' in n and '3' in n: short_names.append('Oli_G3')
        elif 'gbm' in n or 'glioblastoma' in n: short_names.append('GBM_G4')
        else: short_names.append(name)
    return short_names

def get_cached_data(seed):
    """缓存并解包数据集"""
    if seed not in DATASET_CACHE:
        print(f"  CSV_FILE: {CSV_FILE}")
        print(f"  FEATURES_DIR: {FEATURES_DIR}")
        _, _, test_ds, classes = create_datasets(CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed)
        DATASET_CACHE[seed] = (test_ds, classes)
    return DATASET_CACHE[seed]

def get_task_mappings(classes, device):
    """标签映射逻辑"""
    sub_m, ast_m, oli_m = [], [], []
    for name in classes:
        parts = name.split('_')
        sub_name = parts[0]
        grade = int(parts[1].replace('G', ''))
        s_idx = 0 if 'Astrocytoma' in sub_name else (1 if 'GBM' in sub_name else 2)
        sub_m.append(s_idx)
        ast_m.append(grade - 2 if s_idx == 0 else -1)
        oli_m.append(grade - 2 if s_idx == 2 else -1)
    return {
        'subtype': torch.tensor(sub_m).to(device),
        'astro': torch.tensor(ast_m).to(device),
        'oligo': torch.tensor(oli_m).to(device)
    }

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

def plot_metrics_ax(ax, seed_metrics, title):
    df = pd.DataFrame(seed_metrics).melt(var_name="Metric", value_name="Score")
    sns.boxplot(data=df, x="Metric", y="Score", ax=ax, palette="Set3", width=0.5)
    sns.stripplot(data=df, x="Metric", y="Score", ax=ax, color="black", alpha=0.5, size=4)
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Metric: {title}", fontsize=15, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

# ================= 推理逻辑 =================

def collect_results(task, seeds):
    all_y_true, all_y_pred, all_y_score, seed_metrics = [], [], [], []
    for seed in seeds:
        test_ds, classes = get_cached_data(seed)
        mappings = get_task_mappings(classes, DEVICE)
        pth_path = os.path.join(OUTPUT_BASE, f"seed_{seed}", task['pth'])
        if not os.path.exists(pth_path):
            print(f"警告: 未找到 {pth_path}，跳过该seed")
            continue
            
        model = DeepImageTransformer(img_dim=512).to(DEVICE)
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()
        
        loader = get_stage_loader(test_ds, task['sub_f'], task['grade_f'], batch_size=1, is_train=False)
        y_true_s, y_score_s = [], []
        with torch.no_grad():
            for img, lbl, _ in loader:
                out = model(img.to(DEVICE))
                if task['map_key'] == 'final':
                    probs = torch.exp(out['final'])
                    target = lbl.to(DEVICE)
                else:
                    probs = torch.softmax(out[task['map_key']], dim=1)
                    target = mappings[task['map_key']][lbl.to(DEVICE)]
                    mask = (target != -1)
                    probs, target = probs[mask], target[mask]
                
                if target.numel() > 0:
                    y_score_s.extend(probs.cpu().numpy())
                    y_true_s.extend(target.cpu().numpy())
        
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
         'class_names': ['Astrocytoma', 'GBM', 'Oligodendroglioma'], 
         'sub_f': None, 'grade_f': None, 'map_key': 'subtype'},
        {'name': 'Astro Grade', 'pth': 'best_astro.pth', 'n_classes': 3, 
         'class_names': ['Grade 2', 'Grade 3', 'Grade 4'], 
         'sub_f': ['Astrocytoma'], 'grade_f': None, 'map_key': 'astro'},
        {'name': 'Oligo Grade', 'pth': 'best_oligo.pth', 'n_classes': 2, 
         'class_names': ['Grade 2', 'Grade 3'], 
         'sub_f': ['Oligodendroglioma'], 'grade_f': None, 'map_key': 'oligo'},
        {'name': 'Final Classification', 'pth': 'best_final.pth', 'n_classes': 6, 
         'class_names': None, 'sub_f': None, 'grade_f': None, 'map_key': 'final'}
    ]
    
    _, classes = get_cached_data(SEEDS[0])
    tasks[3]['class_names'] = classes

    # ===== 布局: 2行 × 4列 (横向) =====
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    axes_flat = axes.flatten()

    for i, task in enumerate(tasks):
        print(f">>> Analyzing Image Task: {task['name']}...")
        y_true, y_pred, y_score, seed_metrics = collect_results(task, SEEDS)
        
        if y_true is not None:
            plot_roc_ax(axes_flat[2*i], y_true, y_score, task['n_classes'], 
                        task['class_names'], task['name'])
            add_label(axes_flat[2*i], panel_labels[2*i])
            
            plot_metrics_ax(axes_flat[2*i+1], seed_metrics, task['name'])
            add_label(axes_flat[2*i+1], panel_labels[2*i+1])

    plt.tight_layout(h_pad=2.0, w_pad=3.0)
    save_path = os.path.join(AGGREGATED_OUTPUT, "img_roc_metrics_only_2x4.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    abs_path = os.path.abspath(save_path)
    print(f"\n✅ 修改完成！新图已保存至: {abs_path}")
    print("   布局: 2行×4列 (横向)，已移除混淆矩阵")

if __name__ == "__main__":
    main()