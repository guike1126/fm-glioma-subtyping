import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
import os
import re
from itertools import cycle
import pandas as pd
import seaborn as sns

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 导入本地模块
from multimodal_loader import create_multimodal_datasets, get_stage_loader
from train_multimodel import DeepMultimodalModel

# ================= 路径强制切换 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"当前工作目录已切换至: {os.getcwd()}")

# ================= 配置部分（绝对路径） =================
SEEDS = [6003, 4873, 3157, 1107, 9468]  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据路径
BASE_DIR = r"E:\学习\论文2\战桐整理"
CSV_FILE = r"E:\学习\论文2\战桐整理\imputed_data_OS_filled.csv"
FEATURES_DIR = r"E:\学习\论文2\战桐整理\h5_files\h5_files"
SPLIT_DIR = r"E:\学习\论文2\战桐整理"

# 模型权重路径（用户确认的位置）
OUTPUT_BASE = r"E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\outputs_multimodel"

# 图片输出路径（放在脚本所在目录下）
AGGREGATED_OUTPUT = r"E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\multimodal_results_5seed"

os.makedirs(AGGREGATED_OUTPUT, exist_ok=True)

print(f"BASE_DIR = {BASE_DIR}")
print(f"CSV_FILE = {CSV_FILE}")
print(f"FEATURES_DIR = {FEATURES_DIR}")
print(f"OUTPUT_BASE = {OUTPUT_BASE}")

# ================= 工具函数 =================

def get_mappings(classes, device):
    sub_m, astro_m, oligo_m = [], [], []
    for name in classes:
        n = name.lower()
        s = 0 if 'astro' in n else 1 if 'gbm' in n else 2
        sub_m.append(s)
        g_match = re.search(r'g(\d)', n)
        g = int(g_match.group(1)) if g_match else -1
        if s == 0: astro_m.append(0 if g==2 else 1 if g==3 else 2)
        else: astro_m.append(-1)
        if s == 2: oligo_m.append(0 if g==2 else 1)
        else: oligo_m.append(-1)
    return {
        'subtype': torch.tensor(sub_m).to(device),
        'astro': torch.tensor(astro_m).to(device),
        'oligo': torch.tensor(oligo_m).to(device)
    }

def add_label(ax, label):
    """在子图左上角添加加粗标号"""
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='right')

def get_shortened_names(class_names, classifier_name):
    short_names = []
    is_final = (classifier_name == 'Final Classification')
    
    if classifier_name == 'Subtype':
        mapping = {
            'Astrocytoma': 'Astro',
            'GBM': 'GBM',
            'Oligodendroglioma': 'Oligo'
        }
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

def plot_roc_on_ax(ax, y_true, y_score, n_classes, class_names, classifier_name):
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i] = [0, 1], [0, 1]
            roc_auc[i] = 0.5

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    display_names = get_shortened_names(class_names, classifier_name)
    
    for i, color in zip(range(n_classes), colors):
        label_name = display_names[i] if i < len(display_names) else f"Class {i}"
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUC={1:0.3f})'.format(label_name, roc_auc[i]))

    ax.plot([-0.02, 1.02], [-0.02, 1.02], 'k--', lw=1.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('1 - Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.set_title(f"ROC: {classifier_name}", fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11.5, framealpha=0.9, edgecolor='lightgrey')
    ax.grid(alpha=0.3)

def plot_boxplot_on_ax(ax, seed_metrics, classifier_name):
    df = pd.DataFrame(seed_metrics)
    df_melted = df.melt(var_name="Metric", value_name="Score")
    
    sns.boxplot(data=df_melted, x="Metric", y="Score", ax=ax, palette="Set3", width=0.5)
    sns.stripplot(data=df_melted, x="Metric", y="Score", ax=ax, color='black', alpha=0.6, size=5)
    
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Metric: {classifier_name}", fontsize=15, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Test results', fontsize=14)
    ax.tick_params(axis='x', labelsize=14)

def collect_predictions_for_task(task, seeds):
    all_y_true, all_y_pred, all_y_score, seed_metrics = [], [], [], []
    for seed in seeds:
        model_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        pth_path = os.path.join(model_dir, task['pth'])
        if not os.path.exists(pth_path):
            print(f"警告: 未找到 {pth_path}，跳过该seed")
            continue
        
        _, _, test_ds, classes, input_omics_dim = create_multimodal_datasets(
            CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed
        )
        mappings = get_mappings(classes, DEVICE)
        model = DeepMultimodalModel(omics_input_dim=input_omics_dim).to(DEVICE)
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()
        
        loader = get_stage_loader(test_ds, allowed_subtypes=task['target_filter'], batch_size=1, is_train=False)
        if not loader: continue
        
        y_true_s, y_score_s = [], []
        with torch.no_grad():
            for img, om, lbl, _ in loader:
                img, om, lbl = img.to(DEVICE), om.to(DEVICE), lbl.to(DEVICE)
                out = model(img, om)['logits']
                probs = torch.softmax(out[task['logit_key']], dim=1)
                target = mappings[task['map_key']][lbl] if task['map_key'] else lbl
                if task['map_key'] in ['astro', 'oligo']:
                    mask = target != -1
                    if mask.sum() == 0: continue
                    probs, target = probs[mask], target[mask]
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

def run_aggregated_evaluation():
    tasks = [
        {'name': 'Subtype', 'pth': 'best_subtype.pth', 'n_classes': 3, 
         'class_names': ['Astrocytoma', 'GBM', 'Oligodendroglioma'], 
         'target_filter': None, 'logit_key': 'subtype_logits', 'map_key': 'subtype'},
        {'name': 'Astro Grade', 'pth': 'best_astro_grade.pth', 'n_classes': 3, 
         'class_names': ['Grade 2', 'Grade 3', 'Grade 4'], 
         'target_filter': ['Astrocytoma'], 'logit_key': 'astro_grade_logits', 'map_key': 'astro'},
        {'name': 'Oligo Grade', 'pth': 'best_oligo_grade.pth', 'n_classes': 2, 
         'class_names': ['Grade 2', 'Grade 3'], 
         'target_filter': ['Oligodendroglioma'], 'logit_key': 'oligo_grade_logits', 'map_key': 'oligo'},
        {'name': 'Final Classification', 'pth': 'best_final.pth', 'n_classes': 6, 
         'class_names': None, 'target_filter': None, 'logit_key': 'final_logits', 'map_key': None}
    ]
    
    _, _, _, classes, _ = create_multimodal_datasets(
        CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, SEEDS[0]
    )
    tasks[3]['class_names'] = classes

    # ===== 修改：布局改为 2行 × 4列（横向），去掉混淆矩阵 =====
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    axes_flat = axes.flatten()

    for i, task in enumerate(tasks):
        print(f"Processing Task: {task['name']}...")
        y_true, y_pred, y_score, seed_metrics = collect_predictions_for_task(task, SEEDS)
        if y_true is None:
            continue
            
        # ROC 子图 (位置 2*i)
        plot_roc_on_ax(axes_flat[2*i], y_true, y_score, task['n_classes'], 
                       task['class_names'], task['name'])
        add_label(axes_flat[2*i], panel_labels[2*i])
        
        # Metrics 子图 (位置 2*i+1)
        plot_boxplot_on_ax(axes_flat[2*i+1], seed_metrics, task['name'])
        add_label(axes_flat[2*i+1], panel_labels[2*i+1])

    plt.tight_layout(h_pad=2.0, w_pad=3.0)
    save_path = os.path.join(AGGREGATED_OUTPUT, "multimodal_roc_metrics_only_2x4.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    abs_path = os.path.abspath(save_path)
    print(f"\n✅ 修改完成！新图已保存至: {abs_path}")
    print("   布局: 2行×4列 (横向)，已移除混淆矩阵")

if __name__ == "__main__":
    run_aggregated_evaluation()