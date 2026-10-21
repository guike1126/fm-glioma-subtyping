import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import re
from itertools import cycle

# 导入您的原始模块
from multimodal_loader import create_multimodal_datasets, get_stage_loader
from train_multimodel import DeepMultimodalModel

# ================= 配置部分 =================
# 修改这里：将单个 SEED 改为 SEEDS 列表
SEEDS = [9468, 1107, 3157]  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径配置
BASE_DIR = "/seu_nvme/home/liuhongde/220232427"
CSV_FILE = os.path.join(BASE_DIR, "imputed_data_OS_filled.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "features_h5")
SPLIT_DIR = BASE_DIR
OUTPUT_BASE = "./outputs_multimodel" 

# ================= 工具函数 =================

def get_mappings(classes, device):
    """重建标签映射逻辑"""
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

def draw_roc_on_ax(ax, y_true, y_score, n_classes, class_names, title):
    """在指定的 Axes 对象上绘制 ROC 曲线"""
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 计算每一类的ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算 Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算 Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘图
    ax.plot(fpr["micro"], tpr["micro"],
             label='Micro-avg (area = {0:0.4f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)

    ax.plot(fpr["macro"], tpr["macro"],
             label='Macro-avg (area = {0:0.4f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        label_name = class_names[i] if i < len(class_names) else f"Class {i}"
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='{0} (area = {1:0.4f})'.format(label_name, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize='small', framealpha=0.5)
    ax.grid(alpha=0.3)

def run_evaluation(seed):
    print(f"\n{'='*40}")
    print(f"🚀 Running Combined Evaluation for SEED: {seed}")
    print(f"{'='*40}")
    
    model_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print(f"⚠️ Skipping seed {seed}...")
        return

    # 1. 加载数据 (使用当前循环的 seed)
    _, _, test_ds, classes, input_omics_dim = create_multimodal_datasets(
        CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed
    )
    
    mappings = get_mappings(classes, DEVICE)
    model = DeepMultimodalModel(omics_input_dim=input_omics_dim).to(DEVICE)
    model.eval()

    # ================= 任务定义 =================
    tasks = [
        {'name': 'subtype', 'pth': 'best_subtype.pth', 'n_classes': 3, 'class_names': ['Astrocytoma', 'GBM', 'Oligodendroglioma'], 'target_filter': None, 'logit_key': 'subtype_logits', 'map_key': 'subtype'},
        {'name': 'astro_grade', 'pth': 'best_astro_grade.pth', 'n_classes': 3, 'class_names': ['Grade 2', 'Grade 3', 'Grade 4'], 'target_filter': ['Astrocytoma'], 'logit_key': 'astro_grade_logits', 'map_key': 'astro'},
        {'name': 'oligo_grade', 'pth': 'best_oligo_grade.pth', 'n_classes': 2, 'class_names': ['Grade 2', 'Grade 3'], 'target_filter': ['Oligodendroglioma'], 'logit_key': 'oligo_grade_logits', 'map_key': 'oligo'},
        {'name': 'final', 'pth': 'best_final.pth', 'n_classes': 6, 'class_names': classes, 'target_filter': None, 'logit_key': 'final_logits', 'map_key': None}
    ]

    # 准备画布：2行2列
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    task_plotted_count = 0

    # ================= 推理与存储结果 =================
    for i, task in enumerate(tasks):
        pth_path = os.path.join(model_dir, task['pth'])
        
        # 简单检查文件是否存在
        if not os.path.exists(pth_path):
            print(f"  ⚠️ Weight file not found for task '{task['name']}': {pth_path}")
            continue
            
        print(f"  ... Processing Task: {task['name']}")
        
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        
        # 获取Loader (测试集)
        loader = get_stage_loader(test_ds, allowed_subtypes=task['target_filter'], batch_size=1, is_train=False)
        
        if not loader: 
            print(f"  ⚠️ No data found for task '{task['name']}' in this split.")
            continue

        y_score_list, y_true_list = [], []
        
        with torch.no_grad():
            for img, om, lbl, _ in loader:
                img, om, lbl = img.to(DEVICE), om.to(DEVICE), lbl.to(DEVICE)
                out = model(img, om)['logits']
                probs = torch.softmax(out[task['logit_key']], dim=1)
                
                target = mappings[task['map_key']][lbl] if task['map_key'] else lbl
                
                # 过滤掉标签为 -1 的无效样本
                if task['map_key'] in ['astro', 'oligo']:
                    mask = target != -1
                    if mask.sum() == 0: continue
                    probs, target = probs[mask], target[mask]
                
                y_score_list.extend(probs.cpu().numpy())
                y_true_list.extend(target.cpu().numpy())

        if len(y_true_list) > 0:
            draw_roc_on_ax(
                axes[i], 
                np.array(y_true_list), 
                np.array(y_score_list), 
                task['n_classes'], 
                task['class_names'], 
                f"ROC: {task['name'].upper()}"
            )
            task_plotted_count += 1

    # 保存合并后的大图
    if task_plotted_count > 0:
        plt.suptitle(f"Multi-task ROC Curves (Seed {seed})", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        combined_save_path = os.path.join(model_dir, f"combined_auc_roc_grid.png")
        plt.savefig(combined_save_path, dpi=300)
        plt.close()
        print(f"✅ Saved combined ROC grid to: {combined_save_path}")
    else:
        print(f"⚠️ No tasks were plotted for seed {seed}.")
        plt.close()

if __name__ == "__main__":
    total_seeds = len(SEEDS)
    print(f"Started Batch Evaluation for {total_seeds} seeds: {SEEDS}")
    
    for idx, seed in enumerate(SEEDS):
        try:
            run_evaluation(seed)
            print(f"Progress: {idx + 1}/{total_seeds} seeds completed.\n")
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR running seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to next seed...\n")
    
    print("All tasks finished.")