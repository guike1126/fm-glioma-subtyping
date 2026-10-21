import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import re
from itertools import cycle

# ================= 修改点 1: 导入你的图片模型相关模块 =================
from img_model_loader import create_datasets, get_stage_loader  # 对应图片数据加载器
from train_img_model import DeepImageTransformer              # 对应纯图片 Transformer 模型

# ================= 配置部分 =================
SEED = 2026  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径配置
BASE_DIR = "/seu_nvme/home/liuhongde/220232427"
CSV_FILE = os.path.join(BASE_DIR, "imputed_data_OS_filled.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "features_h5")
SPLIT_DIR = BASE_DIR
OUTPUT_BASE = "./outputs_transformer_only" 

# ================= 工具函数 =================

def get_mappings(classes, device):
    """重建标签映射逻辑，与 train_img_model 中的 _init_mappings 保持一致"""
    sub_m, astro_m, oligo_m = [], [], []
    for name in classes:
        parts = name.split('_')
        sub_name, grade = parts[0], int(parts[1].replace('G', ''))
        
        # Subtype mapping
        s = 0 if 'Astrocytoma' in sub_name else (1 if 'GBM' in sub_name else 2)
        sub_m.append(s)
        
        # Grade mapping (Grade-2 -> 0, etc.)
        astro_m.append(grade - 2 if s == 0 else -1)
        oligo_m.append(grade - 2 if s == 2 else -1)

    return {
        'subtype': torch.tensor(sub_m).to(device),
        'astro': torch.tensor(astro_m).to(device),
        'oligo': torch.tensor(oligo_m).to(device)
    }

def draw_roc_on_ax(ax, y_true, y_score, n_classes, class_names, title):
    """(保持不变) 在指定的 Axes 对象上绘制 ROC 曲线"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = dict(); tpr = dict(); roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro/Macro 计算
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘图逻辑
    ax.plot(fpr["micro"], tpr["micro"], label=f'Micro (area = {roc_auc["micro"]:0.4f})', color='deeppink', linestyle=':', lw=3)
    ax.plot(fpr["macro"], tpr["macro"], label=f'Macro (area = {roc_auc["macro"]:0.4f})', color='navy', linestyle=':', lw=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        label_name = class_names[i] if i < len(class_names) else f"Class {i}"
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f'{label_name} ({roc_auc[i]:0.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(title); ax.legend(loc="lower right", fontsize='x-small'); ax.grid(alpha=0.3)

def run_evaluation(seed):
    print(f"\n{'='*40}\nRunning Image-Only Evaluation for SEED: {seed}\n{'='*40}")
    
    model_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return

    # ================= 修改点 2: 加载纯图片数据集 =================
    _, _, test_ds, classes = create_datasets(
        CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed
    )
    mappings = get_mappings(classes, DEVICE)
    
    # 修改点 3: 初始化 DeepImageTransformer (无需 input_omics_dim)
    model = DeepImageTransformer(img_dim=512, num_layers=2).to(DEVICE)
    model.eval()

    # ================= 任务定义 (与 train_img_model 的 key 对应) =================
    tasks = [
        {'name': 'subtype', 'pth': 'best_subtype.pth', 'n_classes': 3, 'class_names': ['Astrocytoma', 'GBM', 'Oligodendroglioma'], 'target_filter': ['Astrocytoma', 'Oligodendroglioma', 'GBM'], 'logit_key': 'subtype', 'map_key': 'subtype'},
        {'name': 'astro_grade', 'pth': 'best_astro.pth', 'n_classes': 3, 'class_names': ['Grade 2', 'Grade 3', 'Grade 4'], 'target_filter': ['Astrocytoma'], 'logit_key': 'astro', 'map_key': 'astro'},
        {'name': 'oligo_grade', 'pth': 'best_oligo.pth', 'n_classes': 2, 'class_names': ['Grade 2', 'Grade 3'], 'target_filter': ['Oligodendroglioma'], 'logit_key': 'oligo', 'map_key': 'oligo'},
        {'name': 'final', 'pth': 'best_final.pth', 'n_classes': 6, 'class_names': classes, 'target_filter': None, 'logit_key': 'final', 'map_key': None}
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    task_plotted_count = 0

    # ================= 推理 =================
    for i, task in enumerate(tasks):
        pth_path = os.path.join(model_dir, task['pth'])
        if not os.path.exists(pth_path):
            print(f"  ⚠️ Skipping {task['name']}: Weight file not found.")
            continue
            
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE, weights_only=False))
        
        # 修改点 4: 使用图片模型的 get_stage_loader (参数不含 omics)
        loader = get_stage_loader(test_ds, allowed_subtypes=task['target_filter'], batch_size=1, is_train=False)
        
        if not loader: continue

        y_score_list, y_true_list = [], []
        with torch.no_grad():
            for img, lbl, _ in loader:  # 只有 img, lbl, path
                img, lbl = img.to(DEVICE), lbl.to(DEVICE)
                
                # 修改点 5: 调用模型并处理 Final 任务的 Log 概率
                out = model(img)
                logits = out[task['logit_key']]
                
                if task['logit_key'] == 'final':
                    # final 已经是 log_softmax，需要 exp 转回概率
                    probs = torch.exp(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                
                target = mappings[task['map_key']][lbl] if task['map_key'] else lbl
                
                if task['map_key'] in ['astro', 'oligo']:
                    mask = target != -1
                    if mask.sum() == 0: continue
                    probs, target = probs[mask], target[mask]
                
                y_score_list.extend(probs.cpu().numpy())
                y_true_list.extend(target.cpu().numpy())

        if len(y_true_list) > 0:
            draw_roc_on_ax(axes[i], np.array(y_true_list), np.array(y_score_list), 
                          task['n_classes'], task['class_names'], f"ROC: {task['name'].upper()}")
            task_plotted_count += 1

    if task_plotted_count > 0:
        plt.suptitle(f"Image-Only Multi-task ROC (Seed {seed})", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(model_dir, f"combined_auc_roc_image_only.png"), dpi=300)
        plt.close()
        print(f"✅ Saved ROC grid to {model_dir}")

if __name__ == "__main__":
    run_evaluation(SEED)