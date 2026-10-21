import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import roc_curve, auc

# 导入您的原始模块
from multimodal_loader import create_multimodal_datasets, get_stage_loader
from train_multimodal_v2 import DeepMultimodalModel

# ================= 1. 配置与路径 =================
SEEDS = [6003, 4873, 3157, 1107, 9468]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 基础路径
BASE_DIR = "/seu_nvme/home/liuhongde/220232427"
CSV_FILE = os.path.join(BASE_DIR, "imputed_data_OS_filled.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "features_h5")
SPLIT_DIR = BASE_DIR
OUTPUT_BASE = "./outputs_multimodel"
ROC_DATA_DIR = "/seu_nvme/home/liuhongde/220232427/fm-glioma-subtyping-main/roc_data" 

# ROAM 特有路径
ROAM_RES_PATH = '/seu_nvme/home/liuhongde/220232427/ROAM-main/ROAM/prediction_results/s1/int_glioma_tumor_subtyping/results.json'
ROAM_NPY_PATH = os.path.join(BASE_DIR, '/seu_nvme/home/liuhongde/220232427/ROAM-main/data_prepare/data_split/tcga_test_split.npy')

AGGREGATED_OUTPUT = "./4_model_compared_results"
os.makedirs(AGGREGATED_OUTPUT, exist_ok=True)

# ================= 2. 核心合并逻辑 (二分类) =================

def merge_to_binary(y_true_3class, y_score_3class):
    """
    统一二分类逻辑：
    标签: 0,1(Astro/GBM) -> 0; 2(Oligo) -> 1
    概率: 取原索引 2 (Oligo) 的概率
    """
    y_true_bin = np.where(y_true_3class == 2, 1, 0)
    y_score_bin = y_score_3class[:, 2] 
    return y_true_bin, y_score_bin

# ================= 3. 各模型数据采集 =================

def collect_ours_binary(seeds):
    """采集 Ours 模型结果并转为二分类"""
    all_y_true, all_y_score = [], []
    for seed in seeds:
        model_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        pth_path = os.path.join(model_dir, 'best_subtype.pth')
        if not os.path.exists(pth_path): continue
        
        _, _, test_ds, classes, input_dim = create_multimodal_datasets(CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed)
        sub_m = [0 if 'astro' in n.lower() else 1 if 'gbm' in n.lower() else 2 for n in classes]
        mapping = torch.tensor(sub_m).to(DEVICE)
        
        model = DeepMultimodalModel(omics_input_dim=input_dim).to(DEVICE)
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()
        
        loader = get_stage_loader(test_ds, allowed_subtypes=None, batch_size=1, is_train=False)
        with torch.no_grad():
            for img, om, lbl, _ in loader:
                out = model(img.to(DEVICE), om.to(DEVICE))['logits']['subtype_logits']
                all_y_score.append(torch.softmax(out, dim=1).cpu().numpy().flatten())
                all_y_true.append(mapping[lbl].cpu().numpy().flatten())
                
    return merge_to_binary(np.array(all_y_true).flatten(), np.array(all_y_score))

def load_external_binary(model_name, k=5):
    """加载 CLAM/Mamba npy 并转为二分类"""
    data_dir = os.path.join(ROC_DATA_DIR, model_name)
    all_probs, all_labels = [], []
    for f in range(k):
        p_path, l_path = os.path.join(data_dir, f'probs_fold_{f}.npy'), os.path.join(data_dir, f'labels_fold_{f}.npy')
        if os.path.exists(p_path):
            all_probs.append(np.load(p_path))
            all_labels.append(np.load(l_path))
    if not all_probs: return None, None
    return merge_to_binary(np.concatenate(all_labels), np.concatenate(all_probs))

def load_roam_binary():
    """加载 ROAM 结果并适配二分类逻辑"""
    if not (os.path.exists(ROAM_RES_PATH) and os.path.exists(ROAM_NPY_PATH)):
        print("⚠️ ROAM 结果文件缺失，跳过 ROAM。")
        return None, None
        
    with open(ROAM_RES_PATH, 'r') as f:
        res_data = json.load(f)
    probs = np.array(res_data['test']['probs']) # [N, 3]

    test_info = np.load(ROAM_NPY_PATH)
    raw_labels = test_info[1].astype(int)
    
    # 按照 ROAM 映射: 3,4 是 Oligo (1)，其余是 Others (0)
    y_true = np.array([1 if l in [3, 4] else 0 for l in raw_labels])
    
    # 统一使用 Index 1 (Oligo) 的得分，这与 ROAM 原本的逻辑一致
    # 注意：确保 ROAM 的 probs 顺序与我们的二分类定义（Oligo居中或在末尾）匹配
    # 根据 auc.py，ROAM 的 Index 1 是 Oligo
    y_scores_pos = probs[:, 1]
    
    return y_true, y_scores_pos

# ================= 4. 绘图 =================

def plot_combined_roc(results_dict, save_path):
    plt.figure(figsize=(8, 8))
    
    # 线条配置：粗细统一为 2.5
    configs = {
        'Multimodal Fusion (Ours)': {'color': '#d62728', 'lw': 2.5},
        'Mamba': {'color': '#1f77b4', 'lw': 2.5},
        'CLAM': {'color': '#ff7f0e', 'lw': 2.5},
        'ROAM': {'color': '#2ca02c', 'lw': 2.5}
    }

    for label, (y_true, y_score) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        c_cfg = configs.get(label, {'color': 'black', 'lw': 2.5})
        
        plt.plot(fpr, tpr, color=c_cfg['color'], lw=c_cfg['lw'], 
                 label=f'{label} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=13)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    plt.title('Subtype Classification Comparison', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=14, frameon=True)
    plt.grid(alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    final_comparison = {}

    # 1. Ours
    print("Collecting Ours...")
    final_comparison['Multimodal Fusion (Ours)'] = collect_ours_binary(SEEDS)

    # 2. Mamba & CLAM
    for m in ['mamba', 'clam_sb']:
        print(f"Loading {m.upper()}...")
        y_t, y_s = load_external_binary(m)
        if y_t is not None:
            final_comparison['Mamba' if m == 'mamba' else 'CLAM'] = (y_t, y_s)

    # 3. ROAM
    print("Loading ROAM...")
    y_t_r, y_s_r = load_roam_binary()
    if y_t_r is not None:
        final_comparison['ROAM'] = (y_t_r, y_s_r)

    # 4. 绘图
    if final_comparison:
        out_file = os.path.join(AGGREGATED_OUTPUT, "subtype_comparison_4models.png")
        plot_combined_roc(final_comparison, out_file)
        print(f"✨ 成功生成 4 模型对比图: {out_file}")