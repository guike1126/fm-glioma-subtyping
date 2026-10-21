import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 导入您的原始模块
from multimodal_loader import create_multimodal_datasets, get_stage_loader
from train_multimodal_v2 import DeepMultimodalModel

# ================= 1. 配置部分 =================
SEEDS = [6003, 4873, 3157, 1107, 9468]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径配置
BASE_DIR = "/seu_nvme/home/liuhongde/220232427"
CSV_FILE = os.path.join(BASE_DIR, "imputed_data_OS_filled.csv")
FEATURES_DIR = os.path.join(BASE_DIR, "features_h5")
SPLIT_DIR = BASE_DIR
OUTPUT_BASE = "./outputs_multimodel"
ROC_DATA_DIR = "/seu_nvme/home/liuhongde/220232427/fm-glioma-subtyping-main/roc_data"  # CLAM/Mamba 数据存放目录
AGGREGATED_OUTPUT = "./paper_comparison_results"

os.makedirs(AGGREGATED_OUTPUT, exist_ok=True)

# ================= 2. 数据采集逻辑 =================

def get_subtype_mapping(classes, device):
    sub_m = []
    for name in classes:
        n = name.lower()
        s = 0 if 'astro' in n else 1 if 'gbm' in n else 2
        sub_m.append(s)
    return torch.tensor(sub_m).to(device)

def collect_multimodal_preds(seeds):
    """采集多模态模型的预测结果"""
    all_y_true, all_y_score = [], []
    for seed in seeds:
        model_dir = os.path.join(OUTPUT_BASE, f"seed_{seed}")
        pth_path = os.path.join(model_dir, 'best_subtype.pth')
        if not os.path.exists(pth_path): continue
        
        _, _, test_ds, classes, input_dim = create_multimodal_datasets(CSV_FILE, FEATURES_DIR, BASE_DIR, SPLIT_DIR, seed)
        mapping = get_subtype_mapping(classes, DEVICE)
        
        model = DeepMultimodalModel(omics_input_dim=input_dim).to(DEVICE)
        model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        model.eval()
        
        loader = get_stage_loader(test_ds, allowed_subtypes=None, batch_size=1, is_train=False)
        with torch.no_grad():
            for img, om, lbl, _ in loader:
                out = model(img.to(DEVICE), om.to(DEVICE))['logits']['subtype_logits']
                all_y_score.append(torch.softmax(out, dim=1).cpu().numpy().flatten())
                all_y_true.append(mapping[lbl].cpu().numpy().flatten())
    return np.array(all_y_true).flatten(), np.array(all_y_score)

def load_external_model_preds(model_name, k=5):
    """加载 CLAM/Mamba 等外部模型的预测数据"""
    data_dir = os.path.join(ROC_DATA_DIR, model_name)
    all_probs, all_labels = [], []
    for f in range(k):
        p_path = os.path.join(data_dir, f'probs_fold_{f}.npy')
        l_path = os.path.join(data_dir, f'labels_fold_{f}.npy')
        if os.path.exists(p_path) and os.path.exists(l_path):
            all_probs.append(np.load(p_path))
            all_labels.append(np.load(l_path))
    if not all_probs: return None, None
    return np.concatenate(all_labels), np.concatenate(all_probs)

# ================= 3. 核心绘图逻辑 =================

def calculate_macro_roc(y_true, y_score, n_classes=3):
    """计算 Macro-average ROC 数据"""
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr_dict, tpr_dict = {}, {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

def plot_comparison_roc(results_dict, save_path):
    """绘制对比图：正方形、实线、学术风格"""
    plt.figure(figsize=(8, 8))
    
    # 定义颜色和线型
    # 按照你的要求全部使用实线，通过颜色区分
    configs = {
        'Multimodal Fusion (Ours)': {'color': '#d62728', 'lw': 2.5}, # 红色突出自己
        'Mamba': {'color': '#1f77b4', 'lw': 2.5},
        'CLAM': {'color': '#ff7f0e', 'lw': 2.5}
    }

    for model_label, (fpr, tpr, roc_auc) in results_dict.items():
        conf = configs.get(model_label, {'color': 'black', 'lw': 2})
        plt.plot(fpr, tpr, color=conf['color'], lw=conf['lw'], linestyle='-',
                 label=f'{model_label} (AUC = {roc_auc:.3f})')

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

# ================= 4. 主流程 =================

if __name__ == "__main__":
    comparison_results = {}

    # 1. 采集 Ours 结果
    print("Processing Multimodal Fusion (Ours)...")
    y_t_ours, y_s_ours = collect_multimodal_preds(SEEDS)
    comparison_results['Multimodal Fusion (Ours)'] = calculate_macro_roc(y_t_ours, y_s_ours)

    # 2. 采集对比模型结果
    for m_name in ['mamba', 'clam_sb']:
        print(f"Processing {m_name.upper()}...")
        y_t_ext, y_s_ext = load_external_model_preds(m_name)
        if y_t_ext is not None:
            label = 'Mamba' if m_name == 'mamba' else 'CLAM'
            comparison_results[label] = calculate_macro_roc(y_t_ext, y_s_ext)

    # 3. 绘图
    if comparison_results:
        out_file = os.path.join(AGGREGATED_OUTPUT, "comparison_subtype_macro_roc.png")
        plot_comparison_roc(comparison_results, out_file)
        print(f"📊 对比图已成功保存至: {out_file}")