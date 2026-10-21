import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ================= 配置部分 =================
# 在这里修改你想评估的种子
SEEDS = [42,2026]

# 根目录配置
ROOT_DIR = "/seu_nvme/home/liuhongde/220232427"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ================= 1. 模型与数据类定义 (集成自 omics_model.py) =================

def get_hierarchy_info(feature_dir):
    """扫描 H5 目录，自动建立 6 分类到 (亚型, 等级) 的层级映射"""
    unique_labels = set()
    # 扫描测试集即可，因为标签空间应该是一致的
    for split in ['test', 'train', 'val']:
        p = os.path.join(feature_dir, split)
        if not os.path.exists(p): continue
        for fn in os.listdir(p):
            if fn.endswith('.h5'):
                with h5py.File(os.path.join(p, fn), 'r') as f:
                    l = f['label'][()]
                    unique_labels.add(l.decode('utf-8') if isinstance(l, bytes) else str(l))
    
    sorted_labels = sorted(list(unique_labels))
    label_to_6class = {n: i for i, n in enumerate(sorted_labels)}
    
    hierarchy = {}
    for name in sorted_labels:
        n_low = name.lower()
        if 'ast' in n_low: 
            sub, grade = 0, (0 if '2' in n_low else (1 if '3' in n_low else 2)) 
        elif 'oli' in n_low:
            sub, grade = 1, (0 if '2' in n_low else 1)
        else: # GBM
            sub, grade = 2, 0 
        hierarchy[label_to_6class[name]] = (sub, grade)
    
    return label_to_6class, hierarchy, sorted_labels

class CascadeDataset(Dataset):
    def __init__(self, feature_dir, split, l2i, hierarchy, filter_sub=None, task='sub'):
        self.data, self.targets = [], []
        split_path = os.path.join(feature_dir, split)
        
        if not os.path.exists(split_path):
            return 

        for fn in os.listdir(split_path):
            if not fn.endswith('.h5'): continue
            with h5py.File(os.path.join(split_path, fn), 'r') as f:
                l_str = f['label'][()].decode('utf-8') if isinstance(f['label'][()], bytes) else str(f['label'][()])
                idx6 = l2i[l_str]
                sub, grd = hierarchy[idx6]
                
                if filter_sub is not None and sub != filter_sub: continue
                
                self.data.append(f['embedding'][:])
                if task == 'sub': self.targets.append(sub)
                elif task == 'grade': self.targets.append(grd)
                else: self.targets.append(idx6) # 6class

        if len(self.data) > 0:
            self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
            self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

class BranchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, out_dim)
        )
    def forward(self, x): return self.net(x)

class UnifiedCascadeModel(nn.Module):
    def __init__(self, sub_net, ast_net, oli_net, hierarchy):
        super().__init__()
        self.sub_net, self.ast_net, self.oli_net = sub_net, ast_net, oli_net
        self.hierarchy = hierarchy

    def forward(self, x):
        sub_p = torch.softmax(self.sub_net(x), dim=1)
        ast_p = torch.softmax(self.ast_net(x), dim=1)
        oli_p = torch.softmax(self.oli_net(x), dim=1)

        batch_size = x.size(0)
        probs_6 = torch.zeros(batch_size, 6).to(x.device)
        for i in range(6):
            s_idx, g_idx = self.hierarchy[i]
            if s_idx == 0: probs_6[:, i] = sub_p[:, 0] * ast_p[:, g_idx]
            elif s_idx == 1: probs_6[:, i] = sub_p[:, 1] * oli_p[:, g_idx]
            else: probs_6[:, i] = sub_p[:, 2] # GBM
        return torch.log(probs_6 + 1e-8) # 返回 log_probs

# ================= 2. 绘图工具函数 =================

def draw_roc_on_ax(ax, y_true, y_score, n_classes, class_names, title):
    """绘制 ROC 曲线的核心函数"""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    ax.plot(fpr["micro"], tpr["micro"],
             label='Micro (area={0:0.3f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)
    ax.plot(fpr["macro"], tpr["macro"],
             label='Macro (area={0:0.3f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        label_name = class_names[i] if i < len(class_names) else f"C{i}"
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='{0} ({1:0.3f})'.format(label_name, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize='x-small', framealpha=0.6)
    ax.grid(alpha=0.3)

# ================= 3. 核心评估流程 =================

def run_omics_evaluation(seed):
    print(f"\n{'='*50}")
    print(f"🧬 Processing Omics Model for SEED: {seed}")
    print(f"{'='*50}")

    # 路径构造
    feat_dir = os.path.join(ROOT_DIR, f"vae_best_features_{seed}")
    model_dir = os.path.join(ROOT_DIR, f"omics_model_{seed}")

    if not os.path.exists(feat_dir) or not os.path.exists(model_dir):
        print(f"❌ Directories missing for seed {seed}. Skipping.")
        return

    # 1. 获取层级信息
    try:
        l2i, hier, names_6 = get_hierarchy_info(feat_dir)
    except Exception as e:
        print(f"❌ Error reading hierarchy info: {e}")
        return

    # 2. 定义任务配置
    tasks = [
        # (任务名, 权重文件, 类别数, 类名, 数据集task, 过滤条件, 模型类型)
        ("Subtype", "best_subtype.pth", 3, ["Ast", "Oli", "GBM"], 'sub', None, 'branch'),
        ("Ast Grade", "best_ast_grade.pth", 3, ["G2", "G3", "G4"], 'grade', 0, 'branch'),
        ("Oli Grade", "best_oli_grade.pth", 2, ["G2", "G3"], 'grade', 1, 'branch'),
        ("Unified Cascade", "best_unified_cascade.pth", 6, names_6, '6class', None, 'cascade')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    plotted_count = 0

    for i, (name, pth_name, n_cls, cls_names, ds_task, filter_sub, model_type) in enumerate(tasks):
        pth_path = os.path.join(model_dir, pth_name)
        if not os.path.exists(pth_path):
            print(f"  ⚠️ Weight missing: {pth_name}")
            continue

        print(f"  Running: {name} ...")
        
        # 准备数据
        ds = CascadeDataset(feat_dir, 'test', l2i, hier, filter_sub=filter_sub, task=ds_task)
        if len(ds) == 0:
            print(f"    ⚠️ No samples found for {name}.")
            continue
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        # 准备模型
        if model_type == 'branch':
            model = BranchNet(128, n_cls).to(DEVICE)
            model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
        else:
            # 级联模型需要先实例化子网络，再加载整体权重
            sub_net = BranchNet(128, 3)
            ast_net = BranchNet(128, 3)
            oli_net = BranchNet(128, 2)
            model = UnifiedCascadeModel(sub_net, ast_net, oli_net, hier).to(DEVICE)
            model.load_state_dict(torch.load(pth_path, map_location=DEVICE))

        # 推理
        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                out = model(x)
                
                # 处理输出
                if model_type == 'cascade':
                    probs = torch.exp(out) # 级联模型输出的是 log_probs
                else:
                    probs = torch.softmax(out, dim=1) # 分支模型输出的是 logits
                
                y_true.extend(y.numpy())
                y_score.extend(probs.cpu().numpy())

        # 画图
        if len(y_true) > 0:
            draw_roc_on_ax(
                axes[i], 
                np.array(y_true), 
                np.array(y_score), 
                n_cls, 
                cls_names, 
                f"Omics ROC: {name}"
            )
            plotted_count += 1

    # 保存结果
    if plotted_count > 0:
        plt.suptitle(f"Omics Model Performance (Seed {seed})", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(model_dir, f"omics_auc_grid.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Plot saved to: {save_path}")
    else:
        plt.close()
        print("⚠️ No plots generated.")

if __name__ == "__main__":
    print(f"Started Batch Omics Evaluation for seeds: {SEEDS}")
    for seed in SEEDS:
        run_omics_evaluation(seed)
    print("\nAll Done.")