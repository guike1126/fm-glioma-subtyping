import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ==========================
# 1. 动态标签与层级解析
# ==========================
def get_hierarchy_info(feature_dir):
    """扫描 H5 目录，自动建立 6 分类到 (亚型, 等级) 的层级映射"""
    unique_labels = set()
    for split in ['train', 'val', 'test']:
        p = os.path.join(feature_dir, split)
        if not os.path.exists(p): continue
        for fn in os.listdir(p):
            if fn.endswith('.h5'):
                with h5py.File(os.path.join(p, fn), 'r') as f:
                    l = f['label'][()]
                    unique_labels.add(l.decode('utf-8') if isinstance(l, bytes) else str(l))
    
    sorted_labels = sorted(list(unique_labels))
    label_to_6class = {n: i for i, n in enumerate(sorted_labels)}
    
    # 建立映射: {6分类索引: (亚型索引, 等级索引)}
    # 亚型: 0: Ast, 1: Oli, 2: GBM
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

# ==========================
# 2. 级联数据加载器
# ==========================
class CascadeDataset(Dataset):
    def __init__(self, feature_dir, split, l2i, hierarchy, filter_sub=None, task='sub'):
        self.data, self.targets, self.info = [], [], []
        split_path = os.path.join(feature_dir, split)
        
        if not os.path.exists(split_path): return

        for fn in os.listdir(split_path):
            if not fn.endswith('.h5'): continue
            with h5py.File(os.path.join(split_path, fn), 'r') as f:
                l_str = f['label'][()].decode('utf-8') if isinstance(f['label'][()], bytes) else str(f['label'][()])
                idx6 = l2i[l_str]
                sub, grd = hierarchy[idx6]
                
                # 针对子任务进行样本过滤
                if filter_sub is not None and sub != filter_sub: continue
                
                self.data.append(f['embedding'][:])
                self.info.append((sub, grd, idx6))
                
                if task == 'sub': self.targets.append(sub)
                elif task == 'grade': self.targets.append(grd)
                else: self.targets.append(idx6)

        if len(self.data) > 0:
            self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
            self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]

# ==========================
# 3. 模型架构
# ==========================
class BranchNet(nn.Module):
    """用于亚型或等级的基础分支网络"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, out_dim)
        )
    def forward(self, x): return self.net(x)

class UnifiedCascadeModel(nn.Module):
    """联合级联模型：基于概率乘积规则整合分支"""
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
            
        return torch.log(probs_6 + 1e-8) # 转为 LogSoftmax 以便使用 NLLLoss

# ==========================
# 4. 评估与训练工具
# ==========================
def evaluate(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    if len(loader.dataset) == 0: return 0.0, 0.0, [], []
    
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            all_p.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_l.extend(y.numpy())
    return accuracy_score(all_l, all_p), f1_score(all_l, all_p, average='weighted'), all_l, all_p

def print_cm(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    header = "      " + "".join([f"{str(l):>12}" for l in labels])
    print(header)
    for i, row in enumerate(cm):
        print(f"{str(labels[i]):<10}" + "".join([f"{count:>12}" for count in row]))
    print("")

def train_stage(model, loaders, device, epochs, lr, patience, name, class_names, save_dir):
    print(f"\n{'='*15} 训练阶段: {name} {'='*15}")
    train_loader, val_loader, test_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 级联层输出的是 Log 概率，用 NLLLoss；分支层直接用 CrossEntropy
    criterion = nn.NLLLoss() if "Unified" in name else nn.CrossEntropyLoss()
    
    best_val_acc = 0
    early_stop_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{name.lower().replace(' ', '_')}.pth")

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        # 验证集评估与早停
        v_acc, v_f1, _, _ = evaluate(model, val_loader, device)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
        
        if (epoch+1) % 5 == 0:
            t_acc, _, _, _ = evaluate(model, train_loader, device)
            print(f"Epoch {epoch+1:3d} | Train ACC: {t_acc:.4f} | Val ACC: {v_acc:.4f} (Best: {best_val_acc:.4f})")

        if early_stop_counter >= patience:
            print(f"--- 触发早停！连续 {patience} 个 Epoch 验证集无提升 ---")
            break

    # 加载最佳模型并在测试集上汇报最终结果
    model.load_state_dict(torch.load(save_path))
    acc, f1, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\n[Final Test Result: {name}] ACC: {acc:.4f}, F1: {f1:.4f}")
    print_cm(y_true, y_pred, class_names)
    return model

# ==========================
# 5. 主执行流程
# ==========================
def run_full_experiment():
    # 配置路径
    FEATURE_DIR = "E:/学习/论文2/战桐整理/代码和结果/代码和结果/vae_best_features_9468"
    SAVE_DIR = "E:/学习/论文2/战桐整理/代码和结果/代码和结果/omics_model_9468"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取标签层级关系
    l2i, hier, names_6 = get_hierarchy_info(FEATURE_DIR)

    # 快捷构建数据加载器的工具函数
    def get_loaders(task, fs=None):
        return (
            DataLoader(CascadeDataset(FEATURE_DIR, 'train', l2i, hier, filter_sub=fs, task=task), batch_size=32, shuffle=True),
            DataLoader(CascadeDataset(FEATURE_DIR, 'val', l2i, hier, filter_sub=fs, task=task), batch_size=32),
            DataLoader(CascadeDataset(FEATURE_DIR, 'test', l2i, hier, filter_sub=fs, task=task), batch_size=32)
        )

    # Stage 1: 亚型分类 (3分类)
    sub_net = train_stage(BranchNet(128, 3).to(DEVICE), get_loaders('sub'), DEVICE, 
                          epochs=100, lr=1e-3, patience=20, name="Subtype", 
                          class_names=["Ast", "Oli", "GBM"], save_dir=SAVE_DIR)

    # Stage 2: Ast 等级分类 (3分类: G2, G3, G4)
    ast_net = train_stage(BranchNet(128, 3).to(DEVICE), get_loaders('grade', 0), DEVICE, 
                          epochs=100, lr=1e-3, patience=20, name="Ast Grade", 
                          class_names=["G2", "G3", "G4"], save_dir=SAVE_DIR)

    # Stage 3: Oli 等级分类 (2分类: G2, G3)
    oli_net = train_stage(BranchNet(128, 2).to(DEVICE), get_loaders('grade', 1), DEVICE, 
                          epochs=100, lr=1e-3, patience=20, name="Oli Grade", 
                          class_names=["G2", "G3"], save_dir=SAVE_DIR)

    # Stage 4: 联合微调 (6分类)
    cascade_model = UnifiedCascadeModel(sub_net, ast_net, oli_net, hier).to(DEVICE)
    train_stage(cascade_model, get_loaders('6class'), DEVICE, 
                epochs=150, lr=2e-4, patience=30, name="Unified Cascade", 
                class_names=names_6, save_dir=SAVE_DIR)

if __name__ == "__main__":
    run_full_experiment()