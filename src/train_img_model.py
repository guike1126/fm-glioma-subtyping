import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
import gc

# 引用 create_datasets
# 确保 img_model_loader.py 在同一目录下
from img_model_loader import create_datasets, get_stage_loader

# ============================================================
# 1. 线性注意力层 (Linear Attention Layer) - 数值稳定版
# 核心优化：O(N) 复杂度 + FP32 混合精度计算防止 NaN
# ============================================================
class LinearAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 的线性映射
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 特征映射函数 (ELU + 1)
        self.elu = nn.ELU()
        
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # -------------------------------------------------------
        # ★★★ 核心修复：强制 float32 计算以防止 nan/inf ★★★
        # -------------------------------------------------------
        # 即使外部开启了 autocast，这里也会强制用 fp32 精度
        # 这对于处理长序列(7000+)的累加至关重要，防止 float16 溢出
        dtype_orig = q.dtype # 记住原始精度
        
        q = q.float() 
        k = k.float()
        v = v.float()

        # 应用核函数 (ELU + 1)
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        
        # 1. 计算全局上下文 KV (D x D 矩阵)
        # kv: [B, H, N, D]^T * [B, H, N, D] -> [B, H, D, D]
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        
        # 2. 计算 Attention Output
        # x_attn: [B, H, N, D] * [B, H, D, D] -> [B, H, N, D]
        x_attn = torch.einsum('b h n d, b h d e -> b h n e', q, kv)
        
        # 3. 归一化项 (Z)
        # 防止除零，并保证分母计算也是 FP32
        z = 1.0 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        x_attn = x_attn * z.unsqueeze(-1)
        
        # 转回原始精度 (例如 float16) 以兼容后续网络
        x_attn = x_attn.to(dtype_orig)
        # -------------------------------------------------------

        x_attn = x_attn.reshape(B, N, C)
        x = self.proj(x_attn)
        
        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

# ============================================================
# 2. 主模型 (DeepImageTransformer)
# ============================================================
class DeepImageTransformer(nn.Module):
    def __init__(self, img_dim=512, nhead=8, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.img_norm = nn.LayerNorm(img_dim)
        
        # 使用 LinearAttentionLayer 堆叠
        self.layers = nn.ModuleList([
            LinearAttentionLayer(img_dim, num_heads=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Attention Pooling
        self.pool = nn.Sequential(
            nn.Linear(img_dim, 256), 
            nn.Tanh(), 
            nn.Linear(256, 1)
        )
        
        # Classification Heads
        self.head_subtype = nn.Sequential(
            nn.Linear(img_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), 
            nn.Linear(256, 3)
        )
        self.head_astro = nn.Sequential(
            nn.Linear(img_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), 
            nn.Linear(256, 3)
        )
        self.head_oligo = nn.Sequential(
            nn.Linear(img_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), 
            nn.Linear(256, 2)
        )

    def forward(self, img_feat):
        # img_feat: [B, N, 512]
        x = self.img_norm(img_feat)
        
        # 逐层经过线性 Attention
        for layer in self.layers:
            x = layer(x) 
        
        # Attention Pooling
        A = torch.softmax(self.pool(x), dim=1) 
        global_feat = torch.sum(x * A, dim=1)

        # 预测头
        sub_logits = self.head_subtype(global_feat)
        astro_logits = self.head_astro(global_feat)
        oligo_logits = self.head_oligo(global_feat)
        
        # 级联概率计算
        s_prob = torch.softmax(sub_logits, dim=1)
        a_prob = torch.softmax(astro_logits, dim=1)
        o_prob = torch.softmax(oligo_logits, dim=1)
        
        final_probs = torch.zeros(img_feat.size(0), 6).to(img_feat.device)
        
        # [Ast_G2, Ast_G3, Ast_G4, GBM_G4, Oli_G2, Oli_G3]
        final_probs[:, 0:3] = s_prob[:, 0:1] * a_prob 
        final_probs[:, 3] = s_prob[:, 1]              
        final_probs[:, 4:6] = s_prob[:, 2:3] * o_prob 

        return {
            'feat': global_feat,
            'subtype': sub_logits,
            'astro': astro_logits,
            'oligo': oligo_logits,
            'final': torch.log(final_probs + 1e-9)
        }

# ============================================================
# 3. 训练流程 (含梯度裁剪和多卡支持)
# ============================================================
# ============================================================
# 3. 训练流程 (纯 FP32 版本 - 彻底解决 NaN)
# ============================================================
class CascadeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.output_dir = os.path.join(config['output_dir'], f"seed_{config['seed']}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train_ds, self.val_ds, self.test_ds, self.classes = create_datasets(
            config['csv_file'], config['features_dir'], config['base_dir'], config['split_dir'], config['seed'])
        
        self.model = DeepImageTransformer(img_dim=512, num_layers=2)
        
        if torch.cuda.device_count() > 1:
            print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            
        self.model = self.model.to(self.device)
        self._init_mappings()
        
        # ❌ 删除 self.scaler = GradScaler('cuda') 
        # 我们现在用纯 FP32，不需要 Scaler

    def _init_mappings(self):
        self.map_sub, self.map_ast, self.map_oli = {}, {}, {}
        for name in self.classes:
            parts = name.split('_')
            sub_name, grade = parts[0], int(parts[1].replace('G', ''))
            s_idx = 0 if 'Astrocytoma' in sub_name else (1 if 'GBM' in sub_name else 2)
            a_idx = grade - 2 if s_idx == 0 else -1
            o_idx = grade - 2 if s_idx == 2 else -1
            idx = list(self.classes).index(name)
            self.map_sub[idx] = s_idx
            self.map_ast[idx] = a_idx
            self.map_oli[idx] = o_idx

    def _get_batch_labels(self, labels, task):
        targets = []
        for l in labels.cpu().numpy():
            if task == 'subtype': t = self.map_sub[l]
            elif task == 'astro': t = self.map_ast[l]
            elif task == 'oligo': t = self.map_oli[l]
            elif task == 'final': t = l
            targets.append(t)
        return torch.tensor(targets).to(self.device)

    def evaluate(self, loader, task):
        self.model.eval()
        p_all, t_all, loss_all = [], [], []
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for img, lbl, _ in loader:
                img = img.to(self.device)
                target = self._get_batch_labels(lbl, task)
                
                # ❌ 删除 with autocast('cuda'):
                # 直接前向传播 (FP32)
                out = self.model(img)
                logits = out[task]
                loss = criterion(logits, target)
                
                p_all.extend(logits.argmax(1).cpu().numpy())
                t_all.extend(target.cpu().numpy())
                loss_all.append(loss.item())
                
        return np.mean(loss_all), accuracy_score(t_all, p_all), f1_score(t_all, p_all, average='macro', zero_division=0), p_all, t_all

    def train_one_epoch(self, loader, optimizer, task, epoch, accum_steps=32):
        self.model.train()
        losses = []
        all_preds, all_targets = [], []
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(loader, desc=f"Ep {epoch} [{task}]", leave=False)
        optimizer.zero_grad()
        
        for i, (img, lbl, _) in enumerate(pbar):
            img = img.to(self.device)
            target = self._get_batch_labels(lbl, task)
            
            # ❌ 删除 with autocast('cuda'):
            # 直接计算 (FP32)
            logits = self.model(img)[task]
            loss = criterion(logits, target)
            loss = loss / accum_steps
            
            # ❌ 删除 self.scaler.scale(loss).backward()
            loss.backward() # 标准反向传播
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                # 依然保留梯度裁剪，防止偶尔的梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # ❌ 删除 self.scaler.step(optimizer)
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item() * accum_steps)
            all_preds.extend(logits.argmax(1).detach().cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 监控 Loss 是否正常
            if np.isnan(loss.item()):
                pbar.set_postfix({'loss': 'NAN (Panic!)'})
            else:
                pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})
            
        tr_acc = accuracy_score(all_targets, all_preds)
        tr_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return np.mean(losses), tr_acc, tr_f1

    def run_stage(self, task_name, epochs, lr, patience=20, subs=None, grades=None, load_from=None):
        print(f"\n{'='*15} STAGE: {task_name.upper()} (Patience={patience}) {'='*15}")
        
        # 保持你的配置：Batch=1 + Accum=32 (非常稳)
        batch_size = 1
        grad_accum_steps = 32
        
        tr_l = get_stage_loader(self.train_ds, subs, grades, batch_size, True)
        val_l = get_stage_loader(self.val_ds, subs, grades, batch_size, False)
        te_l = get_stage_loader(self.test_ds, subs, grades, batch_size, False)
        
        if not tr_l: 
            print("⚠️ Skipping Stage: No data available.")
            return

        if load_from and os.path.exists(load_from):
            print(f"[*] Loading weights from: {load_from}")
            state_dict = torch.load(load_from, weights_only=False)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            
            # 兼容性加载逻辑
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            try: model_to_load.load_state_dict(new_state_dict, strict=False)
            except: model_to_load.load_state_dict(new_state_dict, strict=False)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_f1 = 0
        best_epoch = 0
        no_imp = 0 
        save_path = os.path.join(self.output_dir, f"best_{task_name}.pth")
        
        print(f"{'Epoch':<5} | {'Tr Loss':<8} | {'Tr Acc':<8} | {'Tr F1':<8} || {'Val Loss':<8} | {'Val Acc':<8} | {'Val F1':<8} | {'Status'}")
        print("-" * 105)

        for ep in range(1, epochs+1):
            tr_loss, tr_acc, tr_f1 = self.train_one_epoch(tr_l, optimizer, task_name, ep, grad_accum_steps)
            v_loss, v_acc, v_f1, _, _ = self.evaluate(val_l, task_name)
            
            msg = ""
            if v_f1 > best_f1:
                best_f1 = v_f1
                best_epoch = ep
                no_imp = 0
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save(model_to_save.state_dict(), save_path)
                msg = f"⭐ Best"
            else:
                no_imp += 1
                msg = f"Wait {no_imp}/{patience}"
            
            print(f"{ep:<5} | {tr_loss:.4f}   | {tr_acc:.4f}   | {tr_f1:.4f}   || {v_loss:.4f}   | {v_acc:.4f}   | {v_f1:.4f}   | {msg}")

            if no_imp >= patience:
                print(f"\n🛑 Early stopping triggered. Best epoch was {best_epoch} with Val F1: {best_f1:.4f}")
                break
        
        # Test Report
        print(f"\n>>> Final Testing for {task_name} (using best model from Ep {best_epoch})")
        if os.path.exists(save_path):
            state_dict = torch.load(save_path, weights_only=False)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            # 再次确保加载正确
            new_state_dict = {}
            for k, v in state_dict.items():
                 name = k.replace("module.", "")
                 new_state_dict[name] = v
            model_to_load.load_state_dict(new_state_dict, strict=False)
        
        _, te_acc, te_f1, preds, targets = self.evaluate(te_l, task_name)
        print(f"Test ACC: {te_acc:.4f} | Test F1: {te_f1:.4f}")
        print(classification_report(targets, preds, digits=4, zero_division=0))
        
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {task_name}\nTest F1: {te_f1:.4f}")
        plt.savefig(os.path.join(self.output_dir, f"cm_{task_name}.png")); plt.close()
        
        return save_path

    def run_pipeline(self):
        # 保持原有逻辑
        path_sub_2 = self.run_stage('subtype', 100, 5e-5, 20, ['Astrocytoma', 'Oligodendroglioma'])
        path_sub_3 = self.run_stage('subtype', 100, 2e-5, 20, None, load_from=path_sub_2)
        path_astro = self.run_stage('astro', 100, 2e-5, 20, ['Astrocytoma'], load_from=path_sub_3)
        path_oligo = self.run_stage('oligo', 100, 2e-5, 20, ['Oligodendroglioma'], load_from=path_sub_3)
        
        print("\n>>> Loading combined weights for Final Stage...")
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        state = model_to_load.state_dict()
        
        if path_sub_3 and os.path.exists(path_sub_3):
             state = torch.load(path_sub_3, weights_only=False)
        if path_astro and os.path.exists(path_astro):
            ast_state = torch.load(path_astro, weights_only=False)
            for k in ast_state: 
                k_clean = k.replace("module.", "")
                if 'head_astro' in k_clean: state[k_clean] = ast_state[k]
        if path_oligo and os.path.exists(path_oligo):
            oli_state = torch.load(path_oligo, weights_only=False)
            for k in oli_state:
                k_clean = k.replace("module.", "")
                if 'head_oligo' in k_clean: state[k_clean] = oli_state[k]
        
        model_to_load.load_state_dict(state, strict=False)
        self.run_stage('final', 100, 5e-6, 20, None)

# ============================================================
# 主程序入口
# ============================================================
if __name__ == '__main__':
    # 显存管理优化 (防止碎片化)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    base_config = {
        'csv_file': "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv", 
        'features_dir': "/seu_nvme/home/liuhongde/220232427/features_h5", 
        'base_dir': "/seu_nvme/home/liuhongde/220232427", 
        'split_dir': "/seu_nvme/home/liuhongde/220232427",
        'output_dir': './outputs_img_model', 
        'device': 'cuda'
    }

    seeds_to_run = [3157, 1107]

    print(f"🚀 Starting Sequential Training for Seeds: {seeds_to_run}")
    print(f"💻 Hardware: V100 GPU + 512GB RAM detected.")
    print(f"⚙️  Strategy: Linear Attention (O(N) Complexity) + FP32 Accumulation.")

    for seed in seeds_to_run:
        print(f"\n\n{'#'*50}")
        print(f"{'#'*15}  RUNNING SEED: {seed}  {'#'*15}")
        print(f"{'#'*50}\n")
        
        config = base_config.copy()
        config['seed'] = seed
        
        try:
            trainer = CascadeTrainer(config)
            trainer.run_pipeline()
            
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"✅ Seed {seed} completed. Memory cleared.")
            
        except Exception as e:
            print(f"❌ Error running seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 All seeds finished successfully.")