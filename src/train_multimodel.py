import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from torch.amp import autocast, GradScaler 
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
import gc

# 保持引用，不做修改
from multimodal_loader import create_multimodal_datasets, get_stage_loader

# ============================================================
# 1. 级联模型架构 (保留双向注意力，引入抑制与增强逻辑)
# ============================================================
class DualCrossFusionLayer(nn.Module):
    """双向交叉注意力交互核心"""
    def __init__(self, dim=512, nhead=8, dropout=0.1):
        super().__init__()
        self.attn_o2i = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm_o = nn.LayerNorm(dim)
        self.ffn_o = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Dropout(dropout), nn.Linear(2048, dim))
        self.norm_o_ffn = nn.LayerNorm(dim)
        
        self.attn_i2o = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm_i = nn.LayerNorm(dim)
        self.ffn_i = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Dropout(dropout), nn.Linear(2048, dim))
        self.norm_i_ffn = nn.LayerNorm(dim)

    def forward(self, img_feat, omics_feat):
        # 组学作为 Query 询问图像
        o_res = omics_feat
        o_out, _ = self.attn_o2i(query=omics_feat, key=img_feat, value=img_feat)
        omics_feat = self.norm_o(o_res + o_out)
        omics_feat = self.norm_o_ffn(omics_feat + self.ffn_o(omics_feat))
        
        # 图像作为 Query 询问组学
        i_res = img_feat
        i_out, _ = self.attn_i2o(query=img_feat, key=o_res, value=o_res) 
        img_feat = self.norm_i(i_res + i_out)
        img_feat = self.norm_i_ffn(img_feat + self.ffn_i(img_feat))
        return img_feat, omics_feat

class CascadeTaskHeads(nn.Module):
    """分类头逻辑"""
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.subtype = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), 
            nn.Dropout(dropout), nn.Linear(256, 3)
        )
        self.grade_bottleneck = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
        )
        self.astro_grade = nn.Linear(256, 3)
        self.oligo_grade = nn.Linear(256, 2)

    def forward(self, x):
        sub_logits = self.subtype(x)
        g_feat = self.grade_bottleneck(x)
        a_g_logits = self.astro_grade(g_feat)
        o_g_logits = self.oligo_grade(g_feat)

        s_prob = torch.softmax(sub_logits, dim=1)
        a_g_prob = torch.softmax(a_g_logits, dim=1)
        o_g_prob = torch.softmax(o_g_logits, dim=1)

        final_probs = torch.zeros(x.size(0), 6).to(x.device)
        final_probs[:, 0:3] = s_prob[:, 0:1] * a_g_prob     
        final_probs[:, 3] = s_prob[:, 1]                    
        final_probs[:, 4:6] = s_prob[:, 2:3] * o_g_prob     

        return {
            'subtype_logits': sub_logits, 
            'astro_grade_logits': a_g_logits, 
            'oligo_grade_logits': o_g_logits, 
            'final_logits': torch.log(final_probs + 1e-9)
        }

class DeepMultimodalModel(nn.Module):
    def __init__(self, omics_input_dim, img_dim=512, dropout=0.3, num_fusion_layers=2):
        super().__init__()
        self.omics_proj = nn.Sequential(nn.Linear(omics_input_dim, img_dim), nn.LayerNorm(img_dim), nn.GELU())
        self.img_norm = nn.LayerNorm(img_dim)
        self.fusion_layers = nn.ModuleList([DualCrossFusionLayer(dim=img_dim, dropout=dropout) for _ in range(num_fusion_layers)])
        
        # 引入注意力池化
        self.pool = nn.Sequential(nn.Linear(img_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.heads = CascadeTaskHeads(input_dim=img_dim * 2, dropout=dropout)

    def forward(self, img_feat, omics_feat):
        # 1. Modality Dropout: 训练期间随机关闭 30% 组学通路
        if self.training and torch.rand(1) < 0.3:
            omics_feat_eff = torch.zeros_like(omics_feat)
        else:
            # 2. 组学噪声注入: 降低组学确信度，防止过拟合
            omics_feat_eff = omics_feat + torch.randn_like(omics_feat) * 0.01 if self.training else omics_feat

        o_emb = self.omics_proj(omics_feat_eff).unsqueeze(1)
        i_emb = self.img_norm(img_feat)
        
        for layer in self.fusion_layers:
            i_emb, o_emb = layer(i_emb, o_emb)
            
        # 3. 锐化池化权重 (T=0.5): 强化图像视觉定位
        A = torch.softmax(self.pool(i_emb) / 0.5, dim=1) 
        i_global = torch.sum(i_emb * A, dim=1)
        
        final_feat = torch.cat([i_global, o_emb.squeeze(1)], dim=1)
        return {'logits': self.heads(final_feat)}

# ============================================================
# 2. Loss 体系
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha, label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

class HierarchicalJointLoss(nn.Module):
    def __init__(self, mappings, device, weights):
        super().__init__()
        self.mappings, self.device = mappings, device
        self.l_sub = FocalLoss(alpha=weights['subtype'], gamma=2.0, label_smoothing=0.05)
        self.l_astro = FocalLoss(alpha=weights['astro'], gamma=3.0, label_smoothing=0.1) 
        self.l_oligo = FocalLoss(alpha=weights['oligo'], gamma=3.0, label_smoothing=0.1)
        self.l_final = FocalLoss(alpha=weights['final'], gamma=2.0, label_smoothing=0.05)

    def forward(self, logits, targets, stage='final'):
        sub_t = self.mappings['subtype'][targets]
        if stage == 'subtype': return self.l_sub(logits['subtype_logits'], sub_t)
        if stage == 'astro_grade':
            astro_t = self.mappings['astro'][targets]
            mask = astro_t != -1
            return self.l_astro(logits['astro_grade_logits'][mask], astro_t[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
        if stage == 'oligo_grade':
            oligo_t = self.mappings['oligo'][targets]
            mask = oligo_t != -1
            return self.l_oligo(logits['oligo_grade_logits'][mask], oligo_t[mask]) if mask.sum() > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)
        if stage == 'final':
            return self.l_final(logits['final_logits'], targets) + 0.5 * self.l_sub(logits['subtype_logits'], sub_t)

# ============================================================
# 3. 训练器
# ============================================================
class CascadeTrainer:
    def __init__(self, config):
        self.config, self.device = config, config['device']
        self.train_ds, self.val_ds, self.test_ds, self.classes, input_omics_dim = create_multimodal_datasets(
            config['csv_file'], config['features_dir'], config['omics_base'], config['split_dir'], config['seed'])
        
        sub_m, astro_m, oligo_m = [], [], []
        for name in self.classes:
            n = name.lower()
            s = 0 if 'astro' in n else 1 if 'gbm' in n else 2
            sub_m.append(s)
            g = int(re.search(r'g(\d)', n).group(1)) if re.search(r'g(\d)', n) else -1
            astro_m.append((0 if g==2 else 1 if g==3 else 2) if s==0 else -1)
            oligo_m.append((0 if g==2 else 1) if s==2 else -1)
            
        self.mappings = {k: torch.tensor(v).to(self.device) for k,v in {'subtype': sub_m, 'astro': astro_m, 'oligo': oligo_m}.items()}
        
        all_l = torch.tensor([s['label'] for s in self.train_ds.valid_samples], dtype=torch.long)
        def get_w(labels_tensor, n_classes):
            c = Counter(labels_tensor.numpy())
            w = [np.sqrt(len(labels_tensor)/(n_classes * c.get(i, 1e-6))) for i in range(n_classes)]
            return torch.tensor(w, dtype=torch.float).to(self.device)
            
        weights = {k: get_w(self.mappings[k][all_l].cpu() if k!='final' else all_l, n) for k, n in zip(['subtype', 'astro', 'oligo', 'final'], [3, 3, 2, 6])}

        self.model = DeepMultimodalModel(omics_input_dim=input_omics_dim).to(self.device)
        self.criterion = HierarchicalJointLoss(self.mappings, self.device, weights)
        self.scaler = GradScaler('cuda')
        self.output_dir = os.path.join(config['output_dir'], f"seed_{config['seed']}")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, loader, stage):
        self.model.eval()
        p, t = [], []
        with torch.no_grad(), autocast('cuda'):
            for img, om, lbl, _ in loader:
                img, om, lbl = img.to(self.device), om.to(self.device), lbl.to(self.device)
                out = self.model(img, om)['logits']
                if stage == 'subtype': pred, target = out['subtype_logits'].argmax(1), self.mappings['subtype'][lbl]
                elif stage == 'astro_grade':
                    target = self.mappings['astro'][lbl]
                    mask = target != -1
                    if mask.sum() == 0: continue
                    pred, target = out['astro_grade_logits'].argmax(1)[mask], target[mask]
                elif stage == 'oligo_grade':
                    target = self.mappings['oligo'][lbl]
                    mask = target != -1
                    if mask.sum() == 0: continue
                    pred, target = out['oligo_grade_logits'].argmax(1)[mask], target[mask]
                else: pred, target = out['final_logits'].argmax(1), lbl
                p.extend(pred.cpu().numpy()); t.extend(target.cpu().numpy())
        return p, t

    def test_and_report(self, stage):
        """测试集详细汇报"""
        target_subs = ['Astrocytoma'] if stage == 'astro_grade' else ['Oligodendroglioma'] if stage == 'oligo_grade' else None
        loader = get_stage_loader(self.test_ds, target_subs, batch_size=1, is_train=False)
        if not loader: return
        
        p, t = self.evaluate(loader, stage)
        acc, f1 = accuracy_score(t, p), f1_score(t, p, average='macro', zero_division=0)
        
        print(f"\n[Final Test Result: {stage.upper()}] ACC: {acc:.4f}, F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(t, p))
        print(classification_report(t, p, digits=4, zero_division=0))

    def run_stage(self, stage, epochs, lr, patience, targets=None):
        print(f"\n>>> Stage: {stage.upper()}")
        
        # 4. 非对称学习率策略
        omics_params = [p for n, p in self.model.named_parameters() if "omics_proj" in n]
        other_params = [p for n, p in self.model.named_parameters() if "omics_proj" not in n]
        
        opt = torch.optim.AdamW([{'params': other_params, 'lr': lr}, {'params': omics_params, 'lr': lr * 0.1}], weight_decay=1e-2)
        sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
        train_l = get_stage_loader(self.train_ds, targets, batch_size=self.config['batch_size'], is_train=True)
        val_l = get_stage_loader(self.val_ds, targets, batch_size=self.config['batch_size'], is_train=False)
        
        if not train_l: return

        best_val_f1, no_imp, best_path = 0, 0, os.path.join(self.output_dir, f"best_{stage}.pth")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            pbar = tqdm(train_l, desc=f"Ep {epoch+1}", leave=False)
            for i, (img, om, lbl, _) in enumerate(pbar):
                img, om, lbl = img.to(self.device), om.to(self.device), lbl.to(self.device)
                with autocast('cuda'):
                    out = self.model(img, om)['logits']
                    loss = self.criterion(out, lbl, stage) / self.config['accumulation_steps']
                self.scaler.scale(loss).backward()
                if (i + 1) % self.config['accumulation_steps'] == 0:
                    self.scaler.step(opt); self.scaler.update(); opt.zero_grad()
                total_loss += loss.item() * self.config['accumulation_steps']
            sch.step()
            
            # Epoch 指标显示
            tp, tt = self.evaluate(train_l, stage)
            vp, vt = self.evaluate(val_l, stage)
            t_acc, t_f1 = accuracy_score(tt, tp), f1_score(tt, tp, average='macro', zero_division=0)
            v_acc, v_f1 = accuracy_score(vt, vp), f1_score(vt, vp, average='macro', zero_division=0)
            
            # ================= 修改后的标注逻辑 =================
            save_msg = ""
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                no_imp = 0
                torch.save(self.model.state_dict(), best_path)
                save_msg = f"  --> [BEST] Model Saved! (Best Val F1: {best_val_f1:.4f})"
            else:
                no_imp += 1

            # 打印当前 Epoch 结果，如果保存了会带上标注
            print(f"Ep {epoch+1:02d} | Loss: {total_loss/len(train_l):.4f} | "
                  f"Train ACC/F1: {t_acc:.4f}/{t_f1:.4f} | Val ACC/F1: {v_acc:.4f}/{v_f1:.4f}{save_msg}")
            
            if no_imp >= patience: 
                print(f"Early stopping at epoch {epoch+1}")
                break
            # ===================================================
        
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path))
            self.test_and_report(stage)

    def run(self):
        self.run_stage('subtype', 100, 3e-5, 20, ['Astrocytoma', 'Oligodendroglioma']) 
        self.run_stage('subtype', 100, 1e-5, 20)
        self.run_stage('astro_grade', 100, 1e-5, 25, ['Astrocytoma'])
        self.run_stage('oligo_grade', 100, 1e-5, 25, ['Oligodendroglioma'])
        self.run_stage('final', 100, 5e-6, 30)

if __name__ == '__main__':
    # seeds = [4873, 5486, 6003, 6782, 9994]
    # seeds = [8354, 4842, 9468, 5802, 1754]
    seeds = [1431, 2516, 9458, 7242, 1107, 1542, 8101, 3157, 5084, 9189, 1947, 4619]
    base_config = {
        'csv_file': "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv", 
        'features_dir': "/seu_nvme/home/liuhongde/220232427/features_h5", 
        'omics_base': "/seu_nvme/home/liuhongde/220232427", 
        'split_dir': "/seu_nvme/home/liuhongde/220232427",
        'output_dir': './outputs_multimodel', 
        'device': 'cuda', 'batch_size': 1, 'accumulation_steps': 32
    }
    for seed in seeds:
        print(f"\n{'='*60}\n>>> STARTING SEED: {seed}\n{'='*60}")
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        try:
            CascadeTrainer({**base_config, 'seed': seed}).run()
        except Exception as e: print(f"Error Seed {seed}: {e}")