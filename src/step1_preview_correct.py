import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import cv2
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import openslide
except ImportError:
    print("❌ 错误: 未安装 openslide-python。请先安装。")
    exit(1)

from multimodal_loader import create_multimodal_datasets, get_stage_loader

# ============================================================
# 1. 完整的模型架构定义
# ============================================================
class DualCrossFusionLayer(nn.Module):
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
        o_res = omics_feat
        o_out, _ = self.attn_o2i(query=omics_feat, key=img_feat, value=img_feat)
        omics_feat = self.norm_o(o_res + o_out)
        omics_feat = self.norm_o_ffn(omics_feat + self.ffn_o(omics_feat))
        i_res = img_feat
        i_out, _ = self.attn_i2o(query=img_feat, key=o_res, value=o_res) 
        img_feat = self.norm_i(i_res + i_out)
        img_feat = self.norm_i_ffn(img_feat + self.ffn_i(img_feat))
        return img_feat, omics_feat

class CascadeTaskHeads(nn.Module):
    def __init__(self, input_dim, dropout=0.5):
        super().__init__()
        self.subtype = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 3))
        self.grade_bottleneck = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout))
        self.astro_grade = nn.Linear(256, 3)
        self.oligo_grade = nn.Linear(256, 2)

    def forward(self, x):
        sub_logits = self.subtype(x)
        g_feat = self.grade_bottleneck(x)
        final_probs = torch.zeros(x.size(0), 6).to(x.device)
        s_p = torch.softmax(sub_logits, dim=1)
        a_p = torch.softmax(self.astro_grade(g_feat), dim=1)
        o_p = torch.softmax(self.oligo_grade(g_feat), dim=1)
        final_probs[:, 0:3] = s_p[:, 0:1] * a_p
        final_probs[:, 3] = s_p[:, 1]
        final_probs[:, 4:6] = s_p[:, 2:3] * o_p
        return {'final_logits': torch.log(final_probs + 1e-9)}

class DeepMultimodalModelForVis(nn.Module):
    def __init__(self, omics_input_dim, img_dim=512, num_fusion_layers=2):
        super().__init__()
        self.omics_proj = nn.Sequential(nn.Linear(omics_input_dim, img_dim), nn.LayerNorm(img_dim), nn.GELU())
        self.img_norm = nn.LayerNorm(img_dim)
        self.fusion_layers = nn.ModuleList([DualCrossFusionLayer(dim=img_dim) for _ in range(num_fusion_layers)])
        self.pool = nn.Sequential(nn.Linear(img_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.heads = CascadeTaskHeads(input_dim=img_dim * 2)

    def forward(self, img_feat, omics_feat):
        o_emb = self.omics_proj(omics_feat).unsqueeze(1)
        i_emb = self.img_norm(img_feat)
        for layer in self.fusion_layers:
            i_emb, o_emb = layer(i_emb, o_emb)
            
        attn_logits = self.pool(i_emb) / 0.5  # 温度缩放，与训练对齐
        A = torch.softmax(attn_logits, dim=1)
        
        i_global = torch.sum(i_emb * A, dim=1)
        final_feat = torch.cat([i_global, o_emb.squeeze(1)], dim=1)
        return self.heads(final_feat), A.squeeze(-1)

# ============================================================
# 2. 绘图配色与预览函数
# ============================================================
ATTENTION_COLORS = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
cmap_custom = LinearSegmentedColormap.from_list('attn', ATTENTION_COLORS, N=256)

def save_preview(svs_path, attn, coords, class_name, save_dir, ps=256):
    sid = os.path.basename(svs_path).replace('.svs', '').replace('.SVS', '')
    slide = openslide.OpenSlide(svs_path)
    thumb = slide.read_region((0, 0), slide.level_count-1, slide.level_dimensions[-1]).convert('RGB')
    orig_w, orig_h = slide.dimensions
    slide.close()
    
    thumb_np = np.array(thumb)
    ranks = np.argsort(np.argsort(attn))
    attn_norm = ranks / (len(ranks) - 1)
    
    canvas_h = 1024
    canvas_w = int(1024 * (orig_w / orig_h))
    h_base = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    ps_scale = int(np.ceil(ps * (canvas_h / orig_h)))
    
    for i, (px, py) in enumerate(coords):
        sx, sy = int(px * (canvas_w / orig_w)), int(py * (canvas_h / orig_h))
        if 0 <= sy < canvas_h and 0 <= sx < canvas_w:
            h_base[sy : sy+max(1, ps_scale), sx : sx+max(1, ps_scale)] = attn_norm[i]
            
    h_resized = cv2.resize(h_base, (thumb_np.shape[1], thumb_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    h_m = np.ma.masked_where(h_resized < 0, h_resized)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(thumb_np); axes[0].set_title(f"Original: {sid}"); axes[0].axis('off')
    axes[1].imshow(thumb_np); axes[1].imshow(h_m, cmap=cmap_custom, alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"Correct: {class_name}"); axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{sid}.jpg"), dpi=120)
    plt.close()

# ============================================================
# 3. 主运行逻辑
# ============================================================
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    _, _, test_ds, classes, input_dim = create_multimodal_datasets(args.csv_file, args.img_features_dir, args.omics_base_dir, args.split_dir, args.seed)
    test_loader = get_stage_loader(test_ds, None, None, batch_size=1, is_train=False)
    
    model = DeepMultimodalModelForVis(omics_input_dim=input_dim).to(device)
    model_path = os.path.join(args.model_dir, f"seed_{args.seed}", "best_final.pth")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    for c in classes: os.makedirs(os.path.join(args.output_dir, c), exist_ok=True)

    # 预设的三大 SVS 图片仓库，并去重
    search_dirs = list(set([
        args.svs_dir,
        "/seu_nvme/home/liuhongde/220232427/fm-glioma-subtyping-main/wsi/tcga_20x",
        "/seu_nvme/home/liuhongde/220232427/fm-glioma-subtyping-main/wsi/tcga_40x",
        "/seu_nvme/home/liuhongde/220232427/verified_data/test"
    ]))

    print(f"\n🚀 开始生成预览图 | 种子: {args.seed}")
    print(f"📂 正在以下 {len(search_dirs)} 个目录中联合搜寻 SVS 文件...")
    
    correct_counts = {c: 0 for c in classes}
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None: continue
            img, om, label, slide_ids = batch
            logits, attn = model(img.to(device), om.to(device))
            
            pred_idx = logits['final_logits'].argmax(1).item()
            true_idx = label.item()
            sid = slide_ids[0]
            clean_sid = sid.replace('.svs', '').replace('.SVS', '')
            
            if pred_idx == true_idx:
                class_name = classes[true_idx]
                correct_counts[class_name] += 1
                
                svs_list = []
                # 1. 遍历所有预设目录，先进行精确匹配
                for s_dir in search_dirs:
                    if not os.path.exists(s_dir): continue
                    svs_list.extend(glob(os.path.join(s_dir, "**", f"{clean_sid}.svs"), recursive=True))
                    svs_list.extend(glob(os.path.join(s_dir, "**", f"{clean_sid}.SVS"), recursive=True))
                
                # 2. 如果没找到，进行模糊包含匹配
                if not svs_list:
                    for s_dir in search_dirs:
                        if not os.path.exists(s_dir): continue
                        svs_list.extend(glob(os.path.join(s_dir, "**", f"*{clean_sid}*.svs"), recursive=True))
                        svs_list.extend(glob(os.path.join(s_dir, "**", f"*{clean_sid}*.SVS"), recursive=True))
                
                if not svs_list:
                    print(f"\n⚠️ 预测正确 [{class_name}]，但在三大目录中均未找到 SVS 文件: {sid}")
                    continue
                    
                h5_list = glob(os.path.join(args.img_features_dir, f"*{clean_sid}*.h5"))
                if not h5_list:
                    print(f"\n⚠️ 预测正确 [{class_name}]，但未找到 H5 特征文件: {sid}")
                    continue
                    
                h5_p = h5_list[0]
                with h5py.File(h5_p, 'r') as f:
                    coords = f['coords'][:] if 'coords' in f else f['coordinates'][:]
                    
                save_dir = os.path.join(args.output_dir, class_name)
                # 使用搜寻到的第一个有效 SVS 路径绘图
                save_preview(svs_list[0], attn[0].cpu().numpy(), coords, class_name, save_dir)
                
    print("\n✅ 推理完毕！实际预测正确并成功出图的各类别数量统计：")
    for k, v in correct_counts.items():
        print(f"  - {k}: {v} 张")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--svs_dir', type=str, default='/seu_nvme/home/liuhongde/220232427') # 给了默认值，作为备用
    parser.add_argument('--model_dir', type=str, default='./outputs_multimodel')
    parser.add_argument('--output_dir', type=str, default='./preview_results')
    parser.add_argument('--csv_file', type=str, default='/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv')
    parser.add_argument('--img_features_dir', type=str, default='/seu_nvme/home/liuhongde/220232427/features_h5')
    parser.add_argument('--omics_base_dir', type=str, default='/seu_nvme/home/liuhongde/220232427')
    parser.add_argument('--split_dir', type=str, default='/seu_nvme/home/liuhongde/220232427')
    parser.add_argument('--device', type=str, default='cuda')
    main(parser.parse_args())