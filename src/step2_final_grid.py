import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import h5py
import cv2
import openslide
from scipy.ndimage import gaussian_filter
from glob import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ====== 全局字体设置 ======
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20

from multimodal_loader import create_multimodal_datasets, get_stage_loader


# ============================================================
# 1. 模型架构定义
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
        self.subtype = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(256, 3))
        self.grade_bottleneck = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
                                              nn.Dropout(dropout))
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

        attn_logits = self.pool(i_emb) / 0.5
        A = torch.softmax(attn_logits, dim=1)

        i_global = torch.sum(i_emb * A, dim=1)
        final_feat = torch.cat([i_global, o_emb.squeeze(1)], dim=1)
        return self.heads(final_feat), A.squeeze(-1)


# ============================================================
# 2. 核心辅助逻辑
# ============================================================
ATTENTION_COLORS = ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
cmap_custom = LinearSegmentedColormap.from_list('attn', ATTENTION_COLORS, N=256)


def is_solid_tissue(patch_np, min_ratio=0.75, white_thresh=230):
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < white_thresh
    ratio = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return ratio >= min_ratio


def create_top_patches_grid(svs_path, coords, attn, ps=256, gap=15):
    slide = openslide.OpenSlide(svs_path)
    sorted_idx = np.argsort(attn)[::-1]
    valid_patches = []
    for idx in sorted_idx:
        x, y = coords[idx]
        cx, cy = int(x + ps // 2), int(y + ps // 2)
        p = slide.read_region((cx - ps // 2, cy - ps // 2), 0, (ps, ps)).convert('RGB')
        p_np = np.array(p)
        if is_solid_tissue(p_np, min_ratio=0.75):
            valid_patches.append(p_np)
        if len(valid_patches) == 5: break
    slide.close()
    while len(valid_patches) < 5:
        valid_patches.append(np.full((ps, ps, 3), 255, dtype=np.uint8))
    grid_w, grid_h = ps * 3 + gap * 4, ps * 2 + gap * 3
    canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for i in range(3):
        canvas[gap:gap + ps, gap + i * (ps + gap):gap + i * (ps + gap) + ps] = valid_patches[i]
    offset = (ps + gap) // 2 + gap
    for i in range(3, 5):
        canvas[ps + 2 * gap:2 * ps + 2 * gap, offset + (i - 3) * (ps + gap):offset + (i - 3) * (ps + gap) + ps] = \
        valid_patches[i]
    return canvas


# ============================================================
# 3. 绘图核心函数
# ============================================================
def plot_pro_row(fig, gs, row_idx, sample, ps, alpha, labels=None):
    svs_p, attn, coords, c_name = sample['svs_path'], sample['attn'], sample['coords'], sample['class_name']
    slide = openslide.OpenSlide(svs_p)
    w, h = slide.dimensions
    thumb = np.array(slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1]).convert('RGB'))
    slide.close()

    ranks = np.argsort(np.argsort(attn))
    attn_norm = ranks / (len(ranks) - 1)
    ch, cw = 1024, int(1024 * (w / h))
    hb = np.full((ch, cw), -1.0, dtype=np.float32)
    psc = int(np.ceil(ps * (ch / h)))
    for i, (px, py) in enumerate(coords):
        sx, sy = int(px * (cw / w)), int(py * (ch / h))
        hb[sy:sy + max(1, psc), sx:sx + max(1, psc)] = attn_norm[i]

    hr = cv2.resize(hb, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    hs = gaussian_filter(np.where(hr < 0, 0, hr), sigma=thumb.shape[1] // 100)
    hm, hsm = np.ma.masked_where(hr < 0, hr), np.ma.masked_where(hr < 0, hs)
    pg = create_top_patches_grid(svs_p, coords, attn, ps)

    x_offsets = [0.01, 0.005, 0.008, 0.001]

    for col in range(4):
        ax = fig.add_subplot(gs[row_idx, col])
        if col == 0:
            ax.imshow(thumb)
            ax.set_ylabel(c_name.replace('_', '\n'), fontsize=20, fontweight='bold', rotation=0, labelpad=100,
                          va='center', ha='center')
        elif col == 1:
            ax.set_facecolor('white')
            ax.imshow(hm, cmap=cmap_custom, vmin=0, vmax=1)
        elif col == 2:
            ax.imshow(thumb)
            ax.imshow(hsm, cmap=cmap_custom, alpha=alpha, vmin=0, vmax=1)
        else:
            ax.imshow(pg)

        if row_idx == 0:
            ax.set_title(["Original", "Attention", "Overlay", "Top 5 Patches"][col], fontsize=24, pad=35,
                         fontweight='bold')

        if labels and col < len(labels):
            cell_pos = gs[row_idx, col].get_position(fig)
            ax_pos = ax.get_position()
            current_x_offset = x_offsets[col]
            x_final = cell_pos.x0 - current_x_offset
            y_final = ax_pos.y1

            fig.text(x_final, y_final, labels[col], transform=fig.transFigure,
                     fontsize=28, fontweight='bold', va='top', ha='left')

        ax.set_xticks([])
        ax.set_yticks([])
        [s.set_visible(False) for s in ax.spines.values()]


# ============================================================
# 4. 主程序
# ============================================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_ids = [s.strip().replace('.svs', '').replace('.SVS', '') for s in args.ids.split(',')]
    
    _, _, test_ds, classes, input_dim = create_multimodal_datasets(
        args.csv_file, args.img_features_dir, args.omics_base_dir, args.split_dir, args.seed
    )
    test_loader = get_stage_loader(test_ds, None, None, batch_size=1, is_train=False)

    model = DeepMultimodalModelForVis(omics_input_dim=input_dim).to(device)
    weight_path = os.path.join(args.model_dir, f"seed_{args.seed}", "best_final.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()

    samples = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="提取数据"):
            if batch is None: continue
            img, om, label, slide_ids = batch
            sid = slide_ids[0].replace('.svs', '').replace('.SVS', '')
            
            if sid in target_ids:
                _, attn = model(img.to(device), om.to(device))
                c_name = classes[label.item()]
                
                # 查找svs路径 (加了防报错保护)
                svs_paths = glob(os.path.join(args.svs_dir, "**", slide_ids[0]), recursive=True)
                if not svs_paths:
                    print(f"⚠️ 警告: 找不到 {slide_ids[0]} 的 .svs 文件，跳过该样本。请检查 --svs_dir 路径。")
                    continue
                svs_p = svs_paths[0]
                
                h5_paths = glob(os.path.join(args.img_features_dir, f"*{sid}*.h5"))
                if not h5_paths:
                    print(f"⚠️ 警告: 找不到 {sid} 的 .h5 特征文件，跳过该样本。")
                    continue
                h5_p = h5_paths[0]
                
                with h5py.File(h5_p, 'r') as f:
                    coords = f['coords'][:] if 'coords' in f else f['coordinates'][:]
                    
                samples[c_name] = {'svs_path': svs_p, 'attn': attn[0].cpu().numpy(), 'coords': coords,
                                   'class_name': c_name}

    final_list = [samples.get(c) for c in classes]
    
    # 计算高宽比（跳过为空的）
    ratios = []
    for s in final_list:
        if s:
            slide = openslide.OpenSlide(s['svs_path'])
            ratios.append(slide.dimensions[1] / slide.dimensions[0])
            slide.close()
        else:
            ratios.append(0.8)

    num_rows = len(final_list)
    fig = plt.figure(figsize=(24, sum(ratios) * 5 + 2))
    gs = gridspec.GridSpec(num_rows, 4, height_ratios=ratios, width_ratios=[1, 1, 1, 1.4])

    plt.subplots_adjust(left=0.22, right=0.98, wspace=0.12, hspace=0.35)

    for i, s in enumerate(final_list):
        if s:
            row_labels = [chr(ord('a') + i + (j * num_rows)) for j in range(4)]
            plot_pro_row(fig, gs, i, s, args.patch_size, args.overlay_alpha, labels=row_labels)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    plt.savefig(args.output_file, dpi=200, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--svs_dir', type=str, required=True, help="存放 SVS 文件的根目录")
    parser.add_argument('--ids', type=str, required=True, help="逗号分隔的 slide id，例如 TCGA-xx-xxxx-01Z-00-DX1")
    parser.add_argument('--output_file', type=str, default=r'E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\figures\final_figure_aligned.png')
    parser.add_argument('--model_dir', type=str, default=r'E:\学习\论文2\战桐整理\代码和结果\代码和结果\code1\outputs_multimodel')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--overlay_alpha', type=float, default=0.45)
    
    # ====== 修改为本地绝对路径 ======
    parser.add_argument('--csv_file', type=str, default=r'E:\学习\论文2\战桐整理\imputed_data_OS_filled.csv')
    parser.add_argument('--img_features_dir', type=str, default=r'E:\学习\论文2\战桐整理\features_h5')
    parser.add_argument('--omics_base_dir', type=str, default=r'E:\学习\论文2\战桐整理')
    parser.add_argument('--split_dir', type=str, default=r'E:\学习\论文2\战桐整理')
    
    main(parser.parse_args())