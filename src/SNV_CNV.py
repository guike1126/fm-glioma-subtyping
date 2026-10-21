import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import re

# ==========================
# 1. 基础配置
# ==========================
SEEDS = [6003, 4873, 3157, 1107, 9468] 
SEED_STR = "_".join(map(str, SEEDS))

CONFIG = {
    'raw_csv': "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv",
    'input_root': f"combined_results_seeds_{SEED_STR}",
    'export_subdir': "SNV_CNV",
    'show_n_genes': {'SNV': 20, 'CNV': 10},
    'top50_sample_size': 50 
}

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================
# 2. 智能纵横比绘图引擎
# ==========================
def plot_discrete_landscape(matrix, type_label, class_name, output_path, mode="Top50"):
    # --- A. 样本筛选与排序 ---
    if mode == "All":
        n_genes, n_samples = matrix.shape
        if n_samples == 0 or n_genes == 0: return
        if n_samples > 1:
            from scipy.cluster.hierarchy import linkage, dendrogram
            try:
                Z = linkage(matrix.T, method='ward')
                sample_order = dendrogram(Z, no_plot=True)['leaves']
                matrix = matrix.iloc[:, sample_order]
            except: pass
        show_xticks = False
        
        # 全局图逻辑：固定宽度，高度随基因数增加
        fig_width = 20
        fig_height = max(6, n_genes * 0.5) 
    else:
        # TopFiltered 模式：先进行样本筛选
        sample_signal = matrix.abs().sum(axis=0)
        matrix = matrix.loc[:, sample_signal > 0]
        if matrix.shape[1] > CONFIG['top50_sample_size']:
            top_idx = sample_signal.sort_values(ascending=False).head(CONFIG['top50_sample_size']).index
            matrix = matrix[top_idx]
        
        gene_freq = (matrix != 0).sum(axis=1)
        matrix = matrix.loc[gene_freq.sort_values(ascending=False).index]
        
        n_genes, n_samples = matrix.shape
        if n_samples == 0 or n_genes == 0: return
        show_xticks = False # 统一不显示样本 ID

        # --- B. 核心改进：正方形感智能尺寸计算 ---
        # 考虑到左侧标签和右侧图注约占 4-5 英寸的空间
        # 1. 计算高度：每个基因行约占 0.7 英寸，设定最小高度保证美观
        fig_height = max(6, n_genes * 0.7) 
        
        # 2. 计算宽度：
        # 为了让“热图区域”接近正方形，总宽度 = 理想高度 + 标签/图注补偿
        # 补偿值通常在 4 到 6 之间。同时设定最小总宽度为 11，防止变成瘦长条。
        fig_width = max(11, fig_height + 5)
        
        # 3. 动态比例调整：确保宽度至少是高度的 1.1 倍（视觉上更趋于平衡的正方形感）
        if fig_width < fig_height * 1.1:
            fig_width = fig_height * 1.1

    # --- C. 颜色与图注配置 ---
    if type_label == "SNV":
        colors = ["#FFFFFF", "#E84133"] 
        cmap = ListedColormap(colors)
        plot_data = np.where(matrix.values == 1, 1, 0)
        legend_info = [(colors[0], "Wild-type"), (colors[1], "Mutant")]
    else:
        colors = ["#A50026", "#F46D43", "#FFFFFF", "#ABD9E9", "#313695"]
        cmap = ListedColormap(colors)
        raw_vals = matrix.values
        plot_data = np.full(raw_vals.shape, 2, dtype=int) 
        plot_data[raw_vals == -2] = 0 
        plot_data[raw_vals == -1] = 1 
        plot_data[raw_vals == 0]  = 2 
        plot_data[raw_vals == 1]  = 3 
        plot_data[raw_vals == 2]  = 4 
        legend_info = [
            (colors[0], "Deep Deletion"), (colors[1], "Shallow Loss"), 
            (colors[2], "Diploid / Neutral"), (colors[3], "Low-level Gain"), 
            (colors[4], "High-level Amplification")
        ]

    # --- D. 绘图执行 ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # aspect='auto' 关键：它会根据我们设定的 figsize 拉伸单元格，使其填满画布
    ax.imshow(plot_data, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=len(colors)-1)
    
    ax.set_yticks(np.arange(n_genes))
    ax.set_yticklabels(matrix.index, fontsize=18, fontweight='bold')
    
    if show_xticks:
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    # 增强块状感：设置网格
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_genes, 1), minor=True)
    ax.grid(which="minor", color="#ECECEC", linestyle='-', linewidth=1.2)
    ax.tick_params(which="minor", size=0)

    # 优化图注位置
    patches = [mpatches.Patch(color=c, label=l) for c, l in legend_info]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., title=f"{type_label} Alteration", frameon=False, fontsize=20, title_fontsize=20)

    plt.title(f"{type_label} Landscape: {class_name}", fontsize=22, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================
# 3. 主程序逻辑 (保持不变)
# ==========================
def run_main():
    full_df = pd.read_csv(CONFIG['raw_csv'])
    tasks = ['Subtype', 'Astro_Grade', 'Oligo_Grade', 'Final_6Class']
    
    for task in tasks:
        task_path = os.path.join(CONFIG['input_root'], task)
        if not os.path.exists(task_path): continue
        
        for cls in [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]:
            cls_dir = os.path.join(task_path, cls)
            out_dir = os.path.join(cls_dir, CONFIG['export_subdir'])
            os.makedirs(out_dir, exist_ok=True)
            
            try:
                f_mut = [f for f in os.listdir(cls_dir) if "Mutations_Clean" in f][0]
                df_mut = pd.read_csv(os.path.join(cls_dir, f_mut))
            except: continue

            sub_df = get_task_specific_samples(full_df, task, cls)
            if sub_df.empty: continue

            for t_label in ["SNV", "CNV"]:
                n_limit = CONFIG['show_n_genes'].get(t_label, 20)
                target_genes = df_mut[df_mut['Type'] == t_label].head(n_limit)['Gene'].tolist()
                valid_genes = [g for g in target_genes if g in sub_df.columns]
                
                if not valid_genes: continue
                
                matrix_raw = sub_df[valid_genes].T.copy()
                if t_label == "SNV":
                    matrix_clean = matrix_raw.apply(lambda x: np.where(x == 1, 1, 0)).astype(int)
                else:
                    valid_vals = [-2, -1, 1, 2]
                    matrix_clean = matrix_raw.apply(lambda x: np.where(x.isin(valid_vals), x, 0)).astype(int)
                
                # 同时导出两种模式
                plot_discrete_landscape(matrix_clean, t_label, cls, 
                                       os.path.join(out_dir, f"{t_label}_Global.png"), mode="All")
                plot_discrete_landscape(matrix_clean, t_label, cls, 
                                       os.path.join(out_dir, f"{t_label}_TopFiltered.png"), mode="Top50")

def get_task_specific_samples(full_df, task_name, class_name):
    df = full_df.copy()
    cn = class_name.lower()
    if task_name == 'Subtype' or any(x in cn for x in ['astro', 'gbm', 'oligo']):
        if 'astro' in cn: df = df[df['subtype'].str.contains('Astro', case=False, na=False)]
        elif 'gbm' in cn: df = df[df['subtype'].str.contains('GBM', case=False, na=False)]
        elif 'oligo' in cn: df = df[df['subtype'].str.contains('Oligo', case=False, na=False)]
    grade_match = re.search(r'g(\d)', cn)
    if grade_match:
        g_num = grade_match.group(1)
        df = df[df['grade'].str.contains(f'G{g_num}', case=False, na=False)]
    return df

if __name__ == "__main__":
    run_main()