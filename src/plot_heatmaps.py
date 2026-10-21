import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

# ==================== 配置路径 ====================
RESULT_ROOT = "combined_results_seeds_6003_4873_3157_1107_9468"
DATA_PATH = "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv"
SAVE_DIR = os.path.join(RESULT_ROOT, "Gene_Expression")
os.makedirs(SAVE_DIR, exist_ok=True)


def prepare_data_with_labels(df):
    """增加多级标签列，用于样本筛选和分组"""
    df = df.copy()
    df['slide_id'] = df['slide_id'].apply(lambda x: x if x.endswith('.svs') else x + '.svs')

    df['subtype_name'] = df['subtype'].map({
        'GBM': 'GBM',
        'Astrocytoma': 'Astrocytoma',
        'Oligodendroglioma': 'Oligodendroglioma'
    })

    df['grade_num'] = df['grade'].str.extract(r'(\d+)').astype(int)

    def full_label(row):
        sub = row['subtype_name']
        grd = row['grade_num']
        if sub == 'GBM': return 'GBM_G4'
        return f"{sub}_G{grd}"

    df['full_label'] = df.apply(full_label, axis=1)
    df['short_label'] = df['full_label'].str.replace('Astrocytoma', 'Astro').str.replace('Oligodendroglioma', 'Oligo')
    
    return df


def get_gene_universe(task_name, class_folders):
    """读取指定类别下的 top200 基因并集"""
    genes = set()
    for cls_folder in class_folders:
        path = os.path.join(RESULT_ROOT, task_name, cls_folder, "top200_Expression_Clean.csv")
        if os.path.exists(path):
            feat_df = pd.read_csv(path)
            genes.update(feat_df['Gene'].tolist())
    return list(genes)


def plot_clustermap(task_name, title, sample_selector, group_assigner,
                    gene_universe_func, df, col_order=None):
    """
    竖向长方形热图绘制：
    1. 强制 figsize 为竖长比例
    2. 模糊匹配 (只在 13:1013 列搜索带 .x 后缀的基因)
    3. 隐藏 Y 轴基因名，设置 24pt 字号
    """
    # 筛选样本
    sample_mask = sample_selector(df)
    plot_df = df[sample_mask].copy()
    if plot_df.empty: return

    # 获取待匹配的基因名
    selected_genes_clean = gene_universe_func()
    
    # 【核心】锁定基因表达谱所在的列范围 (第 14-1013 列)
    exp_col_universe = df.columns[13:1013].tolist()
    
    # --- 模糊匹配逻辑 ---
    mapping_dict = {} 
    for g in selected_genes_clean:
        if g in exp_col_universe:
            mapping_dict[g] = g
        elif f"{g}.x" in exp_col_universe:
            mapping_dict[f"{g}.x"] = g
        else:
            fuzzy_matches = [c for c in exp_col_universe if c.startswith(f"{g}.")]
            if fuzzy_matches:
                mapping_dict[fuzzy_matches[0]] = g

    raw_cols_to_use = list(mapping_dict.keys())
    
    if not raw_cols_to_use:
        print(f"  [跳过] {task_name}: 范围内未匹配到基因")
        return

    # 构建矩阵并标准化
    X = plot_df[raw_cols_to_use].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    plot_matrix = pd.DataFrame(X_scaled.T, index=raw_cols_to_use, columns=plot_df.index)

    # 样本分组与颜色
    groups = group_assigner(plot_df)
    unique_groups = sorted(groups.unique())
    palette = sns.color_palette("husl", len(unique_groups))
    lut = dict(zip(unique_groups, palette))

    if col_order is not None:
        order_map = {name: i for i, name in enumerate(col_order)}
        plot_df['_order'] = groups.map(lambda x: order_map.get(x, len(col_order)))
        plot_df = plot_df.sort_values('_order')
        plot_matrix = plot_matrix[plot_df.index]
        groups = group_assigner(plot_df)

    col_colors = groups.map(lut)

    # ==================== 核心：形状控制 ====================
    # 强制设定为竖向长方形：宽度小（10），高度大（18）
    figsize = (15, 18)

    g = sns.clustermap(
        plot_matrix,
        col_colors=col_colors,
        cmap="RdBu_r",
        center=0,
        method='ward',
        metric='euclidean',
        xticklabels=False,
        yticklabels=False,        # 隐藏基因名
        dendrogram_ratio=(0.1, 0.05), # 压缩树状图空间
        figsize=figsize,
        row_cluster=True,
        col_cluster=False         # 样本按指定顺序排列，不聚类
    )

    g.ax_col_dendrogram.set_visible(False)
    g.gs.update(wspace=0.01) # 贴合间距

    # ==================== 布局与字号 (24pt) ====================
    hm_pos = g.ax_heatmap.get_position()
    
    # 重新放置 Colorbar
    cbar_ax = g.fig.add_axes([hm_pos.x1 + 0.02, hm_pos.y0, 0.03, 0.15])
    cb = plt.colorbar(g.ax_heatmap.get_children()[0], cax=cbar_ax)
    cb.ax.tick_params(labelsize=24)
    cb.set_label('Z-Score', size=24)

    # 放置图例
    handles = [Patch(facecolor=color, label=name) for name, color in lut.items()]
    g.ax_heatmap.legend(
        handles=handles, title="Group",
        bbox_to_anchor=(1.05, 0.25), loc='lower left',
        fontsize=24, title_fontsize=24, frameon=True
    )

    g.fig.suptitle(f"{title}", fontsize=28, y=0.98)

    save_path = os.path.join(SAVE_DIR, f"{task_name}_vertical_rectangle.png")
    g.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [成功] {task_name}: 已保存竖向长方形热图。")


if __name__ == "__main__":
    df_raw = pd.read_csv(DATA_PATH)
    df = prepare_data_with_labels(df_raw)

    tasks = [
        ('Subtype', 'Hierarchical Clustering: Subtype', 
         lambda d: d['subtype_name'].isin(['GBM', 'Astrocytoma', 'Oligodendroglioma']),
         lambda d: d['subtype_name'],
         ['GBM', 'Astrocytoma', 'Oligodendroglioma'],
         ['GBM', 'Astrocytoma', 'Oligodendroglioma']),
        
        ('Astro_Grade', 'Hierarchical Clustering: Astro Grades',
         lambda d: d['subtype_name'] == 'Astrocytoma',
         lambda d: d['short_label'],
         ['Astro_G2', 'Astro_G3', 'Astro_G4'],
         ['Astro_G2', 'Astro_G3', 'Astro_G4']),
         
        ('Oligo_Grade', 'Hierarchical Clustering: Oligo Grades',
         lambda d: d['subtype_name'] == 'Oligodendroglioma',
         lambda d: d['short_label'],
         ['Oligo_G2', 'Oligo_G3'],
         ['Oligo_G2', 'Oligo_G3']),

        ('Final_6Class', 'Hierarchical Clustering: 6 Classes',
         lambda d: d['full_label'] != 'Other',
         lambda d: d['full_label'],
         ['Astrocytoma_G2', 'Astrocytoma_G3', 'Astrocytoma_G4', 'GBM_G4', 'Oligodendroglioma_G2', 'Oligodendroglioma_G3'],
         ['Astrocytoma_G2', 'Astrocytoma_G3', 'Astrocytoma_G4', 'GBM_G4', 'Oligodendroglioma_G2', 'Oligodendroglioma_G3'])
    ]

    for name, title, sel, ass, fld, ord_list in tasks:
        u_func = lambda f=fld, n=name: get_gene_universe(n, f)
        plot_clustermap(name, title, sel, ass, u_func, df, col_order=ord_list)

    print("\n所有竖向热图已生成完成！")