import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 基础配置 =================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20  # 全局字体进一步放大

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ================= 核心数据提取函数 =================
def build_matrix(modality, top_n=5):
    matrix_data = []
    
    FINAL_CLASSES = [
        'Final_6Class_Astrocytoma_G2', 
        'Final_6Class_Astrocytoma_G3', 
        'Final_6Class_Astrocytoma_G4',
        'Final_6Class_Oligodendroglioma_G2', 
        'Final_6Class_Oligodendroglioma_G3', 
        'Final_6Class_GBM_G4'
    ]
    
    for cls in FINAL_CLASSES:
        filepath = os.path.join(INPUT_DIR, f"{cls}_{modality}_enrich.csv")
        if not os.path.exists(filepath):
            continue
        
        df = pd.read_csv(filepath)
        df = df[df['source'].isin(['GO:BP', 'KEGG'])].copy()
        if df.empty: continue
        
        df = df.sort_values('p_value').head(top_n)
        df['Term'] = df['source'].str.replace('GO:BP', 'GO') + ': ' + df['name']
        
        for _, row in df.iterrows():
            matrix_data.append({
                'Category': cls,
                'Term': row['Term'],
                'logP': -np.log10(row['p_value'])
            })
    
    if not matrix_data: return None
        
    res_df = pd.DataFrame(matrix_data)
    pivot_df = res_df.pivot(index='Term', columns='Category', values='logP')
    
    for c in FINAL_CLASSES:
        if c not in pivot_df.columns: pivot_df[c] = np.nan
            
    pivot_df['max_logP'] = pivot_df.max(axis=1)
    pivot_df = pivot_df.sort_values('max_logP', ascending=False).head(15)
    pivot_df = pivot_df.drop(columns=['max_logP']).reindex(columns=FINAL_CLASSES)
    
    short_cols = []
    for col in pivot_df.columns:
        c = col.replace('Final_6Class_', '')
        c = c.replace('Astrocytoma', 'Astro').replace('Oligodendroglioma', 'Oligo')
        short_cols.append(c)
    pivot_df.columns = short_cols
    pivot_df.index = [x if len(x) < 45 else x[:42] + '...' for x in pivot_df.index]
    
    return pivot_df

# ================= 绘图函数 =================
def plot_combined_2x2():
    mat_exp = build_matrix("Expression", top_n=5)
    mat_snv = build_matrix("SNV", top_n=5)
    mat_cnv = build_matrix("CNV", top_n=5)
    mat_meth = build_matrix("Methylation", top_n=5)

    # 画布再次拉宽，保证大字号下依然有足够空间
    fig, axes = plt.subplots(2, 2, figsize=(42, 24))

    cbar_args = {'label': '-log10(p)', 'shrink': 0.8, 'pad': 0.03}
    # 热图内部数字字体加大到 20
    annot_kws = {"size": 20, "weight": "bold"}

    if mat_exp is not None:
        ax = sns.heatmap(mat_exp, cmap="YlOrRd", ax=axes[0, 0], cbar_kws=cbar_args, 
                    annot=True, fmt=".1f", annot_kws=annot_kws, linewidths=0.8, linecolor='lightgrey')
        ax.set_title("(a) Expression: Proliferative Programs", fontsize=32, fontweight='bold', pad=20)
    
    if mat_snv is not None:
        ax = sns.heatmap(mat_snv, cmap="Reds", ax=axes[0, 1], cbar_kws=cbar_args, 
                    annot=True, fmt=".1f", annot_kws=annot_kws, linewidths=0.8, linecolor='lightgrey')
        ax.set_title("(b) SNV: Oncogenic Drivers", fontsize=32, fontweight='bold', pad=20)
    
    if mat_cnv is not None:
        ax = sns.heatmap(mat_cnv, cmap="Oranges", ax=axes[1, 0], cbar_kws=cbar_args, 
                    annot=True, fmt=".1f", annot_kws=annot_kws, linewidths=0.8, linecolor='lightgrey')
        ax.set_title("(c) CNV: Structural Remodeling", fontsize=32, fontweight='bold', pad=20)

    if mat_meth is not None:
        ax = sns.heatmap(mat_meth, cmap="Greens", ax=axes[1, 1], cbar_kws=cbar_args, 
                    annot=True, fmt=".1f", annot_kws=annot_kws, linewidths=0.8, linecolor='lightgrey')
        ax.set_title("(d) Methylation: Microenvironment", fontsize=32, fontweight='bold', pad=20)

    for ax in axes.flatten():
        ax.set_xlabel("Glioma Categories", fontsize=26, fontweight='bold', labelpad=20)
        ax.set_ylabel("") 
        # 横轴（X）与纵轴（Y）的刻度标签放大到 22
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=22, fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontsize=22, fontweight='bold')
        
        # 色条字体
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('-log10(p)', size=24, fontweight='bold', labelpad=20)

    # 强行拉大间距，防止大字体打架
    plt.subplots_adjust(wspace=0.8, hspace=0.35, bottom=0.1)
    
    save_path = os.path.join(OUT_DIR, "Figure7_Final6Class_Combined.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 成功生成巨无霸字号版图片: {save_path}")

if __name__ == "__main__":
    print("🚀 正在生成 2x2 综合全景图...")
    plot_combined_2x2()
    print("🎉 完成！")