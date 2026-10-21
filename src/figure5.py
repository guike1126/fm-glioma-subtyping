import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = SCRIPT_DIR
SAVE_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(SAVE_DIR, exist_ok=True)

TASK_FOLDERS = ['Subtype', 'Astro_Grade', 'Oligo_Grade', 'Final_6Class']
EXCLUDE_FOLDERS = {'GO', 'KEGG', 'go', 'kegg'}

def shorten_name(name):
    n = str(name)
    n = n.replace('Oligodendroglioma', 'Oligo').replace('Astrocytoma', 'Astro').replace('Glioblastoma', 'GBM')
    return n

def format_label(task, class_name):
    cls = shorten_name(class_name)
    if task == 'Subtype': return f"{cls} (Subtype)"
    elif task in ('Astro_Grade', 'Oligo_Grade'): return cls
    elif task == 'Final_6Class': return f"{cls} (Final)"
    return cls

def discover_all_classes():
    all_classes = []
    for task in TASK_FOLDERS:
        task_dir = os.path.join(OUTPUT_ROOT, task)
        if not os.path.isdir(task_dir): continue
        subdirs = sorted([d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))])
        subdirs = [d for d in subdirs if d not in EXCLUDE_FOLDERS]
        for cls in subdirs: all_classes.append((task, cls))
    return all_classes

def load_top_features(task, class_name, top_per_modality=200):
    folder = os.path.join(OUTPUT_ROOT, task, class_name)
    if not os.path.exists(folder): return None
    features = []
    
    exp_file = os.path.join(folder, 'top200_Expression_Clean.csv')
    if os.path.exists(exp_file):
        df = pd.read_csv(exp_file).head(top_per_modality)
        for _, row in df.iterrows(): features.append({'Importance': row['Importance'], 'Modality': 'Expression'})
            
    mut_file = os.path.join(folder, 'top60_Mutations_Clean.csv')
    if os.path.exists(mut_file):
        df = pd.read_csv(mut_file)
        snv_df = df[df['Type'] == 'SNV'].head(top_per_modality)
        cnv_df = df[df['Type'] == 'CNV'].head(top_per_modality)
        for _, row in snv_df.iterrows(): features.append({'Importance': row['Importance'], 'Modality': 'SNV'})
        for _, row in cnv_df.iterrows(): features.append({'Importance': row['Importance'], 'Modality': 'CNV'})
            
    meth_file = os.path.join(folder, 'top100_Methylation_Clean.csv')
    if os.path.exists(meth_file):
        df = pd.read_csv(meth_file).head(top_per_modality)
        for _, row in df.iterrows(): features.append({'Importance': row['Importance'], 'Modality': 'Methylation'})
            
    return pd.DataFrame(features) if features else None

def plot_modality_overview_heatmap():
    all_classes = discover_all_classes()
    rows = []
    
    for task, cls in all_classes:
        df = load_top_features(task, cls, top_per_modality=200)
        if df is None or len(df) == 0: continue
        modal_sum = df.groupby('Modality')['Importance'].sum()
        row = {'Label': format_label(task, cls)}
        for m in ['Expression', 'SNV', 'CNV', 'Methylation']: row[m] = modal_sum.get(m, 0)
        rows.append(row)

    if not rows: return

    summary_df = pd.DataFrame(rows).set_index('Label')
    modality_cols = ['Expression', 'SNV', 'CNV', 'Methylation']
    summary_df = summary_df[modality_cols]
    row_norm = summary_df.div(summary_df.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 12))  # 画布稍微调整

    # 【修改点1】：改为更柔和的浅色系 'OrRd'，不再用对比刺眼的 'YlOrRd'
    sns.heatmap(
        row_norm,
        annot=summary_df.values,
        fmt='.3f',
        cmap='OrRd',  # 柔和的浅橘红色
        ax=ax,
        cbar_kws={'label': 'Normalized Importance', 'shrink': 0.7},
        linewidths=1.0,
        linecolor='white',
        # 【修改点2】：热图数字字号变小到 13
        annot_kws={'size': 13, 'weight': 'bold'}, 
    )

    # 【修改点3】：标题字号稍微缩小到 22，轴标签加大到 18
    ax.set_title('Modality Contribution Across Tasks and Classes', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel('Modality', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('Task / Class', fontsize=18, fontweight='bold', labelpad=15)
    
    # 【修改点4】：刻度标签设置为 15
    ax.tick_params(axis='x', labelsize=15, rotation=0)
    ax.tick_params(axis='y', labelsize=15, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Normalized Importance', fontsize=14, fontweight='bold', labelpad=15)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'modality_contribution_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_modality_overview_heatmap()