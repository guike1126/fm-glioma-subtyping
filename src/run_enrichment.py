import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gprofiler import GProfiler
from glob import glob

# 自动获取当前绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_inputs")
OUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_results")
os.makedirs(OUT_DIR, exist_ok=True)

print("初始化 gprofiler...")
gp = GProfiler(return_dataframe=True)

def plot_bubble(df, title, save_path):
    if df.empty: return
    # 取前10个最显著的通路绘图
    df_plot = df.sort_values("p_value").head(10).copy()
    
    # 计算 -log10(p_value) 方便可视化
    df_plot["logP"] = -np.log10(df_plot["p_value"])
    
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_plot, 
        x="logP", 
        y="name", 
        size="intersection_size", 
        hue="p_value", 
        palette="viridis_r", 
        sizes=(50, 400), 
        alpha=0.8
    )
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("-log10(p-value)", fontsize=12)
    plt.ylabel("Pathway / GO Term", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

txt_files = glob(os.path.join(INPUT_DIR, "*.txt"))
print(f"共找到 {len(txt_files)} 个输入文件，开始批量分析...\n")

for txt_file in txt_files:
    base_name = os.path.basename(txt_file).replace(".txt", "")
    with open(txt_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    # 过滤掉基因太少的模态（比如SNV只有30个，跑也行，但富集大概率不显著）
    if len(genes) < 5: 
        print(f"跳过 {base_name}: 基因数量不足5个")
        continue
        
    print(f"正在分析: {base_name} ...")
    
    try:
        # 跑 gprofiler，只保留 KEGG 和 GO:BP (生物过程)
        res = gp.profile(organism="hsapiens", query=genes, sources=["GO:BP", "KEGG"])
        
        if res.empty:
            print(f"  ⚠️ {base_name} 没有富集到显著结果。")
            continue
            
        # 保存结果表
        out_df = res[["source", "name", "p_value", "term_size", "query_size", "intersection_size"]]
        out_csv = os.path.join(OUT_DIR, f"{base_name}_enrich.csv")
        out_df.to_csv(out_csv, index=False)
        
        # 分离 GO 和 KEGG 画图
        go_df = out_df[out_df["source"] == "GO:BP"]
        kegg_df = out_df[out_df["source"] == "KEGG"]
        
        if not go_df.empty:
            plot_bubble(go_df, f"GO:BP Enrichment - {base_name}", os.path.join(OUT_DIR, f"{base_name}_GO.png"))
        if not kegg_df.empty:
            plot_bubble(kegg_df, f"KEGG Enrichment - {base_name}", os.path.join(OUT_DIR, f"{base_name}_KEGG.png"))
            
    except Exception as e:
        print(f"  ❌ 分析 {base_name} 时出错: {e}")

print(f"\n🎉 全部完成！结果保存在 {OUT_DIR}")