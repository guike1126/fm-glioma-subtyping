import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gprofiler import GProfiler
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_inputs")
OUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_results")
os.makedirs(OUT_DIR, exist_ok=True)

gp = GProfiler(return_dataframe=True)

# 只需要重跑这几个超时的文件
failed_files = [
    "Astro_Grade_Astro_G3_Expression.txt",
    "Final_6Class_Astrocytoma_G3_Expression.txt",
    "Final_6Class_Astrocytoma_G4_Expression.txt",
    "Final_6Class_Oligodendroglioma_G2_Expression.txt"
]

def plot_bubble(df, title, save_path):
    if df.empty: return
    df_plot = df.sort_values("p_value").head(10).copy()
    df_plot["logP"] = -np.log10(df_plot["p_value"])
    
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df_plot, x="logP", y="name", size="intersection_size", 
        hue="p_value", palette="viridis_r", sizes=(50, 400), alpha=0.8
    )
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel("-log10(p-value)", fontsize=12)
    plt.ylabel("Pathway / GO Term", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

for filename in failed_files:
    txt_file = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(txt_file):
        print(f"找不到文件: {txt_file}")
        continue
        
    base_name = filename.replace(".txt", "")
    with open(txt_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
        
    print(f"正在重新分析: {base_name} ...")
    
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        try:
            # 增加一个较大的超时设置
            res = gp.profile(organism="hsapiens", query=genes, sources=["GO:BP", "KEGG"])
            
            if res.empty:
                print(f"  ⚠️ {base_name} 没有富集到显著结果。")
            else:
                out_df = res[["source", "name", "p_value", "term_size", "query_size", "intersection_size"]]
                out_df.to_csv(os.path.join(OUT_DIR, f"{base_name}_enrich.csv"), index=False)
                
                go_df = out_df[out_df["source"] == "GO:BP"]
                kegg_df = out_df[out_df["source"] == "KEGG"]
                
                if not go_df.empty:
                    plot_bubble(go_df, f"GO:BP Enrichment - {base_name}", os.path.join(OUT_DIR, f"{base_name}_GO.png"))
                if not kegg_df.empty:
                    plot_bubble(kegg_df, f"KEGG Enrichment - {base_name}", os.path.join(OUT_DIR, f"{base_name}_KEGG.png"))
                print(f"  ✅ 成功: {base_name}")
            
            success = True
            break # 成功则跳出重试循环
            
        except Exception as e:
            print(f"  ❌ 第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                print("  等待 5 秒后重试...")
                time.sleep(5)
                
    if not success:
        print(f"  🚨 {base_name} 重试 {max_retries} 次仍然失败，请检查网络。")
        
    # 每次请求完稍微歇息一下，防封
    time.sleep(2)

print("\n✅ 重跑任务结束！")