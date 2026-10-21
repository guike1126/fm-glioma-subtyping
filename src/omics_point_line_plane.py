import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_integrated_analysis():
    # --- 1. 路径配置 ---
    csv_file = "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv"
    results_root = "/seu_nvme/home/liuhongde/220232427/code1/combined_results_seeds_6003_4873_3157_1107_9468"
    # 输出子目录
    enrich_dir = os.path.join(results_root, "Enrichment_Gene_Lists")
    network_dir = os.path.join(results_root, "Network_Analysis_Files")
    os.makedirs(enrich_dir, exist_ok=True)
    os.makedirs(network_dir, exist_ok=True)

    print("⏳ 正在加载原始数据集...")
    raw_df = pd.read_csv(csv_file)
    
    all_edges = []
    all_nodes = set()

    print("\n🚀 开始全方位分析（点、线、面）...")
    print("=" * 60)

    # 遍历任务和类别
    for task in os.listdir(results_root):
        task_path = os.path.join(results_root, task)
        if not os.path.isdir(task_path): continue
        
        for c_name in os.listdir(task_path):
            cls_path = os.path.join(task_path, c_name)
            corr_file = os.path.join(cls_path, "Methy_Exp_Negative_Regulation_Strong.csv")
            
            if not os.path.exists(corr_file):
                continue
            
            # 读取强相关数据
            corr_df = pd.read_csv(corr_file)
            if corr_df.empty: continue

            # --- 【点】：绘制该类别最强的 Top 1 散点图 ---
            top_pair = corr_df.iloc[0]
            m_feat = top_pair['Methylation_Site']
            e_feat = top_pair['Expression_Gene']
            r_val = top_pair['Pearson_Correlation']

            plt.figure(figsize=(6, 5))
            sns.set_style("whitegrid")
            sns.regplot(x=raw_df[m_feat], y=raw_df[e_feat], 
                        scatter_kws={'alpha':0.4, 's':15, 'color':'#2c3e50'}, 
                        line_kws={'color':'#e74c3c', 'lw':2})
            
            plt.title(f"Class: {c_name}\nStrongest Negative Regulation", fontsize=10)
            plt.xlabel(f"Methylation: {m_feat}")
            plt.ylabel(f"Expression: {e_feat}")
            plt.annotate(f"Pearson r = {r_val:.4f}", xy=(0.05, 0.9), xycoords='axes fraction', 
                         fontsize=11, fontweight='bold', color='#c0392b',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(cls_path, f"Scatter_Top1_Reg.png"), dpi=300)
            plt.close()

            # --- 【线】：收集网络边数据 (r < -0.8) ---
            strong_net = corr_df[corr_df['Pearson_Correlation'] < -0.8].copy()
            if not strong_net.empty:
                strong_net['Task'] = task
                strong_net['Class'] = c_name
                all_edges.append(strong_net)
                all_nodes.update(strong_net['Methylation_Site'].unique())
                all_nodes.update(strong_net['Expression_Gene'].unique())

            # --- 【面】：提取待富集基因列表 (r < -0.5) ---
            regulated_genes = corr_df[corr_df['Pearson_Correlation'] < -0.5]['Expression_Gene'].unique()
            if len(regulated_genes) >= 5:
                list_filename = f"Genes_{task}_{c_name}_r0.5.txt"
                with open(os.path.join(enrich_dir, list_filename), 'w') as f:
                    f.write("\n".join(regulated_genes))

            print(f"✅ 已完成类别: {c_name} (Task: {task})")

    # --- 汇总输出网络文件 ---
    if all_edges:
        master_edges = pd.concat(all_edges)
        master_edges.to_csv(os.path.join(network_dir, "Cytoscape_Edges_r0.8.csv"), index=False)
        
        # 节点属性表
        node_attr = []
        for n in all_nodes:
            # 判断节点类型：如果是甲基化位点则标记
            n_type = "Methylation" if n in master_edges['Methylation_Site'].values else "Expression"
            node_attr.append({'Node': n, 'Type': n_type})
        pd.DataFrame(node_attr).to_csv(os.path.join(network_dir, "Cytoscape_Nodes_Attr.csv"), index=False)

    print("\n" + "=" * 60)
    print("🎉 任务全部圆满完成！")
    print(f"1. 【点】：散点图已存至各类别文件夹下。")
    print(f"2. 【线】：Cytoscape 导入文件见: {network_dir}")
    print(f"3. 【面】：富集分析基因集见: {enrich_dir}")

if __name__ == "__main__":
    run_integrated_analysis()