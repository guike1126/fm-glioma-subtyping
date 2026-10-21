import os
import pandas as pd
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_results")
OUT_FILE = os.path.join(SCRIPT_DIR, "summary.txt")

csv_files = glob(os.path.join(INPUT_DIR, "*_enrich.csv"))

summary_lines = []
summary_lines.append("="*50)
summary_lines.append("富集分析核心结果提取 (Top 15)")
summary_lines.append("="*50 + "\n")

# 只要包含这些关键词的文件，我们都汇总一下
for csv_file in sorted(csv_files):
    filename = os.path.basename(csv_file).replace("_enrich.csv", "")
    
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
            
        # 按 p 值排序
        df = df.sort_values("p_value")
        
        # 看看是不是有GO或者KEGG
        has_go = not df[df["source"] == "GO:BP"].empty
        has_kegg = not df[df["source"] == "KEGG"].empty
        
        if not has_go and not has_kegg:
            continue
            
        summary_lines.append(f"【分析对象】: {filename}")
        
        # 提取 Top 3 GO
        if has_go:
            top_go = df[df["source"] == "GO:BP"].head(3)
            for _, row in top_go.iterrows():
                summary_lines.append(f"  - GO: {row['name']} (p={row['p_value']:.2e})")
                
        # 提取 Top 3 KEGG
        if has_kegg:
            top_kegg = df[df["source"] == "KEGG"].head(15)
            for _, row in top_kegg.iterrows():
                summary_lines.append(f"  - KEGG: {row['name']} (p={row['p_value']:.2e})")
                
        summary_lines.append("-" * 30)
        
    except Exception as e:
        pass

# 保存到文件
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"✅ 数据提取完毕！请打开文件查看并直接复制给我：\n{OUT_FILE}")