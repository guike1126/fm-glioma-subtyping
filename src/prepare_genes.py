import os
import pandas as pd
import glob

# ================= 核心修改区 =================
# 自动获取当前脚本所在的文件夹绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 你的结果文件夹名字（如果你的名字不一样，请改这里）
ROOT_DIR = os.path.join(SCRIPT_DIR, "combined_results_seeds_6003_4873_3157_1107_9468")

# 输出文件夹
OUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_inputs")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"--- 正在检查路径 ---")
print(f"脚本所在目录: {SCRIPT_DIR}")
print(f"期望的数据目录: {ROOT_DIR}")

if not os.path.exists(ROOT_DIR):
    print(f"❌ 错误：找不到数据文件夹！请检查文件夹名字是否为 'combined_results_seeds_6003_4873_3157_1107_9468'，且它是否和此脚本在同一目录下。")
    exit()

print(f"✅ 成功找到数据目录！\n--- 开始遍历 ---")
# ==============================================

# 使用 os.walk 遍历（比 glob 更稳定）
csv_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print(f"共找到 {len(csv_files)} 个 CSV 文件。")

success_count = 0

for csv_file in csv_files:
    # 提取文件名和父目录名
    filename = os.path.basename(csv_file)
    
    # 解析出 task 和 category (例如: Subtype/Astrocytoma)
    # 假设目录结构是: ROOT_DIR/Subtype/Astrocytoma/file.csv
    parent_dir = os.path.dirname(csv_file)
    category = os.path.basename(parent_dir)
    task = os.path.basename(os.path.dirname(parent_dir))
    
    prefix = f"{task}_{category}"
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"读取文件失败: {csv_file}, 错误: {e}")
        continue
    
    # 1. 处理 Expression
    if "Expression" in filename and "Gene" in df.columns:
        genes = df["Gene"].dropna().unique().tolist()
        out_path = os.path.join(OUT_DIR, f"{prefix}_Expression.txt")
        with open(out_path, "w") as f:
            for g in genes: 
                f.write(f"{g}\n")
        success_count += 1
        print(f"已生成表达谱: {prefix}_Expression.txt (包含 {len(genes)} 个基因)")
            
    # 2. 处理 Mutations
    elif "Mutations" in filename and "Gene" in df.columns and "Type" in df.columns:
        for mut_type in ["SNV", "CNV"]:
            sub_df = df[df["Type"] == mut_type]
            if not sub_df.empty:
                genes = sub_df["Gene"].dropna().unique().tolist()
                out_path = os.path.join(OUT_DIR, f"{prefix}_{mut_type}.txt")
                with open(out_path, "w") as f:
                    for g in genes: 
                        f.write(f"{g}\n")
                success_count += 1
                print(f"已生成突变: {prefix}_{mut_type}.txt (包含 {len(genes)} 个基因)")
                    
    # 3. 处理 Methylation
    elif "Methylation" in filename:
        # 通常甲基化文件的列名是 Site 或者 Gene，这里我们做个兼容
        col_name = "Site" if "Site" in df.columns else ("Gene" if "Gene" in df.columns else None)
        if col_name:
            genes = df[col_name].dropna().unique().tolist()
            out_path = os.path.join(OUT_DIR, f"{prefix}_Methylation.txt")
            with open(out_path, "w") as f:
                for g in genes: 
                    f.write(f"{g}\n")
            success_count += 1
            print(f"已生成甲基化: {prefix}_Methylation.txt (包含 {len(genes)} 个基因)")

print(f"\n✅ 任务完成！共生成 {success_count} 个 txt 文件，请去 {OUT_DIR} 查看。")