import os
import pandas as pd

def check_omics_intersections():
    # 更新为你指定的绝对路径
    output_root = "/seu_nvme/home/liuhongde/220232427/code1/combined_results_seeds_6003_4873_3157_1107_9468"
    
    if not os.path.exists(output_root):
        print(f"❌ 找不到目录 {output_root}，请检查路径是否正确。")
        return

    summary_results = []

    print(f"🔍 开始扫描 {output_root} 目录下的组学交集...\n")
    print("=" * 60)

    # 遍历所有任务 (Subtype, Astro_Grade, Oligo_Grade, Final_6Class)
    for task in os.listdir(output_root):
        task_path = os.path.join(output_root, task)
        # 确保是文件夹才进入
        if not os.path.isdir(task_path):
            continue

        # 遍历任务下的所有类别 (Astrocytoma, GBM, Astro_G2 等)
        for c_name in os.listdir(task_path):
            cls_path = os.path.join(task_path, c_name)
            if not os.path.isdir(cls_path):
                continue

            # 确定文件名 (基于你之前的代码逻辑：表达谱Top200，突变Top60，甲基化Top100)
            exp_file = os.path.join(cls_path, "top200_Expression_Clean.csv")
            mut_file = os.path.join(cls_path, "top60_Mutations_Clean.csv")
            methy_file = os.path.join(cls_path, "top100_Methylation_Clean.csv")

            # 检查这三个文件是否都存在
            if not (os.path.exists(exp_file) and os.path.exists(mut_file) and os.path.exists(methy_file)):
                print(f"⚠️ 在 {task}/{c_name} 下文件不全，跳过...")
                continue

            try:
                # 读取数据
                exp_df = pd.read_csv(exp_file)
                mut_df = pd.read_csv(mut_file)
                methy_df = pd.read_csv(methy_file)

                # 提取基因名称，转为字符串集合并剔除空值
                exp_genes = set(exp_df['Gene'].dropna().astype(str))
                mut_genes = set(mut_df['Gene'].dropna().astype(str))
                # 你的代码中甲基化的列名为 'Site'
                methy_genes = set(methy_df['Site'].dropna().astype(str))

                # 计算交集
                methy_exp_inter = methy_genes.intersection(exp_genes)
                methy_mut_inter = methy_genes.intersection(mut_genes)

                # 打印终端结果
                print(f"📂 任务: {task} | 类别: {c_name}")
                print(f"  🧬 甲基化(Top100) & 表达谱(Top200) 交集 ({len(methy_exp_inter)}个):")
                print(f"     -> {', '.join(methy_exp_inter) if methy_exp_inter else '无'}")
                print(f"  🧬 甲基化(Top100) & 突变(Top60) 交集 ({len(methy_mut_inter)}个):")
                print(f"     -> {', '.join(methy_mut_inter) if methy_mut_inter else '无'}")
                print("-" * 60)

                # 记录到汇总字典
                summary_results.append({
                    'Task': task,
                    'Class': c_name,
                    'Methy_Exp_Overlap_Count': len(methy_exp_inter),
                    'Methy_Exp_Genes': ', '.join(methy_exp_inter) if methy_exp_inter else 'None',
                    'Methy_Mut_Overlap_Count': len(methy_mut_inter),
                    'Methy_Mut_Genes': ', '.join(methy_mut_inter) if methy_mut_inter else 'None'
                })

            except Exception as e:
                print(f"❌ 读取 {task}/{c_name} 下的 CSV 时发生错误: {e}")

    # 将所有结果导出为一个汇总 CSV 文件
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(output_root, "Cross_Omics_Intersection_Summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ 扫描完成！汇总表已保存至:\n{summary_path}")
    else:
        print("\n⚠️ 未找到任何有效的结果供汇总，请检查文件夹结构。")

if __name__ == "__main__":
    check_omics_intersections()