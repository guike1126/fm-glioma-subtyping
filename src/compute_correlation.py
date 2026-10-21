import os
import pandas as pd
import numpy as np

def compute_cross_omics_correlation():
    # 配置你的绝对路径
    csv_file = "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv"
    results_root = "/seu_nvme/home/liuhongde/220232427/code1/combined_results_seeds_6003_4873_3157_1107_9468"

    print("⏳ 正在加载原始数据集，这可能需要一点时间...")
    df = pd.read_csv(csv_file)
    
    # 提取所有的列名，用于后续过滤（确保特征名在原始数据中完全匹配）
    all_columns = set(df.columns)

    print(f"\n🚀 开始计算 {results_root} 目录下的甲基化-表达谱相关性...\n")
    print("=" * 70)
    
    # 遍历任务目录
    for task in os.listdir(results_root):
        task_path = os.path.join(results_root, task)
        if not os.path.isdir(task_path): continue
        
        # 遍历类别目录
        for c_name in os.listdir(task_path):
            cls_path = os.path.join(task_path, c_name)
            if not os.path.isdir(cls_path): continue

            exp_file = os.path.join(cls_path, "top200_Expression_Clean.csv")
            methy_file = os.path.join(cls_path, "top100_Methylation_Clean.csv")

            if not (os.path.exists(exp_file) and os.path.exists(methy_file)):
                continue
            
            try:
                # 1. 读取 Top 特征
                exp_df = pd.read_csv(exp_file)
                methy_df = pd.read_csv(methy_file)

                # 提取基因名列表
                exp_genes = exp_df['Gene'].dropna().astype(str).tolist()
                methy_genes = methy_df['Site'].dropna().astype(str).tolist()

                # 确保特征在原始数据中存在
                valid_exp = [g for g in exp_genes if g in all_columns]
                valid_methy = [m for m in methy_genes if m in all_columns]

                if not valid_exp or not valid_methy:
                    print(f"⚠️ {task} | {c_name}: 找不到有效的特征列，跳过。")
                    continue

                # 2. 提取对应的原始数值数据 (跨越所有样本)
                data_exp = df[valid_exp].astype(float)
                data_methy = df[valid_methy].astype(float)

                # 3. 拼接并计算 Pearson 相关系数矩阵
                combined_data = pd.concat([data_methy, data_exp], axis=1)
                
                # 计算全矩阵后，只截取 甲基化行 x 表达谱列 的部分
                corr_matrix = combined_data.corr(method='pearson').loc[valid_methy, valid_exp]

                # 4. 将二维矩阵展平为一维配对表 (Source, Target, Weight)
                corr_pairs = corr_matrix.unstack().reset_index()
                corr_pairs.columns = ['Expression_Gene', 'Methylation_Site', 'Pearson_Correlation']

                # 过滤掉计算产生的 NaN 值，并按相关性升序排列 (最负的排在最前)
                corr_pairs = corr_pairs.dropna()
                corr_pairs = corr_pairs.sort_values(by='Pearson_Correlation', ascending=True)

                # 设定强负调控阈值 (r < -0.3 视为具有中等偏强以上的相关性)
                strong_negative = corr_pairs[corr_pairs['Pearson_Correlation'] < -0.3]
                
                # 5. 终端结果打印
                print(f"📂 任务: {task} | 类别: {c_name}")
                print(f"  ✅ 提取到 {len(valid_methy)} 个甲基化位点 与 {len(valid_exp)} 个表达基因")
                print(f"  📉 发现 r < -0.3 的强负相关调控对: {len(strong_negative)} 个")
                
                if len(strong_negative) > 0:
                    print("  🌟 Top 3 最强表观沉默对:")
                    for _, row in strong_negative.head(3).iterrows():
                        print(f"     -> {row['Methylation_Site']} (甲基化) 抑制 {row['Expression_Gene']} (表达): r = {row['Pearson_Correlation']:.4f}")
                print("-" * 70)

                # 6. 结果持久化保存
                # 完整 20,000 对关系保存
                out_file_all = os.path.join(cls_path, "Methy_Exp_Correlation_All.csv")
                corr_pairs.to_csv(out_file_all, index=False)
                
                # 只保存强负相关关系 (重点关注对象)
                if len(strong_negative) > 0:
                    out_file_sig = os.path.join(cls_path, "Methy_Exp_Negative_Regulation_Strong.csv")
                    strong_negative.to_csv(out_file_sig, index=False)

            except Exception as e:
                print(f"❌ 处理 {task}/{c_name} 时报错: {e}")

    print("\n✅ 所有任务的【表观-转录】调控相关性计算与导出圆满完成！")

if __name__ == "__main__":
    compute_cross_omics_correlation()