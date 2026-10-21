import pandas as pd
import numpy as np
import os
import sys

# ================= 配置路径 =================
root_dir = r'E:/学习/论文2/战桐整理/代码和结果/代码和结果/code1'
csv_path = os.path.join(root_dir, 'imputed_data_OS_filled.csv')
output_dir = os.path.join(root_dir, 'resplit_check')
os.makedirs(output_dir, exist_ok=True)

NEW_SEED = int(sys.argv[1])          # 命令行传入种子
SEED_SUFFIX = f"_{NEW_SEED}"

# ================= 1. 读取数据 =================
print(f"正在读取数据并进行分层划分 (种子: {NEW_SEED})...")
df = pd.read_csv(csv_path)

# 格式标准化
df['slide_id'] = df['slide_id'].astype(str)
df['patient_id'] = df['patient_id'].astype(str)
# 创建分层标签：亚型 + 等级
df['strata'] = df['subtype'] + '_G' + df['grade'].astype(str)

# ================= 2. 基于患者的划分逻辑 =================
# 提取唯一的患者及其对应的分层标签
patient_df = df.drop_duplicates(subset=['patient_id'])[['patient_id', 'strata']]

train_patients, val_patients, test_patients = [], [], []

# 按分层标签进行循环划分
for stratum, stratum_df in patient_df.groupby('strata'):
    patients = stratum_df['patient_id'].tolist()
    n = len(patients)
    
    # 组内随机打乱
    np.random.seed(NEW_SEED) 
    np.random.shuffle(patients)

    # 比例 7:2:1
    n_train = int(np.round(n * 0.7))
    n_val = int(np.round(n * 0.2))
    
    train_patients.extend(patients[:n_train])
    val_patients.extend(patients[n_train:n_train + n_val])
    test_patients.extend(patients[n_train + n_val:])

# 将划分好的患者 ID 映射回所有的 Slide ID
train_slides = df[df['patient_id'].isin(train_patients)]['slide_id'].tolist()
val_slides = df[df['patient_id'].isin(val_patients)]['slide_id'].tolist()
test_slides = df[df['patient_id'].isin(test_patients)]['slide_id'].tolist()

# ================= 3. 保存划分结果 =================
print(f"\n正在保存结果，文件名后缀为: {SEED_SUFFIX}")

def save_list(name, slides):
    path = os.path.join(output_dir, f"{name}{SEED_SUFFIX}.txt")
    with open(path, 'w') as f:
        for s in slides:
            f.write(s + '\n')
    return path

# 1. 保存 TXT 列表
save_list('train_slide', train_slides)
save_list('val_slide', val_slides)
save_list('test_slide', test_slides)

# 2. 保存 CSV 子集 (带元数据)
df_train = df[df['slide_id'].isin(train_slides)]
df_val = df[df['slide_id'].isin(val_slides)]
df_test = df[df['slide_id'].isin(test_slides)]

df_train.to_csv(os.path.join(output_dir, f'train_set_slide{SEED_SUFFIX}.csv'), index=False)
df_val.to_csv(os.path.join(output_dir, f'val_set_slide{SEED_SUFFIX}.csv'), index=False)
df_test.to_csv(os.path.join(output_dir, f'test_set_slide{SEED_SUFFIX}.csv'), index=False)

# 3. 保存全量汇总表
df['split'] = 'unknown'
df.loc[df['slide_id'].isin(train_slides), 'split'] = 'train'
df.loc[df['slide_id'].isin(val_slides), 'split'] = 'val'
df.loc[df['slide_id'].isin(test_slides), 'split'] = 'test'
df.to_csv(os.path.join(output_dir, f'slide_split_summary{SEED_SUFFIX}.csv'), index=False)

# ================= 4. 统计与验证 =================
print("-" * 30)
print(f"统计结果:")
print(f"训练集: {len(train_patients)} 患者 | {len(train_slides)} 切片")
print(f"验证集: {len(val_patients)} 患者 | {len(val_slides)} 切片")
print(f"测试集: {len(test_patients)} 患者 | {len(test_slides)} 切片")

# 泄露检查
overlap = set(train_patients) & set(val_patients) | set(train_patients) & set(test_patients) | set(val_patients) & set(test_patients)
if not overlap:
    print("✅ 验证通过：无患者重叠（Patient-level Independence maintained）。")
else:
    print(f"❌ 警告：检测到患者重叠！数量: {len(overlap)}")