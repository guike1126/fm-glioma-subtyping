import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re

# 请确保以下自定义模块在当前目录下
from train_multimodel import DeepMultimodalModel
from omics_feature_extract import UnsupervisedVAE
from multimodal_loader import create_multimodal_datasets, get_stage_loader

def run_global_multi_seed_attribution(seeds=[6003, 4842, 9468]):
    """
    执行多周知全局归因分析：
    1. 遍历指定的种子列表
    2. 对每个种子加载对应的模型权重和测试集
    3. 仅对预测正确的样本进行归因计算
    4. 跨种子累加重要性得分并取平均值
    5. 按指定数量导出 Top 特征（已剔除 .x/.y 后缀）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 配置路径 ---
    csv_file = "/seu_nvme/home/liuhongde/220232427/imputed_data_OS_filled.csv"
    img_dir = "/seu_nvme/home/liuhongde/220232427/features_h5"
    base_dir = "/seu_nvme/home/liuhongde/220232427"
    split_dir = "/seu_nvme/home/liuhongde/220232427"
    
    # 结果输出根目录
    output_root = f"combined_results_seeds_{'_'.join(map(str, seeds))}"
    os.makedirs(output_root, exist_ok=True)

    # --- 1. 数据预处理 (通用) ---
    df = pd.read_csv(csv_file)
    df['slide_id'] = df['slide_id'].apply(lambda x: x if x.endswith('.svs') else x+'.svs')
    
    # 定义特征列索引范围
    gene_exp_cols = list(df.columns[13:1013])   
    snv_cols = list(df.columns[1013:1063])      
    cnv_cols = list(df.columns[1063:1163])      
    methy_cols = list(df.columns[1163:1363])    
    
    all_feature_cols = gene_exp_cols + methy_cols + snv_cols + cnv_cols
    X_raw_data = df[all_feature_cols].values 
    X_scaled = StandardScaler().fit_transform(X_raw_data.astype(float))

    analysis_tasks = {'Subtype': {}, 'Astro_Grade': {}, 'Oligo_Grade': {}, 'Final_6Class': {}}
    global_classes = None 

    # --- 2. 开始种子循环 ---
    for seed in seeds:
        print(f"\n🚀 正在处理种子 Seed: {seed} ...")
        vae_weight_path = f"/seu_nvme/home/liuhongde/220232427/vae_best_features_{seed}/vae_model.pth"
        mm_weight_path = f"./outputs_multimodel/seed_{seed}/best_final.pth" 

        if not os.path.exists(mm_weight_path):
            print(f"⚠️ 跳过：未找到种子 {seed} 的权重文件")
            continue

        vae = UnsupervisedVAE(input_dim=1350, latent_dim=128).to(device)
        vae.load_state_dict(torch.load(vae_weight_path, weights_only=True, map_location=device))
        vae.eval()

        mm_model = DeepMultimodalModel(omics_input_dim=128).to(device)
        mm_model.load_state_dict(torch.load(mm_weight_path, weights_only=True, map_location=device))
        mm_model.eval()

        _, _, test_ds, classes, _ = create_multimodal_datasets(csv_file, img_dir, base_dir, split_dir, seed)
        test_loader = get_stage_loader(test_ds, batch_size=1, is_train=False)
        if global_classes is None: global_classes = classes

        sub_m, astro_m, oligo_m = [], [], []
        for name in classes:
            n = name.lower()
            s = 0 if 'astro' in n else 1 if 'gbm' in n else 2
            sub_m.append(s)
            g_match = re.search(r'g(\d)', n)
            g = int(g_match.group(1)) if g_match else -1
            if s == 0: astro_m.append(0 if g==2 else 1 if g==3 else 2)
            else: astro_m.append(-1)
            if s == 2: oligo_m.append(0 if g==2 else 1)
            else: oligo_m.append(-1)
        
        map_sub = torch.tensor(sub_m).to(device)
        map_astro = torch.tensor(astro_m).to(device)
        map_oligo = torch.tensor(oligo_m).to(device)

        for img, _, lbl_idx, sids in tqdm(test_loader, desc=f"Seed {seed} Processing"):
            img = img.to(device)
            slide_id = sids[0]
            true_final = lbl_idx[0].item()
            if slide_id not in df['slide_id'].values: continue
            row_idx = df.index[df['slide_id'] == slide_id].tolist()[0]
            
            raw_sample_values = X_raw_data[row_idx]
            raw_tensor = torch.tensor(X_scaled[row_idx:row_idx+1], dtype=torch.float32, device=device).requires_grad_(True)
            
            h = vae.encoder(raw_tensor)
            mu = vae.fc_mu(h)
            out = mm_model(img, mu)['logits']
            
            def process_and_accumulate(logits_key, true_label, task_key):
                if true_label == -1: return
                if true_label not in analysis_tasks[task_key]:
                    analysis_tasks[task_key][true_label] = [np.zeros(1350), 0]
                
                logits = out[logits_key]
                if logits.argmax(dim=1)[0].item() == true_label:
                    mm_model.zero_grad(); vae.zero_grad()
                    logits[0, true_label].backward(retain_graph=True)
                    grad = raw_tensor.grad[0].cpu().numpy()
                    val = raw_tensor[0].detach().cpu().numpy()
                    importance = np.abs(grad * val)
                    for j in range(1200, 1250): # SNV
                        if raw_sample_values[j] == 2: importance[j] = 0
                    for j in range(1250, 1350): # CNV
                        if raw_sample_values[j] == 3: importance[j] = 0
                    analysis_tasks[task_key][true_label][0] += importance
                    analysis_tasks[task_key][true_label][1] += 1
                    raw_tensor.grad.zero_()

            process_and_accumulate('subtype_logits', map_sub[true_final].item(), 'Subtype')
            process_and_accumulate('astro_grade_logits', map_astro[true_final].item(), 'Astro_Grade')
            process_and_accumulate('oligo_grade_logits', map_oligo[true_final].item(), 'Oligo_Grade')
            process_and_accumulate('final_logits', true_final, 'Final_6Class')
        
        del mm_model, vae, test_loader
        torch.cuda.empty_cache()

    # --- 3. 跨种子平均与导出 (已增加去后缀逻辑) ---
    print("\n📊 正在计算多种子平均重要性并执行清洗导出...")
    
    # 辅助函数：清洗基因名后缀
    def clean_name(name_list):
        # 使用正则匹配结尾的 .x 或 .y 并替换为空
        return [re.sub(r'\.[xy]$', '', str(x)) for x in name_list]

    N_EXP, N_SNV, N_CNV, N_METHY = 200, 30, 30, 100
    task_map = {
        'Subtype': ['Astrocytoma', 'GBM', 'Oligodendroglioma'],
        'Astro_Grade': ['Astro_G2', 'Astro_G3', 'Astro_G4'],
        'Oligo_Grade': ['Oligo_G2', 'Oligo_G3'],
        'Final_6Class': global_classes
    }

    for task, data in analysis_tasks.items():
        for c_idx, (total_imp, count) in data.items():
            if count == 0: continue
            avg_imp = total_imp / count
            c_name = task_map[task][c_idx].replace('/', '_')
            save_path = os.path.join(output_root, task, c_name)
            os.makedirs(save_path, exist_ok=True)

            # A. 导出表达谱 (去后缀)
            exp_df = pd.DataFrame({'Gene': clean_name(gene_exp_cols), 'Importance': avg_imp[0:1000]})
            exp_df.sort_values(by='Importance', ascending=False).head(N_EXP).to_csv(
                os.path.join(save_path, f"top{N_EXP}_Expression_Clean.csv"), index=False)
            
            # B. 导出突变 (去后缀)
            snv_df = pd.DataFrame({'Gene': clean_name(snv_cols), 'Importance': avg_imp[1200:1250], 'Type': 'SNV'})
            cnv_df = pd.DataFrame({'Gene': clean_name(cnv_cols), 'Importance': avg_imp[1250:1350], 'Type': 'CNV'})
            top_snv = snv_df.sort_values(by='Importance', ascending=False).head(N_SNV)
            top_cnv = cnv_df.sort_values(by='Importance', ascending=False).head(N_CNV)
            pd.concat([top_snv, top_cnv]).to_csv(
                os.path.join(save_path, f"top{N_SNV+N_CNV}_Mutations_Clean.csv"), index=False)
            
            # C. 导出甲基化 (去后缀)
            methy_df = pd.DataFrame({'Site': clean_name(methy_cols), 'Importance': avg_imp[1000:1200]})
            methy_df.sort_values(by='Importance', ascending=False).head(N_METHY).to_csv(
                os.path.join(save_path, f"top{N_METHY}_Methylation_Clean.csv"), index=False)

    print(f"\n✅ 归因清洗任务圆满完成！结果保存在: {output_root}")

if __name__ == "__main__":
    run_global_multi_seed_attribution(seeds=[6003, 4873, 3157, 1107, 9468])