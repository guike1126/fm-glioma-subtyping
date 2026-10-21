import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import pandas as pd
import numpy as np
import os
import h5py
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import glob
from collections import Counter
import gc

class ImageOnlyDataset(Dataset):
    def __init__(self, slide_ids, labels_df, img_dir, encoder, mode='train'):
        self.slide_ids = slide_ids
        self.img_dir = img_dir
        self.mode = mode
        self.label_encoder = encoder
        
        # --- 核心优化：数据缓存字典 ---
        self.data_cache = {} 
        
        if not self.slide_ids:
            print(f"❌ [Warning] {mode} list is EMPTY. Please check your .txt split files.")
            self.valid_samples = []
            return

        # 筛选数据
        self.labels_df = labels_df[labels_df['slide_id'].isin(slide_ids)].copy()
        if 'label_encoded' not in self.labels_df.columns:
             self.labels_df['label_encoded'] = encoder.transform(self.labels_df['combined_label'])
        
        self.valid_samples = []
        # 初始化并执行预加载
        self._preload_data()

    def _preload_data(self):
        # 1. 建立图像文件映射 (文件名 -> 完整路径)
        img_map = {}
        # 递归查找所有 .h5 文件
        print(f"[{self.mode.upper()}] Scanning H5 files in directory...")
        files = glob.glob(os.path.join(self.img_dir, "**", "*.h5"), recursive=True)
        for f in files: 
            img_map[os.path.basename(f).split('.')[0]] = f

        print(f"[{self.mode.upper()}] Pre-loading features into RAM (Optimized for 512G server)...")
        
        # 2. 匹配样本并加载到内存
        success_count = 0
        missing_files = []
        
        # 使用 tqdm 显示进度条
        for sid in tqdm(self.slide_ids, desc=f"Loading {self.mode} features"):
            short_id = sid.split('.')[0]
            
            # 检查是否有对应的 H5 文件
            if short_id not in img_map:
                missing_files.append(sid)
                continue
                
            # 检查是否有对应的标签
            if sid not in self.labels_df['slide_id'].values:
                continue

            img_path = img_map[short_id]
            
            # --- 读取 H5 并常驻内存 ---
            try:
                with h5py.File(img_path, 'r') as f:
                    feat = None
                    # 智能查找特征键
                    keys_to_check = ['features', 'feats', 'resnet_features']
                    for k in keys_to_check:
                        if k in f.keys(): 
                            feat = f[k][:]
                            break
                    
                    # 如果没找到常用键，尝试通过维度查找 (排除坐标)
                    if feat is None:
                        for k in f.keys():
                            if len(f[k].shape) == 2 and f[k].shape[1] > 2:
                                feat = f[k][:]
                                break
                    
                    if feat is not None:
                        # 转为 Tensor 并存入字典
                        self.data_cache[sid] = torch.from_numpy(feat).float()
                        
                        # 记录样本信息
                        row = self.labels_df[self.labels_df['slide_id'] == sid].iloc[0]
                        self.valid_samples.append({
                            'slide_id': sid,
                            'label': row['label_encoded'],
                            'label_name': row['combined_label'], 
                            'subtype': row['subtype'],
                            'grade': row['grade']
                        })
                        success_count += 1
            except Exception as e:
                print(f"Error reading {sid}: {e}")

        # 3. 打印详细统计信息
        print(f"\n{'='*20} Dataset Statistics: {self.mode.upper()} {'='*20}")
        print(f"  > Total Slides listed: {len(self.slide_ids)}")
        print(f"  > Successfully Loaded: {len(self.valid_samples)}")
        if missing_files:
            print(f"  > ⚠️ Missing H5 files: {len(missing_files)} (e.g., {missing_files[0]})")
        
        if len(self.valid_samples) > 0:
            # 维度检查
            first_sid = self.valid_samples[0]['slide_id']
            feat_shape = self.data_cache[first_sid].shape
            print(f"  > [Feature Check] Shape: {feat_shape} (Patches x Dim)")
            
            # 类别统计
            label_counts = Counter([s['label_name'] for s in self.valid_samples])
            print(f"  > Class Distribution:")
            for name in sorted(label_counts.keys()):
                print(f"    • {name:<25}: {label_counts[name]} slides")
        print("="*60 + "\n")

    def __len__(self): 
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        sid = sample['slide_id']
        
        # 直接从内存获取，速度极快
        img_feat = self.data_cache.get(sid)
        
        # 兜底防止意外 None
        if img_feat is None:
            img_feat = torch.zeros(1, 512)
            
        return img_feat, torch.tensor(sample['label'], dtype=torch.long), sid

def collate_fn(batch):
    # batch: list of (img_feat, label, slide_id)
    # 动态 Padding：找出当前 Batch 中 Patch 数量最大值
    max_n = max([x[0].size(0) for x in batch])
    dim = batch[0][0].size(1)
    
    imgs, lbls, sids = [], [], []
    for feat, label, sid in batch:
        n = feat.size(0)
        if n < max_n: 
            # 补零
            padding = torch.zeros(max_n - n, dim)
            feat = torch.cat([feat, padding], dim=0)
        imgs.append(feat)
        lbls.append(label)
        sids.append(sid)
        
    return torch.stack(imgs), torch.stack(lbls), sids

def get_stage_loader(dataset, allowed_subtypes=None, allowed_grades=None, batch_size=32, is_train=True):
    """
    batch_size: 默认设为 32，利用 V100 显存
    """
    if not dataset or len(dataset) == 0: return None
    
    # 根据阶段任务筛选样本索引
    indices = []
    for i, s in enumerate(dataset.valid_samples):
        # 筛选亚型
        if allowed_subtypes is not None and s['subtype'] not in allowed_subtypes: 
            continue
        # 筛选分级
        if allowed_grades is not None:
            try: g = int(float(s['grade']))
            except: g = -1
            if g not in allowed_grades: 
                continue
        indices.append(i)

    if not indices: return None
    
    subset = Subset(dataset, indices)
    sampler = None
    
    # 训练集使用加权采样解决不平衡
    if is_train:
        labels = [dataset.valid_samples[i]['label'] for i in indices]
        counts = pd.Series(labels).value_counts().to_dict()
        weights = [1.0 / counts.get(l, 1) for l in labels]
        sampler = WeightedRandomSampler(weights, len(indices))
    
    # num_workers=2 即可，因为数据已在内存，不需要太高 worker
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, 
                      shuffle=(is_train and not sampler), 
                      collate_fn=collate_fn, num_workers=2, pin_memory=True)

def create_datasets(csv, img_d, base_d, split_d, seed):
    print(f"\n🚀 Initializing Datasets for SEED: {seed}")
    df = pd.read_csv(csv)
    # 确保 ID 格式一致
    df['slide_id'] = df['slide_id'].apply(lambda x: x if x.endswith('.svs') else x+'.svs')
    # 组合标签
    df['combined_label'] = df['subtype'].str.strip() + '_' + df['grade'].astype(str)
    
    # 拟合编码器
    enc = LabelEncoder().fit(sorted(df['combined_label'].unique()))
    print(f"ℹ️  Labels encoded: {enc.classes_}")
    
    def load_split_file(split_name):
        path = os.path.join(split_d, f"{split_name}_slide_{seed}.txt")
        if os.path.exists(path): 
            return [l.strip() for l in open(path) if l.strip()]
        print(f"❌ [Loader] File not found: {path}")
        return []
    
    train_ids = load_split_file('train')
    val_ids = load_split_file('val')
    test_ids = load_split_file('test')
    
    tr = ImageOnlyDataset(train_ids, df, img_d, enc, 'train')
    va = ImageOnlyDataset(val_ids, df, img_d, enc, 'val')
    te = ImageOnlyDataset(test_ids, df, img_d, enc, 'test')
    
    return tr, va, te, enc.classes_