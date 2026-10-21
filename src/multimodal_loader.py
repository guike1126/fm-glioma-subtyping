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

class MultimodalDataset(Dataset):
    def __init__(self, slide_ids, labels_df, img_dir, omics_dir, encoder, mode='train'):
        self.slide_ids = slide_ids
        self.img_dir = img_dir
        self.omics_dir = omics_dir
        self.mode = mode
        self.label_encoder = encoder
        
        if not self.slide_ids:
            print(f"❌ ERROR: {mode} list is EMPTY. Please check your .txt split files.")
            self.valid_samples = []
            return

        self.labels_df = labels_df[labels_df['slide_id'].isin(slide_ids)].copy()
        if 'label_encoded' not in self.labels_df.columns:
             self.labels_df['label_encoded'] = encoder.transform(self.labels_df['combined_label'])
        
        self.valid_samples = []
        self._init_indices()

    def _init_indices(self):
        # 1. 建立文件映射
        omics_map = {}
        if os.path.exists(self.omics_dir):
            files = glob.glob(os.path.join(self.omics_dir, "*.h5"))
            for f in files: omics_map[os.path.basename(f).split('.')[0]] = f
        
        img_map = {}
        files = glob.glob(os.path.join(self.img_dir, "**", "*.h5"), recursive=True)
        for f in files: img_map[os.path.basename(f).split('.')[0]] = f

        # 2. 匹配样本
        for sid in tqdm(self.slide_ids, desc=f"Scanning {self.mode}", leave=False):
            short_id = sid.split('.')[0]
            if short_id in img_map and short_id in omics_map:
                if sid in self.labels_df['slide_id'].values:
                    row = self.labels_df[self.labels_df['slide_id'] == sid].iloc[0]
                    self.valid_samples.append({
                        'slide_id': sid,
                        'img_path': img_map[short_id],
                        'omics_path': omics_map[short_id],
                        'label': row['label_encoded'],
                        'label_name': row['combined_label'], # 用于统计打印
                        'subtype': row['subtype'],
                        'grade': row['grade']
                    })

        # 3. 统计
        print(f"\n{'='*20} Dataset Info: {self.mode.upper()} {'='*20}")
        if len(self.valid_samples) == 0:
            print(f"  > ❌ No samples matched! Check paths.")
        else:
            # 自动检查第一张图的特征维度
            first_sample = self.valid_samples[0]
            try:
                with h5py.File(first_sample['img_path'], 'r') as f:
                    img_feat = None
                    for k in ['features', 'feats', 'resnet_features']:
                        if k in f.keys(): img_feat = f[k][:]; break
                    if img_feat is None:
                        for k in f.keys():
                            if len(f[k].shape) == 2 and f[k].shape[1] > 2:
                                img_feat = f[k][:]; break
                    
                    if img_feat is not None:
                        print(f"  > [Dim Check] Sample shape: {img_feat.shape}")
            except: pass

            label_counts = Counter([s['label_name'] for s in self.valid_samples])
            print(f"  > Total: {len(self.valid_samples)}")
            for name in sorted(label_counts.keys()):
                print(f"    - {name:<25}: {label_counts[name]}")
        print("="*60)

    def __len__(self): return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        try:
            with h5py.File(sample['omics_path'], 'r') as f:
                key = list(f.keys())[0]
                omics_feat = torch.from_numpy(f[key][:]).float().flatten()
        except: omics_feat = torch.zeros(64) 

        img_feat = None
        try:
            with h5py.File(sample['img_path'], 'r') as f:
                for k in ['features', 'feats', 'resnet_features']:
                    if k in f.keys(): img_feat = torch.from_numpy(f[k][:]).float(); break
                if img_feat is None:
                    for k in f.keys():
                        if len(f[k].shape) == 2 and f[k].shape[1] > 2:
                            img_feat = torch.from_numpy(f[k][:]).float(); break
        except: pass
        
        if img_feat is None: img_feat = torch.zeros(1, 512)
        return img_feat, omics_feat, torch.tensor(sample['label'], dtype=torch.long), sample['slide_id']

def collate_fn(batch):
    max_n = max([x[0].size(0) for x in batch])
    dim = batch[0][0].size(1)
    imgs, oms, lbls, sids = [], [], [], []
    for i, o, l, s in batch:
        if i.size(0) < max_n: i = torch.cat([i, torch.zeros(max_n - i.size(0), dim)], dim=0)
        imgs.append(i); oms.append(o); lbls.append(l); sids.append(s)
    return torch.stack(imgs), torch.stack(oms), torch.stack(lbls), sids

# 修正后的 5 参数版本
def get_stage_loader(dataset, allowed_subtypes=None, allowed_grades=None, batch_size=1, is_train=True):
    if not dataset or len(dataset) == 0: return None
    indices = []
    for i, s in enumerate(dataset.valid_samples):
        if allowed_subtypes is not None and s['subtype'] not in allowed_subtypes: continue
        if allowed_grades is not None:
            try: g = int(float(s['grade']))
            except: g = -1
            if g not in allowed_grades: continue
        indices.append(i)
    if not indices: return None
    
    subset = Subset(dataset, indices)
    sampler = None
    if is_train:
        labels = [dataset.valid_samples[i]['label'] for i in indices]
        counts = pd.Series(labels).value_counts().to_dict()
        weights = [1.0 / counts.get(l, 1) for l in labels]
        sampler = WeightedRandomSampler(weights, len(indices))
    
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, shuffle=(is_train and not sampler), 
                      collate_fn=collate_fn, num_workers=0, pin_memory=True)

def create_multimodal_datasets(csv, img_d, base_d, split_d, seed):
    om_root = os.path.join(base_d, f"vae_best_features_{seed}")
    df = pd.read_csv(csv)
    df['slide_id'] = df['slide_id'].apply(lambda x: x if x.endswith('.svs') else x+'.svs')
    df['combined_label'] = df['subtype'].str.strip() + '_' + df['grade'].astype(str)
    enc = LabelEncoder().fit(sorted(df['combined_label'].unique()))
    
    def ld(s):
        p = os.path.join(split_d, f"{s}_slide_{seed}.txt")
        if os.path.exists(p): return [l.strip() for l in open(p)]
        return []
    
    tr = MultimodalDataset(ld('train'), df, img_d, os.path.join(om_root, 'train'), enc, 'train')
    va = MultimodalDataset(ld('val'), df, img_d, os.path.join(om_root, 'val'), enc, 'val')
    te = MultimodalDataset(ld('test'), df, img_d, os.path.join(om_root, 'test'), enc, 'test')
    return tr, va, te, enc.classes_, 128