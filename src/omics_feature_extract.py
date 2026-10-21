import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

# ==========================
# 1. 配置参数
# ==========================
class Config:
    csv_file = "E:/学习/论文2/战桐整理/代码和结果/代码和结果/imputed_data_OS_filled.csv"
    output_dir = "E:/学习/论文2/战桐整理/代码和结果/代码和结果/vae_best_features_9468"
    
    input_dim = 1350   # 根据 CSV 的 cont+disc 列总数自动调整
    latent_dim = 128   # 隐空间维度
    epochs = 200
    batch_size = 64
    lr = 1e-4
    beta = 0.01        # KL 散度权重

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================
# 2. 掩码自编码器 (Masked VAE) 模型
# ==========================
class UnsupervisedVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# ==========================
# 3. 核心执行流程
# ==========================
def run_unsupervised_extraction():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # --- A. 数据加载 ---
    print("正在加载原始数据...")
    df = pd.read_csv(Config.csv_file)
    pid = df['slide_id'].str[:12]
    print("slides:", len(df), "| patients:", pid.nunique())
    cont = list(df.columns[13:1013]) + list(df.columns[1163:1363])
    disc = list(df.columns[1013:1163])
    
    X_raw = df[cont + disc].values.astype(float)
    raw_labels = (df['subtype'].str.strip() + df['grade'].astype(str).str.strip()).values
    
    # --- B. 构建缺失值掩码 (Mask) ---
    print("正在构建缺失值掩码...")
    mask = np.ones_like(X_raw)
    disc_start_idx = len(cont) 
    
    for i, col_name in enumerate(disc):
        idx = disc_start_idx + i
        col_name_lower = col_name.lower()
        if 'snv' in col_name_lower:
            mask[:, idx] = (X_raw[:, idx] != 2.0).astype(float)
        elif 'cnv' in col_name_lower:
            mask[:, idx] = (X_raw[:, idx] != 3.0).astype(float)
        else:
            mask[:, idx] = ((X_raw[:, idx] != 2.0) & (X_raw[:, idx] != 3.0)).astype(float)

    # --- C. 掩码感知标准化 ---
    print("正在进行掩码感知标准化...")
    X_scaled = np.zeros_like(X_raw)
    for j in range(X_raw.shape[1]):
        valid_indices = (mask[:, j] == 1)
        if valid_indices.sum() > 0:
            col_mean = X_raw[valid_indices, j].mean()
            col_std = X_raw[valid_indices, j].std()
            if col_std == 0: col_std = 1e-6
            X_scaled[valid_indices, j] = (X_raw[valid_indices, j] - col_mean) / col_std

    input_dim = X_scaled.shape[1]
    
    # --- D. 训练 Masked VAE ---
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    model = UnsupervisedVAE(input_dim, Config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    print(f"🚀 开始 Masked-VAE 训练 (输入维度 {input_dim} -> 隐空间 {Config.latent_dim})")
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x, m = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            
            # 仅计算非缺失部位的重构误差
            mse_elements = F.mse_loss(recon, x, reduction='none')
            recon_loss = torch.sum(mse_elements * m) / torch.clamp(torch.sum(m), min=1.0)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + Config.beta * kld_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Avg Loss: {total_loss/len(loader):.6f}")

    # --- E. 提取并保存特征 ---
    model.eval()
    print("\n正在提取 128 维组学特征并保存...")
    all_loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in all_loader:
            _, mu, _ = model(batch[0].to(device))
            embeddings.append(mu.cpu().numpy())
    embeddings = np.vstack(embeddings)

    try:
        split_df = pd.read_csv("E:/学习/论文2/战桐整理/代码和结果/代码和结果/slide_split_summary_6003.csv")
        slide_to_split = dict(zip(split_df['slide_id'], split_df['split']))
    except:
        slide_to_split = {sid: 'all' for sid in df['slide_id']}

    for i, slide_id in enumerate(df['slide_id']):
        split = slide_to_split.get(slide_id, 'others')
        save_p = os.path.join(Config.output_dir, split)
        os.makedirs(save_p, exist_ok=True)
        
        sid = slide_id[:-4] if slide_id.endswith('.svs') else slide_id
        with h5py.File(os.path.join(save_p, f"{sid}.h5"), 'w') as f:
            f.create_dataset('embedding', data=embeddings[i])
            f.create_dataset('label', data=str(raw_labels[i]))
    torch.save(model.state_dict(), os.path.join(Config.output_dir, "vae_model.pth"))

    df['pid'] = df['slide_id'].str[:12]
    df['split'] = df['slide_id'].map(slide_to_split)
    print(df.groupby('split').agg(slides=('slide_id', 'size'), patients=('pid', 'nunique')))
    print("跨划分的患者数:", (df.groupby('pid')['split'].nunique() > 1).sum())

    print(f"✅ 特征提取完成！已保存至: {Config.output_dir}")

if __name__ == "__main__":
    run_unsupervised_extraction()