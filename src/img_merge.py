import os
import glob
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "enrichment_results")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 按照模态分类
modalities = ["Expression", "SNV", "CNV", "Methylation"]

for mod in modalities:
    # 找到该模态下的所有 png 图片
    png_files = glob.glob(os.path.join(INPUT_DIR, f"*_{mod}_*.png"))
    # 排序保证顺序一致
    png_files = sorted(png_files)
    
    if not png_files:
        print(f"⚠️ 没有找到 {mod} 的图片。")
        continue
        
    print(f"正在合并 {mod} 的 {len(png_files)} 张图片...")
    
    images = []
    for f in png_files:
        img = Image.open(f)
        # 转换为 RGB（防止 RGBA 格式导致 PDF 报错）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
        
    # 保存为 PDF
    pdf_path = os.path.join(OUT_DIR, f"Supplementary_Figure_{mod}_Bubble_Plots.pdf")
    images[0].save(pdf_path, save_all=True, append_images=images[1:])
    print(f"✅ 已生成: {pdf_path}")

print("🎉 全部 PDF 生成完毕！")