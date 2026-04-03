# download_model.py
from huggingface_hub import snapshot_download
import os

# 设置下载目录
local_dir = "./models/audioldm-s-full-v2"
os.makedirs(local_dir, exist_ok=True)

print(f"正在下载模型到 {local_dir} ...")

snapshot_download(
    repo_id="cvssp/audioldm-s-full-v2",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 确保下载的是实体文件
    # 如果您在国内，下载慢，可以尝试设置镜像，或者使用梯子
    # endpoint="https://hf-mirror.com" 
)

print("下载完成！")