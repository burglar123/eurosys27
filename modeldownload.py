from modelscope import snapshot_download
import os

# 定义在 webdav 上的存放路径
model_base_path = '/dev/shm'

# 如果文件夹不存在就创建它
if not os.path.exists(model_base_path):
    os.makedirs(model_base_path)

print("正在下载 OPT-30B 到 /webdav...")
target_dir = snapshot_download('facebook/opt-30b', cache_dir=model_base_path)

print("正在下载 OPT-1.3B 到 /webdav...")
draft_dir = snapshot_download('facebook/opt-1.3b', cache_dir=model_base_path)

print(f"下载完成！")
print(f"30B 路径: {target_dir}")
print(f"1.3B 路径: {draft_dir}")