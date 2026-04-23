from modelscope import snapshot_download

# 指定一个硬盘路径，比如 /root/data/opt-30b
model_dir = snapshot_download(
    'facebook/opt-1.3b', 
    cache_dir='/dev/shm', # 换成你的硬盘路径
    ignore_file_pattern=['*.msgpack', '*.h5', '*.ot'] # 过滤掉 Flax, TF 和 Rust 模型文件
)