#!/usr/bin/env bash

# 获取当前日期和时间，格式为 YYYY-MM-DD_HH-MM-SS
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 定义日志文件的名称，包括时间戳
log_file="embedding_server_$current_time.log"

# 启动 Python 脚本，并将输出重定向到日志文件和标准输出
# CUDA_VISIBLE_DEVICES=0 python embedding_server.py --models_dir_path embedding_models/ --use_gpu
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python embedding_server.py --models_dir_path embedding_models/ --use_gpu' > "$log_file" 2>&1 &
