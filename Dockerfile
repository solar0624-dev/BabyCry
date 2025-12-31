# 使用轻量级的 Python 3.9 基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 关键步骤：安装系统级依赖 (libsndfile 用于音频处理)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源加速安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制当前目录下所有文件到容器中
COPY . .

# 暴露端口
EXPOSE 80

# 启动命令 (使用 gunicorn 生产级服务器)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:80", "app:app"]
