# 多阶段构建 - 前端构建阶段
FROM node:18-alpine AS frontend-builder

WORKDIR /app

# 复制前端依赖文件
COPY package*.json ./
COPY tsconfig*.json ./
COPY vite.config.ts ./
COPY tailwind.config.js ./
COPY postcss.config.js ./

# 安装前端依赖
RUN npm install

# 复制前端源码
COPY src ./src
COPY public ./public

# 构建前端
RUN npm run build

# Python运行环境阶段
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制Python依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制后端代码
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY migrate.py .

# 复制启动脚本
COPY run.py .

# 复制前端构建产物到静态文件目录
COPY --from=frontend-builder /app/dist ./static

# 创建数据目录
RUN mkdir -p /app/data

# 创建非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# 启动命令
# 启动命令
CMD ["python", "run.py"]
