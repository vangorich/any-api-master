# Any API

<div align="center">
  <h3>🚀 终极 Gemini API 代理解决方案</h3>
  <p>兼容 OpenAI 格式，配备预设管理、正则处理和密钥管理等高级功能</p>
  <p>
    <a href="#-快速开始">快速开始</a> •
    <a href="#-docker-部署推荐">Docker部署</a> •
    <a href="#-功能特性">功能特性</a> •
    <a href="#-数据库迁移">数据库迁移</a>
  </p>
</div>

## ✨ 功能特性

- 🔄 **OpenAI 格式兼容** - 无缝兼容 OpenAI API 格式，轻松迁移现有应用
- 🎨 **智能预设管理** - 动态注入系统提示词，支持变量替换 ({{roll}}, {{random}})
- 🔧 **正则表达式处理** - 请求前/响应后的高级文本处理规则
- 🔐 **密钥管理系统** - 支持官方密钥和专属密钥，自动轮换和状态监控
- 📊 **实时日志监控** - 详细的请求日志，包含延迟、令牌使用等统计
- 👥 **多用户支持** - 完整的用户认证和权限管理系统
- 🎯 **流式响应支持** - 完整支持 SSE 流式输出
- 🌐 **现代化 Web 界面** - React + TypeScript 构建的精美管理后台
- 🐳 **Docker 部署** - 一键部署，开箱即用

## 🚀 快速开始

### 方式一：🐳 Docker 部署（推荐）

最简单快捷的部署方式，无需配置 Python 和 Node.js 环境。

```bash
# 1. 克隆项目
git clone https://github.com/foamcold/any-api.git
cd any-api

# 2. 配置环境变量（可选）
cp .env.example .env
# 编辑 .env 文件，至少修改 SECRET_KEY

# 3. 启动服务
docker-compose up -d

# 4. 查看日志
docker-compose logs -f

# 5. 访问应用
# 前端: http://localhost:8000
# API文档: http://localhost:8000/docs
```

**停止服务：**
```bash
docker-compose down
```

**更新应用：**
```bash
git pull
docker-compose down
docker-compose build
docker-compose up -d
```

### 方式二：传统部署

#### 环境要求

- Python 3.10+
- Node.js 18+
- SQLite (默认) 或其他数据库

#### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/foamcold/gproxy.git
cd gproxy
```

2. **后端设置**

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件配置你的设置
```

3. **前端设置**

```bash
npm install
npm run build  # 生产环境构建
# 或 npm run dev  # 开发环境
```

4. **启动应用**

```bash
# 启动后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 开发模式（前端热重载）
# npm run dev
```

5. **访问应用**

- 前端界面: `http://localhost:8000`（生产构建）或 `http://localhost:5173`（开发模式）
- API 端点: `http://localhost:8000/v1/chat/completions`
- API 文档: `http://localhost:8000/docs`

## 📦 数据库迁移

本项目使用 Alembic 进行数据库版本管理。

### 手动迁移

数据库的结构变更需要通过迁移命令手动执行。

首次运行或更新版本后，请务必执行数据库迁移：

```bash
# 升级到最新版本
python migrate.py upgrade

# 查看当前版本
python migrate.py current

# 查看迁移历史
python migrate.py history

# 创建新迁移（开发时）
python migrate.py revision "描述"
```

**Docker 环境下执行迁移：**

```bash
# 进入容器
docker-compose exec app sh

# 执行迁移命令
python migrate.py upgrade
```

更多详情请查看 [数据库迁移文档](docs/database_migration.md)。

## 📖 使用说明

### 基本使用

与 OpenAI API 完全兼容，只需替换 base_url 和 API 密钥:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-exclusive-key"  # 在管理后台生成
)

response = client.chat.completions.create(
    model="gemini-1.5-flash",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.choices[0].message.content)
```

### 预设管理

在管理后台创建预设以自动注入系统提示词:

```json
[
  {
    "role": "system",
    "content": "你是一个专业的助手。今天的日期是 {{date}}。"
  }
]
```

支持的变量:
- `{{date}}` - 当前日期
- `{{time}}` - 当前时间
- `{{random}}` - 随机数
- `{{roll:<sides>}}` - 掷骰子 (例: {{roll:6}})

### 正则规则

创建正则规则进行文本处理:

- **预处理** (请求) - 在发送到 Gemini 前处理用户输入
- **后处理** (响应) - 在返回给客户端前处理 AI 响应

示例: 过滤敏感词

```
模式: \b(敏感词1|敏感词2)\b
替换: ***
```

### 密钥管理

- **专属密钥**: 为用户生成的访问密钥，用于身份验证
- **官方密钥**: Gemini API 密钥，用于实际调用 API

系统自动在多个官方密钥间轮换，确保高可用性。

## 🛠️ 配置

### 环境变量

主要配置项可通过 `.env` 文件设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/sql_app.db` | 数据库连接URL |
| `SECRET_KEY` | - | JWT密钥（生产环境必须修改！） |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `43200` | Token过期时间（30天） |
| `VITE_API_STR` | `/api` | API基础路径 |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | Gemini API地址 |

完整示例请查看 `.env.example` 文件。

### Docker 配置

编辑 `docker-compose.yml` 来自定义配置：

```yaml
environment:
  - SECRET_KEY=your-secret-key-here
  # 其他配置...
```

或使用 `.env` 文件（推荐）：

```bash
# .env
SECRET_KEY=your-very-secure-secret-key-change-it
```

## 🏗️ 项目结构

```
any-api/
├── app/                    # 后端应用
│   ├── api/               # API 路由
│   │   └── endpoints/     # 端点处理器
│   ├── core/              # 核心配置
│   ├── models/            # 数据库模型
│   ├── schemas/           # Pydantic schemas
│   └── services/          # 业务逻辑
├── alembic/               # 数据库迁移
│   └── versions/          # 迁移脚本
├── src/                   # 前端源码
│   ├── pages/            # 页面组件
│   └── components/       # UI 组件
├── docs/                  # 文档
├── Dockerfile             # Docker 镜像构建
├── docker-compose.yml     # Docker Compose 配置
├── migrate.py             # 迁移管理工具
├── requirements.txt       # Python 依赖
└── README.md             # 本文件
```

## 🔧 开发

### 后端开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行开发服务器（自动重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端开发

```bash
npm install
npm run dev  # 启动开发服务器
```

### 代码规范

- 后端: 遵循 PEP 8 规范
- 前端: 使用 ESLint 和 Prettier

## 📝 API 文档

完整的 API 文档可在运行应用后访问:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 主要端点

- `POST /v1/chat/completions` - 聊天完成 (OpenAI 兼容)
- `GET /v1/models` - 列出可用模型
- `POST /api/auth/login/access-token` - 用户登录
- `GET /api/presets/` - 获取预设列表
- `POST /api/keys/exclusive` - 生成专属密钥

## 🐛 故障排查
ports:
  - "8080:8000"  # 改为8080或其他可用端口
```

### 数据库相关

**问题：表不存在**

请手动执行迁移：
```bash
python migrate.py upgrade
```

## 🤝 贡献

欢迎贡献！请随时提交 Issue 或 Pull Request。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Google Gemini](https://ai.google.dev/) - 强大的 AI 模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [React](https://react.dev/) - UI 库
- [Alembic](https://alembic.sqlalchemy.org/) - 数据库迁移工具

## 📮 联系方式

- GitHub Issues: [提交问题](https://github.com/foamcold/any-api/issues)

---

<div align="center">
  Made with ❤️ by Your Name
</div>
