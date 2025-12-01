# 数据库迁移管理文档

本项目使用 [Alembic](https://alembic.sqlalchemy.org/) 进行数据库版本管理和迁移。

## 🚀 快速开始

数据库的结构管理和版本控制需要通过手动执行迁移命令来完成。

**核心原则**
- 首次部署项目时，必须手动执行数据库迁移来创建所有表。
- 每次更新应用版本后，如果涉及到数据库模型变更，也必须手动执行迁移。

## 基本概念

### 什么是数据库迁移?

数据库迁移是一种管理数据库结构变更的方法。当你修改数据库模型(如添加字段、创建新表)时,需要将这些变更应用到数据库中。Alembic帮助你:

- 自动生成迁移脚本
- 追踪数据库版本
- 支持升级和降级
- 在团队间同步数据库结构

### 迁移脚本

迁移脚本位于 `alembic/versions/` 目录,每个脚本包含:

- `upgrade()` - 应用变更的代码
- `downgrade()` - 回滚变更的代码
- 版本标识和依赖关系

## 常用命令

项目提供了便捷的 `migrate.py` 工具,封装了Alembic的常用命令:

### 查看帮助

```bash
python migrate.py
```

### 升级数据库到最新版本

```bash
python migrate.py upgrade
```

这是最常用的命令,会执行所有未应用的迁移。

### 查看当前数据库版本

```bash
python migrate.py current
```

### 查看迁移历史

```bash
python migrate.py history
```

显示所有迁移脚本及其状态。

### 降级一个版本

```bash
python migrate.py downgrade
```

⚠️ **警告**: 降级操作可能导致数据丢失,请谨慎使用!

### 创建新迁移

```bash
python migrate.py revision "添加用户头像字段"
```

Alembic会自动检测模型变更并生成迁移脚本。

### 标记数据库版本(不执行迁移)

# 3. 执行数据库迁移
python migrate.py upgrade

# 4. 启动应用
uvicorn app.main:app --reload
```

### 现有项目升级

如果你的数据库已经存在(使用旧的迁移脚本创建),需要先标记为已迁移:

```bash
# 1. 备份数据库(重要!)
cp data/sql_app.db data/sql_app.db.backup

# 2. 安装新依赖
pip install -r requirements.txt

# 3. 将现有数据库标记为最新版本
python migrate.py stamp head

# 4. 启动应用
# 启动应用
uvicorn app.main:app --reload
```

### 开发新功能时修改数据库

假设你要为User模型添加一个avatar字段:

```bash
# 1. 修改模型文件 (app/models/user.py)
# 添加: avatar = Column(String, nullable=True)

# 2. 生成迁移脚本
python migrate.py revision "添加用户头像字段"

# 3. 检查生成的迁移脚本
# 位于 alembic/versions/[timestamp]_添加用户头像字段.py
# 确认upgrade()和downgrade()函数正确

# 4. 执行迁移
python migrate.py upgrade

# 5. 测试功能
```

## 最佳实践

### 1. 每次变更创建一个迁移

不要在一个迁移中混合多个不相关的变更。这样可以:
- 更容易理解每个迁移的目的
- 降级时更精确
- 出问题时更容易定位

### 2. 检查自动生成的迁移

Alembic的自动检测不是100%完美,建议:
- 检查生成的upgrade()和downgrade()函数
- 测试迁移是否能正常执行
- 确认没有遗漏的变更

### 3. 版本控制

将迁移脚本提交到Git:
```bash
git add alembic/versions/*.py
git commit -m "添加用户头像字段的数据库迁移"
```

### 4. 团队协作

- 在拉取代码后,记得运行 `python migrate.py upgrade`
- 如果多人同时创建迁移,可能需要手动合并

### 5. 生产环境部署

```bash
# 1. 备份数据库
# 2. 运行迁移
python migrate.py upgrade
# 3. 如果出问题,可以降级
python migrate.py downgrade
# 4. 然后恢复备份
```

## 处理迁移冲突

如果两个开发者同时创建了迁移,会出现分支:

```
    revision1 (head1)
   /
base
   \
    revision2 (head2)
```

解决方法:

```bash
# 1. 创建一个合并迁移
alembic merge -m "合并分支" head1 head2

# 2. 升级到最新版本
python migrate.py upgrade
```

## 高级用法

### 直接使用Alembic命令

如果需要更精细的控制:

```bash
# 升级到特定版本
alembic upgrade <revision_id>

# 降级到特定版本
alembic downgrade <revision_id>

# 查看SQL(不执行)
alembic upgrade head --sql

# 创建空白迁移(手动编写)
alembic revision -m "描述"
```

### 数据迁移

有时需要在结构变更的同时迁移数据:

```python
def upgrade() -> None:
    # 1. 添加新字段(允许NULL)
    op.add_column('users', sa.Column('full_name', sa.String(), nullable=True))
    
    # 2. 迁移数据
    connection = op.get_bind()
    connection.execute(
        sa.text("UPDATE users SET full_name = username WHERE full_name IS NULL")
    )
    
    # 3. 设置字段为NOT NULL
    op.alter_column('users', 'full_name', nullable=False)
```

## 故障排查

### 迁移执行失败

```bash
# 1. 查看错误信息
python migrate.py upgrade

# 2. 如果数据库处于不一致状态,可能需要:
# - 手动修复数据库
# - 或者降级到上一个稳定版本
python migrate.py downgrade

# 3. 重新尝试
python migrate.py upgrade
```

### 迁移检测不到模型变更

确保:
1. 模型已在 `alembic/env.py` 中导入
2. 模型继承自正确的Base
3. SQLAlchemy版本兼容

### "Can't locate revision" 错误

这通常意味着数据库中记录的版本在迁移脚本中找不到:

```bash
# 查看数据库当前版本
python migrate.py current

# 如果版本不存在,可以重新标记
python migrate.py stamp head
```

## 参考资源

- [Alembic官方文档](https://alembic.sqlalchemy.org/)
- [SQLAlchemy文档](https://docs.sqlalchemy.org/)
- 项目迁移脚本: `alembic/versions/`
- 迁移配置: `alembic.ini` 和 `alembic/env.py`
