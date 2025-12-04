import json
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.api import deps
from app.models.system_config import SystemConfig as SystemConfigModel
from app.models.user import User
from app.models.key import OfficialKey
from app.models.log import Log
from app.models.key import ExclusiveKey
from app.schemas.system_config import SystemConfig, SystemConfigUpdate
from sqlalchemy import func, desc
from app.core.logging import setup_logging
import logging

router = APIRouter()

@router.get("/stats")
async def get_system_stats(
    db: AsyncSession = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
):
    """
    获取系统统计信息
    """
    # 总请求数 (从日志统计)
    total_requests_result = await db.execute(
        select(func.count(Log.id))
        .filter(Log.user_id == current_user.id)
    )
    total_requests = total_requests_result.scalar_one_or_none() or 0

    # 活跃密钥数 (不重复的、状态正常的渠道密钥)
    active_keys_result = await db.execute(
        select(func.count(func.distinct(OfficialKey.key)))
        .filter(
            OfficialKey.user_id == current_user.id,
            OfficialKey.is_active == True,
            (OfficialKey.last_status == "active") | (OfficialKey.last_status == "200")
        )
    )
    active_keys = active_keys_result.scalar_one_or_none() or 0

    # 总令牌数 (从日志统计)
    total_tokens_result = await db.execute(
        select(func.sum(Log.input_tokens + Log.output_tokens))
        .filter(Log.user_id == current_user.id)
    )
    total_tokens = total_tokens_result.scalar_one_or_none() or 0
    
    # 平均延迟（最近10条无报错）
    avg_latency_result = await db.execute(
        select(func.avg(Log.latency))
        .filter(Log.user_id == current_user.id, Log.status == 'ok')
        .order_by(desc(Log.created_at))
        .limit(10)
    )
    avg_latency = avg_latency_result.scalar_one_or_none() or 0

    return {
        "total_requests": total_requests,
        "active_keys": active_keys,
        "total_tokens": total_tokens,
        "avg_latency": avg_latency,
    }

@router.get("/config", response_model=SystemConfig)
async def get_system_config(
    db: AsyncSession = Depends(deps.get_db),
    current_user: Optional[User] = Depends(deps.get_optional_current_user),
) -> Any:
    """
    获取系统配置。
    - 未登录用户可获取公开配置。
    - 管理员可获取包含敏感信息的完整配置。
    """
    result = await db.execute(select(SystemConfigModel))
    config = result.scalars().first()
    
    if not config:
        config = SystemConfigModel()
        db.add(config)
        await db.commit()
        await db.refresh(config)

    # 基础公开配置
    config_dict = {
        "id": config.id,
        "site_name": config.site_name,
        "server_url": config.server_url,
        "allow_registration": config.allow_registration,
        "allow_password_login": config.allow_password_login,
        "require_email_verification": config.require_email_verification,
        "enable_turnstile": config.enable_turnstile,
        "enable_captcha": config.enable_captcha,
        "enable_ip_rate_limit": config.enable_ip_rate_limit,
        "email_whitelist_enabled": config.email_whitelist_enabled,
        "email_whitelist": json.loads(config.email_whitelist) if config.email_whitelist else [],
        "log_level": config.log_level,
        "turnstile_site_key": config.turnstile_site_key if config.enable_turnstile else None,
    }

    # 如果是管理员，补充敏感信息
    if current_user and current_user.role in ["admin", "super_admin"]:
        config_dict.update({
            "smtp_host": config.smtp_host,
            "smtp_port": config.smtp_port,
            "smtp_user": config.smtp_user,
            "smtp_password": config.smtp_password,
            "smtp_from": config.smtp_from,
            "smtp_use_tls": config.smtp_use_tls,
            "turnstile_secret_key": config.turnstile_secret_key,
        })
    else:
        # 对于非管理员或未登录用户，确保敏感字段为 None
        config_dict.update({
            "smtp_host": None,
            "smtp_port": 587,
            "smtp_user": None,
            "smtp_password": None,
            "smtp_from": None,
            "smtp_use_tls": True,
            "turnstile_secret_key": None,
        })

    return config_dict

@router.put("/config", response_model=SystemConfig)
async def update_system_config(
    *,
    db: AsyncSession = Depends(deps.get_db),
    config_in: SystemConfigUpdate,
    current_user: User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    更新系统配置（仅管理员）
    """
    result = await db.execute(select(SystemConfigModel))
    config = result.scalars().first()
    
    if not config:
        config = SystemConfigModel()
        db.add(config)

    # 更新配置
    config.site_name = config_in.site_name
    config.server_url = config_in.server_url
    config.allow_registration = config_in.allow_registration
    config.allow_password_login = config_in.allow_password_login
    config.require_email_verification = config_in.require_email_verification
    config.enable_turnstile = config_in.enable_turnstile
    config.enable_captcha = config_in.enable_captcha
    config.enable_ip_rate_limit = config_in.enable_ip_rate_limit
    config.email_whitelist_enabled = config_in.email_whitelist_enabled
    config.email_whitelist = json.dumps(config_in.email_whitelist)
    
    # SMTP配置
    config.smtp_host = config_in.smtp_host
    config.smtp_port = config_in.smtp_port
    config.smtp_user = config_in.smtp_user
    if config_in.smtp_password:  # 只在提供了密码时更新
        config.smtp_password = config_in.smtp_password
    config.smtp_from = config_in.smtp_from
    config.smtp_use_tls = config_in.smtp_use_tls
    
    # Turnstile配置
    config.turnstile_site_key = config_in.turnstile_site_key
    if config_in.turnstile_secret_key:  # 只在提供了密钥时更新
        config.turnstile_secret_key = config_in.turnstile_secret_key
    
    # 日志配置
    config.log_level = config_in.log_level

    await db.commit()
    await db.refresh(config)
    
    # 如果日志等级发生变化，立即应用新的日志设置
    if config_in.log_level:
        try:
            setup_logging(log_level=config_in.log_level)
            logging.info(f"管理员已更新日志等级为: {config_in.log_level}")
        except Exception as e:
            logging.error(f"应用新日志等级失败: {e}")

    # 返回完整配置
    return {
        "id": config.id,
        "site_name": config.site_name,
        "server_url": config.server_url,
        "allow_registration": config.allow_registration,
        "allow_password_login": config.allow_password_login,
        "require_email_verification": config.require_email_verification,
        "enable_turnstile": config.enable_turnstile,
        "enable_captcha": config.enable_captcha,
        "enable_ip_rate_limit": config.enable_ip_rate_limit,
        "email_whitelist_enabled": config.email_whitelist_enabled,
        "email_whitelist": json.loads(config.email_whitelist),
        "smtp_host": config.smtp_host,
        "smtp_port": config.smtp_port,
        "smtp_user": config.smtp_user,
        "smtp_password": config.smtp_password,
        "smtp_from": config.smtp_from,
        "smtp_use_tls": config.smtp_use_tls,
        "turnstile_site_key": config.turnstile_site_key,
        "turnstile_secret_key": config.turnstile_secret_key,
        "log_level": config.log_level,
    }
