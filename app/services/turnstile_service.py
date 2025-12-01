import aiohttp
from typing import Optional

class TurnstileService:
    """Cloudflare Turnstile 验证服务"""
    
    VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    
    def __init__(self):
        self.secret_key: Optional[str] = None
        
    def configure(self, secret_key: str):
        """配置密钥"""
        self.secret_key = secret_key
        
    def is_configured(self) -> bool:
        """检查是否已配置"""
        return bool(self.secret_key)
    
    async def verify_token(self, token: str, ip: Optional[str] = None) -> bool:
        """
        验证Turnstile token
        
        Args:
            token: 前端获取的token
            ip: 用户IP地址（可选）
            
        Returns:
            验证是否成功
        """
        if not self.is_configured():
            raise Exception("Turnstile not configured")
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    'secret': self.secret_key,
                    'response': token,
                }
                
                if ip:
                    data['remoteip'] = ip
                
                async with session.post(self.VERIFY_URL, data=data) as response:
                    result = await response.json()
                    print(f"Turnstile验证结果: {result}")
                    if not result.get('success', False):
                        print(f"Turnstile验证失败: {result.get('error-codes', [])}")
                    return result.get('success', False)
        except Exception as e:
            print(f"Turnstile verification error: {e}")
            return False

# 全局实例
turnstile_service = TurnstileService()
