from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    
    # Application
    VERSION: str = "1.0.0"
    VITE_API_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-it-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200 # 30 days
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/sql_app.db"

    # Proxy
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"

    model_config = SettingsConfigDict(extra='ignore', env_file='.env', env_file_encoding='utf-8')

settings = Settings()
