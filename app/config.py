from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: Literal["development", "test", "production"] = "development"
    app_timezone: str = "Asia/Ho_Chi_Minh"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini-2024-07-18"
    openai_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    openai_max_retries: int = Field(default=2, ge=0, le=5)
    openai_max_output_tokens: int = Field(default=1200, ge=100, le=4000)

    internal_api_key: SecretStr | None = None
    allow_public_extraction: bool = False
    cors_allowed_origins: str = ""
    max_prompt_chars: int = Field(default=50_000, ge=5_000, le=200_000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def openai_api_key_value(self) -> str | None:
        if self.openai_api_key is None:
            return None
        value = self.openai_api_key.get_secret_value().strip()
        return value or None

    @property
    def internal_api_key_value(self) -> str | None:
        if self.internal_api_key is None:
            return None
        value = self.internal_api_key.get_secret_value().strip()
        return value or None

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if (
            self.app_env == "production"
            and not self.internal_api_key_value
            and not self.allow_public_extraction
        ):
            raise ValueError(
                "INTERNAL_API_KEY is required in production unless "
                "ALLOW_PUBLIC_EXTRACTION is enabled"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
