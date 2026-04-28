from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: SecretStr
    gemini_api_key: SecretStr

    @field_validator("groq_api_key", "gemini_api_key")
    @classmethod
    def reject_placeholder_keys(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if not raw or raw.startswith("your_"):
            raise ValueError("Placeholder API key detected — set a real key in .env")
        return value

    log_level: str = Field(default="INFO", validation_alias="PREDICTOR_LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), validation_alias="PREDICTOR_DATA_DIR")

    @field_validator("data_dir")
    @classmethod
    def ensure_data_dirs_exist(cls, value: Path) -> Path:
        value = value.resolve()
        value.mkdir(parents=True, exist_ok=True)
        (value / "outputs").mkdir(exist_ok=True)
        (value / "logs").mkdir(exist_ok=True)
        return value

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"


settings = Settings()
