"""Configuration management for the application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Application configuration."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://creditbench:creditbench@localhost:5432/creditbench")

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

    # Data Source (local OneDrive mount)
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # Azure Blob Storage (optional — if set, DuckDB queries parquet from Azure)
    AZURE_STORAGE_ACCOUNT: str = os.getenv("AZURE_STORAGE_ACCOUNT", "")
    AZURE_STORAGE_SAS_TOKEN: str = os.getenv("AZURE_STORAGE_SAS_TOKEN", "")
    AZURE_BLOB_CONTAINER: str = os.getenv("AZURE_BLOB_CONTAINER", "creditbench")
    AZURE_FUNDAMENTALS_PATH: str = os.getenv("AZURE_FUNDAMENTALS_PATH", "foundamentals")

    # Embedding (Gemini)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    @property
    def use_azure(self) -> bool:
        return bool(self.AZURE_STORAGE_ACCOUNT and self.AZURE_STORAGE_SAS_TOKEN)

    # Application Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required but not set")
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is required but not set")


config = Config()
settings = config  # Alias for compatibility
