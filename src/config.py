"""Centralised configuration loaded from environment variables / .env file.

All modules should import settings from here rather than reading os.environ directly.
This makes it easy to mock in tests and switch between local / Docker / AWS environments.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (safe no-op in Docker / AWS where env vars are injected)
load_dotenv()


@dataclass
class GmailConfig:
    """Gmail API credentials and scan parameters."""

    credentials_path: str = field(
        default_factory=lambda: os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")
    )
    token_path: str = field(
        default_factory=lambda: os.getenv("GMAIL_TOKEN_PATH", "token.json")
    )
    user_id: str = field(
        default_factory=lambda: os.getenv("GMAIL_USER_ID", "me")
    )
    target_sender: str = field(
        default_factory=lambda: os.getenv(
            "GMAIL_TARGET_SENDER", "research@lolcsecurities.com"
        )
    )
    subject_keyword: str = field(
        default_factory=lambda: os.getenv("GMAIL_SUBJECT_KEYWORD", "Comp Sheet")
    )


@dataclass
class StorageConfig:
    """Local file storage paths."""

    data_dir: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "data/raw")
    )
    checkpoint_file: str = field(
        default_factory=lambda: os.getenv(
            "CHECKPOINT_FILE", "data/checkpoints/backlog_checkpoint.json"
        )
    )
    log_file: str = field(
        default_factory=lambda: os.getenv("LOG_FILE", "logs/pipeline.log")
    )

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


@dataclass
class HttpConfig:
    """HTTP / OneDrive download settings."""

    onedrive_timeout: int = field(
        default_factory=lambda: int(os.getenv("ONEDRIVE_TIMEOUT", "30"))
    )


@dataclass
class MicrosoftConfig:
    """Microsoft OAuth / OneDrive / SharePoint settings."""

    # Azure AD App Registration settings
    client_id: str = field(
        default_factory=lambda: os.getenv("MS_CLIENT_ID", "")
    )
    # For public client apps (desktop), client_secret is optional
    client_secret: str = field(
        default_factory=lambda: os.getenv("MS_CLIENT_SECRET", "")
    )
    tenant_id: str = field(
        default_factory=lambda: os.getenv("MS_TENANT_ID", "common")
    )
    # Token cache file (similar to Gmail token.json)
    token_cache_path: str = field(
        default_factory=lambda: os.getenv("MS_TOKEN_CACHE_PATH", "ms_token_cache.json")
    )
    # Scopes needed for OneDrive/SharePoint file access
    scopes: list[str] = field(
        default_factory=lambda: [
            "https://graph.microsoft.com/Files.Read",
            "https://graph.microsoft.com/Files.Read.All",
            "https://graph.microsoft.com/Sites.Read.All",
        ]
    )


@dataclass
class LLMConfig:
    """Week 2: LLM extraction settings (stub — populated in Week 2)."""

    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )


@dataclass
class DatabaseConfig:
    """Week 3: Database settings (stub — populated in Week 3)."""

    url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/cse_pipeline",
        )
    )


@dataclass
class AWSConfig:
    """Week 5: AWS settings (stub — populated in Week 5)."""

    region: str = field(
        default_factory=lambda: os.getenv("AWS_REGION", "ap-south-1")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("AWS_S3_BUCKET", "cse-pipeline-pdfs")
    )


@dataclass
class Settings:
    """Top-level settings object. Import this in application code."""

    gmail: GmailConfig = field(default_factory=GmailConfig)
    microsoft: MicrosoftConfig = field(default_factory=MicrosoftConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def configure_logging(self) -> None:
        """Set up root logger with file + stream handlers."""
        self.storage.ensure_dirs()
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        handlers: list[logging.Handler] = [logging.StreamHandler()]
        try:
            handlers.append(logging.FileHandler(self.storage.log_file))
        except OSError:
            pass  # Graceful: if log dir can't be created, stream-only is fine
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=handlers,
        )


# Module-level singleton — import from here in other modules
settings = Settings()
