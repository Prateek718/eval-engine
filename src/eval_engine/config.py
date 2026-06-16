"""Typed application settings, read from environment / .env.

One config surface for the whole system. Two reasons it exists rather than
scattered ``os.environ`` reads:

- The Qdrant location lives behind a setting, so the backend swaps from
  embedded local-file (dev) to a containerised server (deployment) with a
  config change, not a code change.
- CI can inject dummy values cleanly; nothing reads the environment directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Gemini embedding ---
    gemini_api_key: str = Field(default="", description="Google AI Studio API key")
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_dim: int = Field(default=768, description="MRL-truncated output dimensionality")

    # --- Qdrant ---
    # Embedded local-file mode in dev (a path). For a server, set qdrant_url
    # instead and leave qdrant_path empty; the indexer prefers url when set.
    qdrant_path: str = Field(
        default="./qdrant_data", description="Embedded local-file storage path"
    )
    qdrant_url: str = Field(default="", description="Qdrant server URL; overrides path when set")
    qdrant_collection: str = Field(default="irdai_corpus")

    # --- Agent ---
    # The grading target's model. Centralised here (not hardcoded in the
    # runner) for two reasons: gemini-2.5-flash retires 2026-10-16, so the
    # migration is a config change; and it's the single variable exposed for
    # A/B model comparison.
    agent_model: str = Field(default="gemini-2.5-flash")

    # --- Langfuse (observability) ---
    # Tracing is optional: absent keys -> NullTracer, agent runs untraced.
    # Cloud free tier in dev; host swaps to a self-hosted URL by config alone,
    # same as the Qdrant url/path split above.
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # --- MLflow (eval-run tracking) ---
    # Session-level eval aggregates land here for cross-run trends. URI is
    # DagsHub's hosted server; absent it, tracking falls back to local ./mlruns.
    mlflow_tracking_uri: str = Field(default="")
    mlflow_experiment: str = Field(default="eval-engine")

    # --- L3 regression monitoring ---
    # Each run writes its score sheet to current_sheet_path (overwritten); drift
    # is scored against the frozen baseline_sheet_path when it exists. A baseline
    # is established deliberately by promoting a chosen current sheet -- absent
    # one, the first run records itself and scores no drift. Thresholds are the
    # absolute mean deltas that count as drift; tuned against the first real
    # baseline, hence overridable, not hardcoded at the call site.
    current_sheet_path: str = Field(default="data/latest_scores.json")
    baseline_sheet_path: str = Field(default="data/baseline_scores.json")
    drift_threshold_score: float = Field(default=0.10, description="abs mean delta on [0,1] scores")
    drift_threshold_step_overage: float = Field(default=0.10)
    drift_threshold_amount_fraction: float = Field(default=0.10)
    drift_threshold_denial_rate: float = Field(
        default=0.065, description="~2/31 denied-share shift"
    )

    # --- Prometheus (metrics publishing) ---
    # The eval is a batch job, so runs push metrics to a Pushgateway that
    # Prometheus scrapes. Absent URL -> NullMetricsPublisher, run unpublished.
    pushgateway_url: str = Field(default="", description="Prometheus Pushgateway URL")


def get_settings() -> Settings:
    """Construct settings from the environment. Kept as a function so tests
    can override the environment before settings are read."""
    return Settings()
