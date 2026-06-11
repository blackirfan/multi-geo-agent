from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LM Studio — Gemma 3 12B via OpenAI-compatible API
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "gemma-3-12b"
    lm_studio_api_key: str = "lm-studio"

    # LLM inference knobs
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # Storage paths
    duckdb_path: str = "data/georeasoner.duckdb"
    data_vector_dir: Path = Path("data/vector")
    data_raster_dir: Path = Path("data/raster")
    reports_dir: Path = Path("reports")

    # Study area: Sylhet District, Bangladesh
    # Bounding box in (west, south, east, north) / (min_lon, min_lat, max_lon, max_lat)
    study_bbox_west: float = 91.5
    study_bbox_south: float = 24.0
    study_bbox_east: float = 92.5
    study_bbox_north: float = 25.5
    study_area_name: str = "Sylhet, Bangladesh"

    log_level: str = "INFO"

    @property
    def study_bbox(self) -> tuple[float, float, float, float]:
        """Return (west, south, east, north) bounding box for the study area."""
        return (
            self.study_bbox_west,
            self.study_bbox_south,
            self.study_bbox_east,
            self.study_bbox_north,
        )

    @property
    def lm_studio_url(self) -> str:
        return self.lm_studio_base_url.rstrip("/")


settings = Settings()
