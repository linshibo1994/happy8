import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Happy8 Prediction API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    CORS_ORIGINS: List[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "CORS_ORIGINS",
            [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()],
        )


settings = Settings()


METHOD_MAPPING: Dict[str, str] = {
    "frequency": "predict_by_frequency",
    "hot_cold": "predict_by_hot_cold",
    "missing": "predict_by_missing",
    "markov": "predict_by_markov",
    "markov_2nd": "predict_by_markov_2nd",
    "markov_3rd": "predict_by_markov_3rd",
    "adaptive_markov": "predict_by_adaptive_markov",
    "transformer": "predict_by_transformer",
    "gnn": "predict_by_gnn",
    "monte_carlo": "predict_by_monte_carlo",
    "clustering": "predict_by_clustering",
    "advanced_ensemble": "predict_by_advanced_ensemble",
    "bayesian": "predict_by_bayesian",
    "super_predictor": "predict_by_super_predictor",
    "high_confidence": "predict_by_high_confidence",
    "lstm": "predict_by_lstm",
    "ensemble": "predict_by_ensemble_learning",
}


METHOD_GROUPS: Dict[str, List[str]] = {
    "统计类": ["frequency", "hot_cold", "missing"],
    "马尔可夫": ["markov", "markov_2nd", "markov_3rd", "adaptive_markov"],
    "深度学习": ["transformer", "gnn", "lstm"],
    "机器学习": ["monte_carlo", "clustering", "advanced_ensemble", "ensemble"],
    "智能综合": ["bayesian", "super_predictor", "high_confidence"],
}

