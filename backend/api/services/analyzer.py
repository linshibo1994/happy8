import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
src_root_str = str(SRC_ROOT)
if src_root_str not in sys.path:
    sys.path.insert(0, src_root_str)

class AnalyzerSingleton:
    _instance: Optional[Any] = None
    _is_loaded: bool = False
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, ensure_loaded: bool = True) -> Any:
        with cls._lock:
            if cls._instance is None:
                from happy8_analyzer import Happy8Analyzer  # noqa: E402

                cls._instance = Happy8Analyzer(data_dir=str(PROJECT_ROOT / "data"))

            if ensure_loaded and not cls._is_loaded:
                cls._instance.load_data()
                cls._is_loaded = True

            return cls._instance

    @classmethod
    def reload_data(cls) -> int:
        analyzer = cls.get_instance(ensure_loaded=False)
        analyzer.historical_data = None
        data = analyzer.load_data()
        cls._is_loaded = True
        return len(data)

    @classmethod
    def refresh_from_network(cls) -> Dict[str, Any]:
        analyzer = cls.get_instance(ensure_loaded=False)
        crawled = analyzer.crawl_all_historical_data()
        analyzer.historical_data = None
        data = analyzer.load_data()
        cls._is_loaded = True
        latest_issue = str(data.iloc[0]["issue"]) if len(data) > 0 else None
        return {
            "crawled_count": crawled,
            "total_records": int(len(data)),
            "latest_issue": latest_issue,
        }

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        if cls._instance is None:
            return {
                "initialized": False,
                "loaded": False,
                "total_records": 0,
                "latest_issue": None,
            }

        analyzer = cls._instance
        data = analyzer.historical_data
        if data is None:
            return {
                "initialized": cls._instance is not None,
                "loaded": False,
                "total_records": 0,
                "latest_issue": None,
            }

        latest_issue = str(data.iloc[0]["issue"]) if len(data) > 0 else None
        return {
            "initialized": cls._instance is not None,
            "loaded": cls._is_loaded,
            "total_records": int(len(data)),
            "latest_issue": latest_issue,
        }


def get_analyzer_instance(ensure_loaded: bool = True) -> Any:
    return AnalyzerSingleton.get_instance(ensure_loaded=ensure_loaded)


def refresh_data_from_network() -> Dict[str, Any]:
    return AnalyzerSingleton.refresh_from_network()


def reload_local_data() -> int:
    return AnalyzerSingleton.reload_data()


def get_analyzer_status() -> Dict[str, Any]:
    return AnalyzerSingleton.get_status()
