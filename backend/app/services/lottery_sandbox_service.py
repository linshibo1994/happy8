"""快乐8数据沙盘分析服务。"""

from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.prediction import LotteryResult
from app.services.lottery_service import LotteryService


class LotterySandboxService:
    """封装快乐8沙盘规则过滤、间隔分析与规律总结。"""

    MIN_INTERVAL_SAMPLE_SIZE = 2

    def __init__(self, db: Session):
        self.db = db
        self.lottery_service = LotteryService(db)

    async def filter_results(
        self,
        rules: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """按沙盘规则过滤历史开奖数据。"""
        rules = self._normalize_rules(rules)
        rows = self._load_results(start_date=start_date, end_date=end_date, issue=issue)
        matched = [self._serialize_with_features(row) for row in rows if self._matches_rules(row, rules)]

        return {
            "results": matched[offset:offset + limit],
            "total": len(matched),
            "rules": rules,
            "limit": limit,
            "offset": offset,
        }

    async def analyze_intervals(
        self,
        rules: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """统计规则命中期之间的间隔。"""
        rules = self._normalize_rules(rules)
        rows = list(reversed(self._load_results(start_date=start_date, end_date=end_date, issue=issue)))
        matched_positions = []
        matched_results = []

        for index, row in enumerate(rows):
            if self._matches_rules(row, rules):
                matched_positions.append(index)
                matched_results.append(self._serialize_with_features(row))

        intervals = [
            matched_positions[index] - matched_positions[index - 1]
            for index in range(1, len(matched_positions))
        ]
        sample_size = len(intervals)

        return {
            "rules": rules,
            "matched_count": len(matched_results),
            "sample_size": sample_size,
            "is_sample_sufficient": sample_size >= self.MIN_INTERVAL_SAMPLE_SIZE,
            "intervals": intervals,
            "avg_interval": round(sum(intervals) / sample_size, 2) if intervals else None,
            "min_interval": min(intervals) if intervals else None,
            "max_interval": max(intervals) if intervals else None,
            "latest_matches": list(reversed(matched_results[-10:])),
            "message": None if sample_size >= self.MIN_INTERVAL_SAMPLE_SIZE else "间隔样本不足，结论仅供参考",
        }

    async def summarize_patterns(
        self,
        rules: Optional[List[str]] = None,
        periods: int = 100,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """汇总指定窗口内的沙盘规律。"""
        rules = self._normalize_rules(rules)
        rows = self._load_results(start_date=start_date, end_date=end_date, issue=issue)[:periods]
        feature_rows = [self._serialize_with_features(row) for row in rows]
        rule_counts = Counter()
        zone_counter = Counter()

        for item in feature_rows:
            for rule in self._available_rules():
                if item["features"].get(rule):
                    rule_counts[rule] += 1
            for zone, count in item["features"]["eight_zone_distribution"].items():
                zone_counter[zone] += count

        matched = [item for item in feature_rows if self._features_match_rules(item["features"], rules)]
        highlights = self._build_summary_highlights(rule_counts, len(feature_rows), zone_counter)

        return {
            "rules": rules,
            "periods": len(feature_rows),
            "matched_count": len(matched),
            "match_rate": round(len(matched) / len(feature_rows), 4) if feature_rows else 0,
            "rule_counts": dict(rule_counts),
            "eight_zone_distribution": dict(zone_counter),
            "highlights": highlights,
            "latest_matches": matched[:10],
        }

    def _load_results(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
    ) -> List[LotteryResult]:
        query = self.lottery_service._build_history_query(start_date, end_date, issue)
        return query.order_by(LotteryResult.draw_date.desc(), LotteryResult.issue.desc()).all()

    def _serialize_with_features(self, row: LotteryResult) -> Dict[str, Any]:
        data = self.lottery_service._format_lottery_result(row)
        data["features"] = self._analyze_numbers(data["numbers"])
        return data

    def _matches_rules(self, row: LotteryResult, rules: List[str]) -> bool:
        return self._features_match_rules(self._analyze_numbers(row.numbers), rules)

    def _features_match_rules(self, features: Dict[str, Any], rules: List[str]) -> bool:
        if not rules:
            return True
        return all(bool(features.get(rule)) for rule in rules)

    def _analyze_numbers(self, numbers: List[int]) -> Dict[str, Any]:
        sorted_numbers = sorted({int(number) for number in numbers})
        consecutive_runs = self._find_runs(sorted_numbers, step=1)
        gap_runs = self._find_runs(sorted_numbers, step=2)
        eight_zone_distribution = self._calculate_eight_zone_distribution(sorted_numbers)

        return {
            "two_consecutive": any(len(run) >= 2 for run in consecutive_runs),
            "three_consecutive": any(len(run) >= 3 for run in consecutive_runs),
            "four_consecutive": any(len(run) >= 4 for run in consecutive_runs),
            "eight_zone_cross": self._has_eight_zone_cross(sorted_numbers),
            "gap_number": any(len(run) >= 2 for run in gap_runs),
            "consecutive_gap_number": bool(consecutive_runs and gap_runs),
            "consecutive_runs": consecutive_runs,
            "gap_runs": gap_runs,
            "eight_zone_distribution": eight_zone_distribution,
        }

    @staticmethod
    def _find_runs(numbers: List[int], step: int) -> List[List[int]]:
        runs = []
        current = []

        for number in numbers:
            if not current or number - current[-1] == step:
                current.append(number)
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = [number]

        if len(current) >= 2:
            runs.append(current)

        return runs

    @staticmethod
    def _calculate_eight_zone_distribution(numbers: List[int]) -> Dict[str, int]:
        distribution = {f"zone_{index}": 0 for index in range(1, 9)}
        for number in numbers:
            zone_index = min((number - 1) // 10 + 1, 8)
            distribution[f"zone_{zone_index}"] += 1
        return distribution

    @staticmethod
    def _has_eight_zone_cross(numbers: List[int]) -> bool:
        boundaries = {10, 20, 30, 40, 50, 60, 70}
        number_set = set(numbers)
        return any(boundary in number_set and boundary + 1 in number_set for boundary in boundaries)

    @classmethod
    def _available_rules(cls) -> List[str]:
        return [
            "two_consecutive",
            "three_consecutive",
            "four_consecutive",
            "eight_zone_cross",
            "gap_number",
            "consecutive_gap_number",
        ]

    @classmethod
    def _normalize_rules(cls, rules: Optional[List[str]]) -> List[str]:
        if not rules:
            return []

        aliases = {
            "two": "two_consecutive",
            "three": "three_consecutive",
            "four": "four_consecutive",
            "cross": "eight_zone_cross",
            "gap": "gap_number",
            "consecutive_gap": "consecutive_gap_number",
        }
        available = set(cls._available_rules())
        normalized = []

        for rule in rules:
            value = aliases.get(str(rule).strip(), str(rule).strip())
            if value in available and value not in normalized:
                normalized.append(value)

        return normalized

    @staticmethod
    def _build_summary_highlights(
        rule_counts: Counter,
        periods: int,
        zone_counter: Counter,
    ) -> List[str]:
        if periods <= 0:
            return ["暂无可分析样本"]

        highlights = []
        if rule_counts:
            top_rule, top_count = rule_counts.most_common(1)[0]
            highlights.append(f"{top_rule} 命中 {top_count} 期，占比 {top_count / periods:.1%}")

        if zone_counter:
            top_zone, top_count = zone_counter.most_common(1)[0]
            highlights.append(f"{top_zone} 出号最多，共 {top_count} 个")

        return highlights
