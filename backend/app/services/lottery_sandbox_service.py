"""快乐8数据沙盘分析服务。"""

from __future__ import annotations

from collections import Counter
from datetime import date
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
        periods: int = 100,
        event_type: str = "consecutive",
        level: int = 3,
        scope: str = "global",
        zones: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """按沙盘规则过滤历史开奖数据。"""
        rows = self._window_results(start_date, end_date, issue, periods)
        events = self._build_events(rows, event_type, level, scope, zones, rules)

        return {
            "results": events[offset:offset + limit],
            "events": events[offset:offset + limit],
            "total": len(events),
            "rules": self._normalize_rules(rules),
            "limit": limit,
            "offset": offset,
            "window_size": periods,
            "actual_periods": len(rows),
            "event_type": event_type,
            "level": level,
            "scope": self._normalize_scope(scope),
            "zones": self._normalize_zones(zones),
        }

    async def analyze_intervals(
        self,
        rules: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
        periods: int = 100,
        event_type: str = "consecutive",
        level: int = 3,
        scope: str = "global",
        zones: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """统计规则命中期之间的空窗期数。"""
        rows = self._window_results(start_date, end_date, issue, periods)
        events = self._build_events(rows, event_type, level, scope, zones, rules)
        interval_rows = self._build_interval_rows(events, rows)
        gaps = [row["gap"] for row in interval_rows if isinstance(row.get("gap"), int)]
        sample_size = len(gaps)

        return {
            "rules": self._normalize_rules(rules),
            "matched_count": len(events),
            "sample_size": sample_size,
            "is_sample_sufficient": sample_size >= self.MIN_INTERVAL_SAMPLE_SIZE,
            "intervals": gaps,
            "rows": interval_rows,
            "avg_interval": round(sum(gaps) / sample_size, 2) if gaps else None,
            "min_interval": min(gaps) if gaps else None,
            "max_interval": max(gaps) if gaps else None,
            "latest_matches": events[:10],
            "message": None if sample_size >= self.MIN_INTERVAL_SAMPLE_SIZE else "间隔样本不足，结论仅供参考",
            "window_size": periods,
            "actual_periods": len(rows),
        }

    async def summarize_patterns(
        self,
        rules: Optional[List[str]] = None,
        periods: int = 100,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
        event_type: str = "consecutive",
        level: int = 3,
        scope: str = "global",
        zones: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """汇总指定窗口内的沙盘规律。"""
        rows = self._window_results(start_date, end_date, issue, periods)
        events = self._build_events(rows, event_type, level, scope, zones, rules)
        interval_rows = self._build_interval_rows(events, rows)
        gaps = [row["gap"] for row in interval_rows if isinstance(row.get("gap"), int)]
        zone_counter = Counter()

        for event in events:
            for zone in event["zones"]:
                zone_counter[zone] += 1

        latest_issue = events[0]["issue"] if events else None
        current_missing = self._current_missing(rows, latest_issue)
        summary = {
            "sample_periods": len(rows),
            "event_level": int(level),
            "hit_periods": len(events),
            "hit_rate": round(len(events) / len(rows), 4) if rows else 0,
            "total_groups": sum(event["group_count"] for event in events),
            "avg_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "median_gap": median(gaps) if gaps else None,
            "max_gap": max(gaps) if gaps else None,
            "current_missing": current_missing,
            "latest_issue": latest_issue,
            "top_zones": [
                {"zone": zone, "count": count}
                for zone, count in zone_counter.most_common()
            ],
            "baseline_delta": None,
            "updated_at": None,
        }

        return {
            "rules": self._normalize_rules(rules),
            "periods": len(rows),
            "matched_count": len(events),
            "match_rate": summary["hit_rate"],
            "rule_counts": self._count_legacy_rules(rows),
            "eight_zone_distribution": self._aggregate_eight_zones(rows),
            "highlights": self._build_summary_highlights(summary),
            "latest_matches": events[:10],
            "summary": summary,
            "window_size": periods,
            "actual_periods": len(rows),
        }

    def _window_results(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None,
        periods: int = 100,
    ) -> List[LotteryResult]:
        limit = max(1, min(int(periods or 100), 1000))
        return self._load_results(start_date=start_date, end_date=end_date, issue=issue)[:limit]

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

    def _build_events(
        self,
        rows: Sequence[LotteryResult],
        event_type: str,
        level: int,
        scope: str,
        zones: Optional[List[int]],
        rules: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_rules = self._normalize_rules(rules)
        normalized_event_type = self._normalize_event_type(event_type, rules)
        normalized_scope = self._normalize_scope(scope)
        normalized_level = self._normalize_level(level, rules)
        normalized_zones = self._normalize_zones(zones)
        events = []

        for row in rows:
            numbers = sorted({int(number) for number in row.numbers})
            if normalized_rules:
                features = self._analyze_numbers(numbers)
                if not self._features_match_rules(features, normalized_rules):
                    continue
                groups = self._groups_for_legacy_rules(numbers, normalized_rules, normalized_level)
            else:
                groups = self._match_groups(
                    numbers=numbers,
                    event_type=normalized_event_type,
                    level=normalized_level,
                    scope=normalized_scope,
                    zones=normalized_zones,
                )
            if not groups:
                continue

            event_zones = sorted({self._zone_of_number(number) for group in groups for number in group})
            events.append({
                "issue": row.issue,
                "draw_date": row.draw_date.isoformat() if row.draw_date else None,
                "openedAt": row.draw_date.isoformat() if row.draw_date else None,
                "numbers": numbers,
                "event_type": normalized_event_type,
                "scope": normalized_scope,
                "zones": event_zones,
                "groups": groups,
                "longest_length": max(len(group) for group in groups),
                "group_count": len(groups),
                "label": self._event_label(normalized_event_type, normalized_level),
                "features": self._analyze_numbers(numbers),
            })

        return events

    def _groups_for_legacy_rules(
        self,
        numbers: List[int],
        rules: List[str],
        level: int,
    ) -> List[List[int]]:
        if "eight_zone_cross" in rules:
            return self._find_eight_zone_cross_groups(numbers)
        if "gap_number" in rules:
            return self._find_gap_groups(numbers)
        if "consecutive_gap_number" in rules:
            return self._find_mixed_groups(numbers)
        return self._find_consecutive_groups(numbers, level)

    def _match_groups(
        self,
        numbers: List[int],
        event_type: str,
        level: int,
        scope: str,
        zones: List[int],
    ) -> List[List[int]]:
        scoped_sets = self._scoped_number_sets(numbers, scope, zones)
        groups = []

        for number_set in scoped_sets:
            if event_type == "gap":
                groups.extend(self._find_gap_groups(number_set))
            elif event_type in {"mixed", "interval"}:
                groups.extend(self._find_mixed_groups(number_set))
            else:
                groups.extend(self._find_consecutive_groups(number_set, level))

        return self._unique_groups(groups)

    def _scoped_number_sets(self, numbers: List[int], scope: str, zones: List[int]) -> List[List[int]]:
        if scope == "global":
            return [numbers]

        target_zones = zones or list(range(1, 9))
        return [
            [number for number in numbers if self._zone_of_number(number) == zone]
            for zone in target_zones
        ]

    def _matches_rules(self, row: LotteryResult, rules: List[str]) -> bool:
        return self._features_match_rules(self._analyze_numbers(row.numbers), rules)

    def _features_match_rules(self, features: Dict[str, Any], rules: List[str]) -> bool:
        if not rules:
            return True
        return all(bool(features.get(rule)) for rule in rules)

    def _analyze_numbers(self, numbers: List[int]) -> Dict[str, Any]:
        sorted_numbers = sorted({int(number) for number in numbers})
        consecutive_runs = self._find_consecutive_groups(sorted_numbers, level=2)
        gap_runs = self._find_gap_chains(sorted_numbers)
        mixed_runs = self._find_mixed_groups(sorted_numbers)
        eight_zone_distribution = self._calculate_eight_zone_distribution(sorted_numbers)

        return {
            "two_consecutive": any(len(run) >= 2 for run in consecutive_runs),
            "three_consecutive": any(len(run) >= 3 for run in consecutive_runs),
            "four_consecutive": any(len(run) >= 4 for run in consecutive_runs),
            "eight_zone_cross": self._has_eight_zone_cross(sorted_numbers),
            "gap_number": bool(self._find_gap_groups(sorted_numbers)),
            "consecutive_gap_number": bool(mixed_runs),
            "consecutive_runs": consecutive_runs,
            "gap_runs": gap_runs,
            "mixed_runs": mixed_runs,
            "eight_zone_distribution": eight_zone_distribution,
        }

    @classmethod
    def _find_consecutive_groups(cls, numbers: List[int], level: int) -> List[List[int]]:
        runs = cls._find_runs(numbers, step=1)
        return [run for run in runs if len(run) >= level]

    @staticmethod
    def _find_gap_groups(numbers: List[int]) -> List[List[int]]:
        number_set = set(numbers)
        return [[number, number + 2] for number in numbers if number + 2 in number_set]

    @classmethod
    def _find_gap_chains(cls, numbers: List[int]) -> List[List[int]]:
        return cls._find_runs(numbers, step=2)

    @staticmethod
    def _find_mixed_groups(numbers: List[int]) -> List[List[int]]:
        groups = []
        sorted_numbers = sorted(numbers)

        for start in range(len(sorted_numbers)):
            for length in (3, 4):
                group = sorted_numbers[start:start + length]
                if len(group) == length and LotterySandboxService._is_mixed_group(group):
                    groups.append(group)

        return groups

    @staticmethod
    def _is_mixed_group(group: List[int]) -> bool:
        if len(group) < 3:
            return False
        diffs = [group[index + 1] - group[index] for index in range(len(group) - 1)]
        return all(diff in {1, 2} for diff in diffs) and 1 in diffs and 2 in diffs

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
    def _unique_groups(groups: Iterable[List[int]]) -> List[List[int]]:
        seen = set()
        unique = []
        for group in groups:
            key = tuple(group)
            if key in seen:
                continue
            seen.add(key)
            unique.append(group)
        return unique

    @staticmethod
    def _calculate_eight_zone_distribution(numbers: List[int]) -> Dict[str, int]:
        distribution = {f"zone_{index}": 0 for index in range(1, 9)}
        for number in numbers:
            zone_index = LotterySandboxService._zone_of_number(number)
            distribution[f"zone_{zone_index}"] += 1
        return distribution

    @staticmethod
    def _zone_of_number(number: int) -> int:
        return max(1, min((int(number) - 1) // 10 + 1, 8))

    @staticmethod
    def _has_eight_zone_cross(numbers: List[int]) -> bool:
        boundaries = {10, 20, 30, 40, 50, 60, 70}
        number_set = set(numbers)
        return any(boundary in number_set and boundary + 1 in number_set for boundary in boundaries)

    @staticmethod
    def _find_eight_zone_cross_groups(numbers: List[int]) -> List[List[int]]:
        boundaries = {10, 20, 30, 40, 50, 60, 70}
        number_set = set(numbers)
        return [[boundary, boundary + 1] for boundary in sorted(boundaries) if boundary in number_set and boundary + 1 in number_set]

    def _build_interval_rows(
        self,
        events: Sequence[Dict[str, Any]],
        rows: Sequence[LotteryResult],
    ) -> List[Dict[str, Any]]:
        chronological = list(reversed(events))
        row_index = {row.issue: index for index, row in enumerate(reversed(rows))}
        interval_rows = []

        for index, event in enumerate(chronological):
            next_event = chronological[index + 1] if index + 1 < len(chronological) else None
            current_index = row_index.get(event["issue"])
            next_index = row_index.get(next_event["issue"]) if next_event else None
            distance = (
                next_index - current_index
                if isinstance(current_index, int) and isinstance(next_index, int)
                else None
            )
            gap = max(0, distance - 1) if isinstance(distance, int) else None
            interval_rows.append({
                "issue": event["issue"],
                "draw_date": event.get("draw_date"),
                "next_issue": next_event["issue"] if next_event else None,
                "gap": gap,
                "distance": distance,
            })

        return interval_rows

    @staticmethod
    def _current_missing(rows: Sequence[LotteryResult], latest_issue: Optional[str]) -> Optional[int]:
        if not latest_issue:
            return None
        for index, row in enumerate(rows):
            if row.issue == latest_issue:
                return index
        return None

    def _count_legacy_rules(self, rows: Sequence[LotteryResult]) -> Dict[str, int]:
        counter = Counter()
        for row in rows:
            features = self._analyze_numbers(row.numbers)
            for rule in self._available_rules():
                if features.get(rule):
                    counter[rule] += 1
        return dict(counter)

    def _aggregate_eight_zones(self, rows: Sequence[LotteryResult]) -> Dict[str, int]:
        counter = Counter()
        for row in rows:
            for zone, count in self._calculate_eight_zone_distribution(row.numbers).items():
                counter[zone] += count
        return dict(counter)

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
            "mixed": "consecutive_gap_number",
            "consecutive_gap": "consecutive_gap_number",
        }
        available = set(cls._available_rules())
        normalized = []

        for rule in rules:
            value = aliases.get(str(rule).strip(), str(rule).strip())
            if value in available and value not in normalized:
                normalized.append(value)

        return normalized

    @classmethod
    def _normalize_event_type(cls, event_type: str, rules: Optional[List[str]] = None) -> str:
        normalized_rules = cls._normalize_rules(rules)
        if "gap_number" in normalized_rules:
            return "gap"
        if "consecutive_gap_number" in normalized_rules:
            return "mixed"
        if event_type in {"consecutive", "gap", "mixed", "interval"}:
            return event_type
        return "consecutive"

    @classmethod
    def _normalize_level(cls, level: int, rules: Optional[List[str]] = None) -> int:
        normalized_rules = cls._normalize_rules(rules)
        if "four_consecutive" in normalized_rules:
            return 4
        if "three_consecutive" in normalized_rules:
            return 3
        if "two_consecutive" in normalized_rules:
            return 2
        return int(level) if int(level) in {2, 3, 4} else 3

    @staticmethod
    def _normalize_scope(scope: str) -> str:
        return "zone" if scope == "zone" else "global"

    @staticmethod
    def _normalize_zones(zones: Optional[List[int]]) -> List[int]:
        if not zones:
            return []
        return sorted({int(zone) for zone in zones if 1 <= int(zone) <= 8})

    @staticmethod
    def _event_label(event_type: str, level: int) -> str:
        if event_type == "gap":
            return "隔号"
        if event_type == "mixed":
            return "连号隔号"
        if event_type == "interval":
            return f"{level}连间隔"
        return f"{level}连号"

    @staticmethod
    def _build_summary_highlights(summary: Dict[str, Any]) -> List[str]:
        periods = summary["sample_periods"]
        if periods <= 0:
            return ["暂无可分析样本"]

        highlights = [
            f"命中 {summary['hit_periods']} 期，占比 {summary['hit_rate']:.1%}",
        ]
        if summary["avg_gap"] is not None:
            highlights.append(f"平均空窗 {summary['avg_gap']} 期")
        if summary["top_zones"]:
            top_zone = summary["top_zones"][0]
            highlights.append(f"{top_zone['zone']}区命中最多，共 {top_zone['count']} 次")
        return highlights
