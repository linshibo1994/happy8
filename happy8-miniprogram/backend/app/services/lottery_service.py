from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pathlib import Path
import sys
from app.models.prediction import LotteryResult
import logging
import asyncio

logger = logging.getLogger(__name__)

class LotteryService:
    def __init__(self, db: Session):
        self.db = db
    
    async def get_latest_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最新开奖结果"""
        try:
            results = (
                self.db.query(LotteryResult)
                .order_by(LotteryResult.draw_date.desc())
                .limit(limit)
                .all()
            )
            
            return [self._format_lottery_result(result) for result in results]
        except Exception as e:
            logger.error(f"获取最新开奖结果失败: {str(e)}")
            raise
    
    async def get_historical_results(
        self,
        limit: int = 20,
        offset: int = 0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取历史开奖结果"""
        try:
            query = self.db.query(LotteryResult)
            
            # 添加筛选条件
            if start_date:
                query = query.filter(LotteryResult.draw_date >= start_date)
            if end_date:
                query = query.filter(LotteryResult.draw_date <= end_date)
            if issue:
                query = query.filter(LotteryResult.issue.like(f"%{issue}%"))
            
            results = (
                query.order_by(LotteryResult.draw_date.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            return [self._format_lottery_result(result) for result in results]
        except Exception as e:
            logger.error(f"获取历史开奖结果失败: {str(e)}")
            raise
    
    async def get_result_by_issue(self, issue: str) -> Optional[Dict[str, Any]]:
        """根据期号获取开奖结果"""
        try:
            result = (
                self.db.query(LotteryResult)
                .filter(LotteryResult.issue == issue)
                .first()
            )
            
            if result:
                return self._format_lottery_result(result)
            return None
        except Exception as e:
            logger.error(f"获取期号{issue}开奖结果失败: {str(e)}")
            raise
    
    async def get_statistics(self, periods: int, stat_type: str) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            # 获取最近N期数据
            results = (
                self.db.query(LotteryResult)
                .order_by(LotteryResult.draw_date.desc())
                .limit(periods)
                .all()
            )
            
            if stat_type == "frequency":
                return self._calculate_frequency_stats(results)
            elif stat_type == "hot_cold":
                return self._calculate_hot_cold_stats(results)
            elif stat_type == "missing":
                return self._calculate_missing_stats(results)
            elif stat_type == "zone":
                return self._calculate_zone_stats(results)
            else:
                raise ValueError(f"不支持的统计类型: {stat_type}")
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            raise
    
    async def get_trends(self, periods: int, numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """获取走势数据"""
        try:
            results = (
                self.db.query(LotteryResult)
                .order_by(LotteryResult.draw_date.desc())
                .limit(periods)
                .all()
            )
            
            if numbers:
                # 特定号码走势
                return self._calculate_number_trends(results, numbers)
            else:
                # 整体走势
                return self._calculate_overall_trends(results)
                
        except Exception as e:
            logger.error(f"获取走势数据失败: {str(e)}")
            raise
    
    async def search_results(
        self,
        numbers: Optional[List[int]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """搜索开奖结果"""
        try:
            query = self.db.query(LotteryResult)
            
            # 按号码搜索
            if numbers:
                for number in numbers:
                    query = query.filter(LotteryResult.numbers.contains([number]))
            
            # 按日期范围搜索
            if start_date:
                query = query.filter(LotteryResult.draw_date >= start_date)
            if end_date:
                query = query.filter(LotteryResult.draw_date <= end_date)
            
            results = (
                query.order_by(LotteryResult.draw_date.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            
            return [self._format_lottery_result(result) for result in results]
        except Exception as e:
            logger.error(f"搜索开奖结果失败: {str(e)}")
            raise
    
    async def sync_latest_data(self) -> int:
        """同步最新开奖数据"""
        try:
            # 获取最新期号
            latest_result = (
                self.db.query(LotteryResult)
                .order_by(LotteryResult.draw_date.desc())
                .first()
            )
            
            # 从数据源获取最新数据
            new_results = await self._fetch_latest_from_source(latest_result)
            
            # 保存到数据库
            saved_count = 0
            for result_data in new_results:
                existing = (
                    self.db.query(LotteryResult)
                    .filter(LotteryResult.issue == result_data["issue"])
                    .first()
                )
                
                if not existing:
                    lottery_result = LotteryResult(**result_data)
                    self.db.add(lottery_result)
                    saved_count += 1
            
            self.db.commit()
            return saved_count
            
        except Exception as e:
            logger.error(f"同步最新数据失败: {str(e)}")
            self.db.rollback()
            raise
    
    def _format_lottery_result(self, result: LotteryResult) -> Dict[str, Any]:
        """格式化开奖结果"""
        return {
            "id": result.id,
            "issue": result.issue,
            "draw_date": result.draw_date.isoformat(),
            "numbers": result.numbers,
            "sum_value": result.sum_value,
            "odd_count": result.odd_count,
            "even_count": result.even_count,
            "big_count": result.big_count,
            "small_count": result.small_count,
            "created_at": result.created_at.isoformat()
        }
    
    def _calculate_frequency_stats(self, results: List[LotteryResult]) -> Dict[str, Any]:
        """计算频率统计"""
        frequency = {}
        total_draws = len(results)
        
        for result in results:
            for number in result.numbers:
                frequency[number] = frequency.get(number, 0) + 1
        
        # 计算出现率
        frequency_rate = {
            num: count / total_draws for num, count in frequency.items()
        }
        
        return {
            "type": "frequency",
            "periods": total_draws,
            "frequency": frequency,
            "frequency_rate": frequency_rate,
            "most_frequent": max(frequency.items(), key=lambda x: x[1]) if frequency else None,
            "least_frequent": min(frequency.items(), key=lambda x: x[1]) if frequency else None
        }
    
    def _calculate_hot_cold_stats(self, results: List[LotteryResult]) -> Dict[str, Any]:
        """计算热冷号统计"""
        frequency = {}
        for result in results:
            for number in result.numbers:
                frequency[number] = frequency.get(number, 0) + 1
        
        # 排序并分类
        sorted_numbers = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        total_numbers = len(sorted_numbers)
        
        hot_count = total_numbers // 4
        cold_count = total_numbers // 4
        
        hot_numbers = [num for num, _ in sorted_numbers[:hot_count]]
        cold_numbers = [num for num, _ in sorted_numbers[-cold_count:]]
        normal_numbers = [num for num, _ in sorted_numbers[hot_count:-cold_count or None]]
        
        return {
            "type": "hot_cold",
            "periods": len(results),
            "hot_numbers": hot_numbers,
            "cold_numbers": cold_numbers,
            "normal_numbers": normal_numbers,
            "frequency": frequency
        }
    
    def _calculate_missing_stats(self, results: List[LotteryResult]) -> Dict[str, Any]:
        """计算遗漏统计"""
        missing_count = {}
        last_appearance = {}
        
        for i, result in enumerate(reversed(results)):
            for number in range(1, 81):
                if number in result.numbers:
                    last_appearance[number] = i
                else:
                    if number not in last_appearance:
                        missing_count[number] = missing_count.get(number, 0) + 1
        
        # 计算当前遗漏值
        current_missing = {}
        for number in range(1, 81):
            if number in last_appearance:
                current_missing[number] = len(results) - 1 - last_appearance[number]
            else:
                current_missing[number] = len(results)
        
        return {
            "type": "missing",
            "periods": len(results),
            "current_missing": current_missing,
            "max_missing": max(current_missing.values()) if current_missing else 0,
            "avg_missing": sum(current_missing.values()) / len(current_missing) if current_missing else 0
        }
    
    def _calculate_zone_stats(self, results: List[LotteryResult]) -> Dict[str, Any]:
        """计算区域统计"""
        zone_stats = {
            "zone_1": {"count": 0, "numbers": []},  # 1-20
            "zone_2": {"count": 0, "numbers": []},  # 21-40
            "zone_3": {"count": 0, "numbers": []},  # 41-60
            "zone_4": {"count": 0, "numbers": []}   # 61-80
        }
        
        for result in results:
            for number in result.numbers:
                if 1 <= number <= 20:
                    zone_stats["zone_1"]["count"] += 1
                    zone_stats["zone_1"]["numbers"].append(number)
                elif 21 <= number <= 40:
                    zone_stats["zone_2"]["count"] += 1
                    zone_stats["zone_2"]["numbers"].append(number)
                elif 41 <= number <= 60:
                    zone_stats["zone_3"]["count"] += 1
                    zone_stats["zone_3"]["numbers"].append(number)
                elif 61 <= number <= 80:
                    zone_stats["zone_4"]["count"] += 1
                    zone_stats["zone_4"]["numbers"].append(number)
        
        return {
            "type": "zone",
            "periods": len(results),
            "zone_stats": zone_stats
        }
    
    def _calculate_number_trends(self, results: List[LotteryResult], numbers: List[int]) -> Dict[str, Any]:
        """计算特定号码走势"""
        trends = {}
        for number in numbers:
            trend = []
            for result in reversed(results):
                trend.append(1 if number in result.numbers else 0)
            trends[number] = trend
        
        return {
            "type": "number_trends",
            "numbers": numbers,
            "periods": len(results),
            "trends": trends
        }
    
    def _calculate_overall_trends(self, results: List[LotteryResult]) -> Dict[str, Any]:
        """计算整体走势"""
        trends = {
            "sum_values": [],
            "odd_counts": [],
            "even_counts": [],
            "big_counts": [],
            "small_counts": []
        }
        
        for result in reversed(results):
            trends["sum_values"].append(result.sum_value)
            trends["odd_counts"].append(result.odd_count)
            trends["even_counts"].append(result.even_count)
            trends["big_counts"].append(result.big_count)
            trends["small_counts"].append(result.small_count)
        
        return {
            "type": "overall_trends",
            "periods": len(results),
            "trends": trends
        }
    
    async def _fetch_latest_from_source(self, latest_result: Optional[LotteryResult]) -> List[Dict[str, Any]]:
        """从数据源获取最新数据"""
        try:
            project_root = Path(__file__).resolve().parents[5]
            src_path = project_root / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from happy8_analyzer import Happy8Crawler

            crawler = Happy8Crawler()
            raw_results = await asyncio.to_thread(crawler.crawl_recent_data, 200)
            if not raw_results:
                return []

            latest_issue = latest_result.issue if latest_result else None
            new_results = []

            for item in raw_results:
                issue = str(item.issue)
                if latest_issue and issue <= latest_issue:
                    continue

                draw_dt = self._parse_draw_datetime(item.date, getattr(item, "time", "00:00:00"))
                numbers = [int(num) for num in item.numbers]
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                even_count = len(numbers) - odd_count
                big_count = sum(1 for n in numbers if n >= 41)
                small_count = len(numbers) - big_count
                zone_distribution = {
                    "zone_1": sum(1 for n in numbers if 1 <= n <= 20),
                    "zone_2": sum(1 for n in numbers if 21 <= n <= 40),
                    "zone_3": sum(1 for n in numbers if 41 <= n <= 60),
                    "zone_4": sum(1 for n in numbers if 61 <= n <= 80),
                }

                new_results.append(
                    {
                        "issue": issue,
                        "draw_date": draw_dt,
                        "numbers": numbers,
                        "sum_value": sum(numbers),
                        "odd_count": odd_count,
                        "even_count": even_count,
                        "big_count": big_count,
                        "small_count": small_count,
                        "zone_distribution": zone_distribution,
                    }
                )

            # 按期号升序返回，保证落库顺序稳定
            return sorted(new_results, key=lambda x: x["issue"])
        except Exception as e:
            logger.error(f"从数据源获取数据失败: {str(e)}")
            return []

    @staticmethod
    def _parse_draw_datetime(draw_date: str, draw_time: str) -> datetime:
        """解析开奖日期时间，兼容仅日期和完整时间两种格式。"""
        date_value = (draw_date or "").strip()
        time_value = (draw_time or "00:00:00").strip()
        candidates = [f"{date_value} {time_value}".strip(), date_value]

        for text in candidates:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    continue

        # 最后兜底，使用当前时间避免同步中断
        return datetime.now()
