from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from pathlib import Path
import sys
import re
from app.models.prediction import LotteryResult
import logging
import asyncio
import httpx

logger = logging.getLogger(__name__)

class LotteryService:
    def __init__(self, db: Session):
        self.db = db
        self.public_source_timeout = 5.0
    
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
            query = self._build_history_query(start_date, end_date, issue)
            
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

    async def count_historical_results(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None
    ) -> int:
        """获取历史开奖结果真实总数，供分页组件使用。"""
        try:
            query = self._build_history_query(start_date, end_date, issue)
            return query.count()
        except Exception as e:
            logger.error(f"统计历史开奖结果总数失败: {str(e)}")
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
                    try:
                        with self.db.begin_nested():
                            lottery_result = LotteryResult(**result_data)
                            self.db.add(lottery_result)
                            self.db.flush()
                        saved_count += 1
                    except IntegrityError:
                        # 多进程部署时可能并发写入同一期，依赖唯一索引兜底并跳过。
                        logger.warning("期号 %s 已由其他同步任务写入，跳过", result_data["issue"])
            
            self.db.commit()
            return saved_count
            
        except Exception as e:
            logger.error(f"同步最新数据失败: {str(e)}")
            self.db.rollback()
            raise

    async def sync_latest_data_with_summary(self) -> Dict[str, Any]:
        """同步最新开奖数据并返回同步摘要。"""
        updated_count = await self.sync_latest_data()
        latest_result = await self.get_latest_result()

        return {
            "updated_count": updated_count,
            "latest_result": latest_result,
            "synced_at": datetime.now().isoformat(),
        }

    async def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """获取单条最新开奖结果。"""
        result = (
            self.db.query(LotteryResult)
            .order_by(LotteryResult.draw_date.desc())
            .first()
        )
        if not result:
            return None

        return self._format_lottery_result(result)
    
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
            "zone_distribution": result.zone_distribution,
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
        """从公开数据源获取最新数据，失败时降级到本地引擎抓取器。"""
        latest_issue = latest_result.issue if latest_result else None

        public_results = await self._fetch_from_public_sources(latest_issue)
        if public_results:
            return public_results

        return await self._fetch_from_engine_crawler(latest_result)

    async def _fetch_from_public_sources(self, latest_issue: Optional[str]) -> List[Dict[str, Any]]:
        """尝试从外部公开来源拉取快乐8最新/历史数据。"""
        source_urls = [
            "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
            "?name=kl8&issueCount=200",
            "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
            "?name=kl8&pageNo=1&pageSize=200&systemType=PC",
        ]

        headers = {
            "User-Agent": "Happy8Backend/1.0 (+https://www.cwl.gov.cn/)",
            "Referer": "https://www.cwl.gov.cn/",
            "Accept": "application/json,text/plain,*/*",
        }

        for url in source_urls:
            try:
                async with httpx.AsyncClient(timeout=self.public_source_timeout) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    payload = response.json()

                parsed = self._parse_public_source_payload(payload, latest_issue)
                if parsed:
                    logger.info("从外部公开来源同步快乐8数据成功: %s 条", len(parsed))
                    return parsed
            except Exception as exc:
                logger.warning("外部公开来源拉取失败，已准备降级: %s", exc)

        return []

    async def _fetch_from_engine_crawler(self, latest_result: Optional[LotteryResult]) -> List[Dict[str, Any]]:
        """使用项目既有引擎抓取器作为降级数据源。"""
        try:
            project_root = Path(__file__).resolve().parents[3]
            engine_path = project_root / "engine"
            if str(engine_path) not in sys.path:
                sys.path.insert(0, str(engine_path))

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
            logger.error(f"从降级数据源获取数据失败: {str(e)}")
            return []

    def _build_history_query(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        issue: Optional[str] = None
    ):
        """构建历史列表查询，保证列表与总数筛选条件一致。"""
        query = self.db.query(LotteryResult)

        if start_date:
            query = query.filter(LotteryResult.draw_date >= start_date)
        if end_date:
            query = query.filter(LotteryResult.draw_date <= end_date)
        if issue:
            query = query.filter(LotteryResult.issue.like(f"%{issue}%"))

        return query

    def _parse_public_source_payload(
        self,
        payload: Dict[str, Any],
        latest_issue: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """解析公开接口返回，兼容中国福彩网常见字段命名。"""
        rows = self._extract_public_rows(payload)
        results = []

        for row in rows:
            normalized = self._normalize_public_row(row)
            if not normalized:
                continue
            if latest_issue and normalized["issue"] <= latest_issue:
                continue
            results.append(normalized)

        return sorted(results, key=lambda x: x["issue"])

    def _extract_public_rows(self, payload: Any) -> List[Dict[str, Any]]:
        """从多种公开接口结构中提取开奖列表。"""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        for key in ("result", "data", "rows", "list"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = self._extract_public_rows(value)
                if nested:
                    return nested

        return []

    def _normalize_public_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将公开来源单期开奖数据标准化为 LotteryResult 入库结构。"""
        issue = str(
            row.get("code")
            or row.get("issue")
            or row.get("draw_num")
            or row.get("drawNum")
            or ""
        ).strip()
        if not issue:
            return None

        numbers = self._parse_public_numbers(
            row.get("red")
            or row.get("numbers")
            or row.get("openCode")
            or row.get("open_code")
            or row.get("result")
        )
        if len(numbers) != 20:
            return None

        draw_date = str(
            row.get("date")
            or row.get("drawDate")
            or row.get("draw_date")
            or row.get("time")
            or ""
        )
        draw_time = str(row.get("week") or row.get("drawTime") or "00:00:00")
        draw_dt = self._parse_draw_datetime(draw_date[:10], draw_time)

        return self._build_result_payload(issue, draw_dt, numbers)

    @staticmethod
    def _parse_public_numbers(raw_numbers: Any) -> List[int]:
        """解析公开来源中的开奖号码。"""
        if isinstance(raw_numbers, list):
            values = raw_numbers
        elif isinstance(raw_numbers, str):
            values = re.findall(r"\d+", raw_numbers)
        else:
            values = []

        numbers = []
        for value in values:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= 80:
                numbers.append(number)

        return numbers

    @staticmethod
    def _build_result_payload(issue: str, draw_dt: datetime, numbers: List[int]) -> Dict[str, Any]:
        """构造统一的开奖结果入库字典。"""
        numbers = [int(num) for num in numbers]
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

        return {
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
