"""预测服务 - 完全集成原始Happy8算法"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.prediction import LotteryResult, PredictionHistory, AlgorithmConfig
from app.models.membership import MembershipLevel
from app.core.exceptions import BusinessException
from app.core.cache import CacheService, CacheKeyManager
from app.core.logging import prediction_logger as logger
from app.services.membership_service import MembershipService
from app.services.happy8_algorithm_adapter import Happy8AlgorithmAdapter


class PredictionService:
    """预测服务 - 基于原始Happy8算法"""
    
    def __init__(self, db: Session, cache: CacheService):
        self.db = db
        self.cache = cache
        self.membership_service = MembershipService(db, cache)
        
        # 初始化原始Happy8算法适配器
        self.algorithm_adapter = Happy8AlgorithmAdapter()
        
        # 检查原始算法可用性
        if not self.algorithm_adapter.is_original_available():
            logger.error("原始Happy8算法不可用，预测功能将受限")
        else:
            available_algorithms = self.algorithm_adapter.get_all_available_algorithms()
            logger.info(f"成功加载原始Happy8算法: {available_algorithms}")
        
        # 算法映射表（将我们的API算法名映射到原始算法名）
        self.algorithm_mapping = {
            "frequency": "frequency",
            "hot_cold": "hot_cold", 
            "missing": "missing",  # 如果原始没有，适配器会创建
            "markov": "adaptive_markov",  # 优先使用自适应马尔可夫
            "ml_ensemble": "advanced_ensemble",  # 优先使用高级集成
            "deep_learning": "transformer",  # 优先使用Transformer
            "super_predictor": "super_predictor",
            # 扩展算法
            "markov_basic": "markov",
            "markov_2nd": "markov_2nd", 
            "markov_3rd": "markov_3rd",
            "adaptive_markov": "adaptive_markov",
            "lstm": "lstm",
            "transformer": "transformer",
            "gnn": "gnn",
            "bayesian": "bayesian",
            "monte_carlo": "monte_carlo",
            "clustering": "clustering",
            "ensemble_basic": "ensemble",
            "ensemble": "ensemble",
            "advanced_ensemble": "advanced_ensemble",
            "high_confidence": "high_confidence"
        }
        
        # 算法显示名称
        self.algorithm_display_names = {
            "frequency": "频率分析",
            "hot_cold": "冷热分析",
            "missing": "遗漏分析",
            "markov": "自适应马尔可夫链",
            "ml_ensemble": "高级机器学习集成",
            "deep_learning": "Transformer深度学习",
            "super_predictor": "超级预测器",
            "markov_basic": "基础马尔可夫链",
            "markov_2nd": "二阶马尔可夫链",
            "markov_3rd": "三阶马尔可夫链",
            "adaptive_markov": "自适应马尔可夫链",
            "lstm": "LSTM深度学习",
            "transformer": "Transformer深度学习",
            "gnn": "图神经网络",
            "bayesian": "贝叶斯推理",
            "monte_carlo": "蒙特卡洛模拟",
            "clustering": "聚类分析",
            "ensemble_basic": "基础集成学习",
            "ensemble": "基础集成学习",
            "advanced_ensemble": "高级机器学习集成",
            "high_confidence": "高置信度预测器"
        }
        self._ensure_algorithm_configs()
    
    async def get_available_algorithms(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户可用的算法列表"""
        try:
            # 获取所有活跃算法配置
            algorithms = self.db.query(AlgorithmConfig).filter(
                AlgorithmConfig.is_active == True
            ).order_by(AlgorithmConfig.sort_order).all()
            
            # 检查用户权限
            available_algorithms = []
            original_algorithms = self.algorithm_adapter.get_all_available_algorithms()
            
            for algo in algorithms:
                # 检查原始算法是否真的可用
                mapped_algo = self.algorithm_mapping.get(algo.algorithm_name)
                is_algorithm_available = (
                    mapped_algo in original_algorithms or 
                    algo.algorithm_name == "missing"  # missing有特殊处理
                )
                
                if not is_algorithm_available:
                    logger.warning(f"算法 {algo.algorithm_name} 在原始系统中不可用")
                    continue
                
                # 检查用户权限
                has_permission = await self.membership_service.check_permission(
                    user_id, algo.required_level
                )
                
                # 获取算法详细信息
                algorithm_info = await self.algorithm_adapter.get_algorithm_info(mapped_algo)
                
                algorithm_data = {
                    "id": algo.id,
                    "algorithm_name": algo.algorithm_name,
                    "display_name": algo.display_name,
                    "description": algo.description,
                    "required_level": algo.required_level,
                    "has_permission": has_permission,
                    "default_params": self._parse_default_params(algo.default_params),
                    "avg_execution_time": algo.avg_execution_time,
                    "success_rate": algo.success_rate,
                    "usage_count": algo.usage_count,
                    "is_original_available": is_algorithm_available,
                    "original_algorithm": mapped_algo,
                    "complexity": algorithm_info.get("complexity", "unknown"),
                    "data_requirements": algorithm_info.get("data_requirements", {})
                }
                available_algorithms.append(algorithm_data)
            
            return available_algorithms
            
        except Exception as e:
            logger.error(f"获取可用算法失败: {e}")
            raise BusinessException.data_not_found("获取算法列表失败")

    def _ensure_algorithm_configs(self):
        """确保数据库中存在完整算法配置，兼容旧库仅7个算法的情况。"""
        try:
            from app.utils.algorithm_config_updater import update_algorithm_configs

            config_items = update_algorithm_configs()
            existing_rows = self.db.query(AlgorithmConfig).all()
            existing_by_name = {row.algorithm_name: row for row in existing_rows}
            changed = False

            for item in config_items:
                name = item["algorithm_name"]
                required_level = str(item.get("required_level", "free")).lower()
                default_params = item.get("default_params")
                row = existing_by_name.get(name)

                if row:
                    # 仅做必要的规范化修复，保留运行期统计字段
                    if row.required_level != required_level:
                        row.required_level = required_level
                        changed = True
                    if row.display_name != item.get("display_name"):
                        row.display_name = item.get("display_name")
                        changed = True
                    if row.description != item.get("description"):
                        row.description = item.get("description")
                        changed = True
                    if row.sort_order != item.get("sort_order"):
                        row.sort_order = item.get("sort_order")
                        changed = True
                    if not row.default_params and default_params is not None:
                        row.default_params = default_params
                        changed = True
                    continue

                self.db.add(
                    AlgorithmConfig(
                        algorithm_name=name,
                        display_name=item.get("display_name", name),
                        description=item.get("description"),
                        default_params=default_params,
                        required_level=required_level,
                        is_active=True,
                        sort_order=item.get("sort_order", 0),
                        usage_count=0,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                )
                changed = True

            if changed:
                self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.warning(f"自动修复算法配置失败: {exc}")
    
    async def check_prediction_limit(self, user_id: int) -> bool:
        """检查用户是否可以进行预测"""
        try:
            # 获取用户会员信息
            membership = await self.membership_service.get_user_membership(user_id)
            if not membership:
                return False
            
            # 检查会员是否有效
            if not await self.membership_service.check_membership_validity(user_id):
                return False
            
            # 检查日预测次数限制
            if membership.level == MembershipLevel.FREE:
                return membership.predictions_today < 5
            elif membership.level == MembershipLevel.VIP:
                return membership.predictions_today < 50
            else:  # PREMIUM
                return True  # 无限制
                
        except Exception as e:
            logger.error(f"检查预测限制失败: {e}")
            return False
    
    async def predict_numbers(
        self,
        user_id: int,
        algorithm: str,
        target_issue: str,
        periods: int = 30,
        count: int = 5,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行号码预测 - 使用原始Happy8算法"""
        try:
            # 检查原始算法适配器是否可用
            if not self.algorithm_adapter.is_original_available():
                raise BusinessException.prediction_failed("原始Happy8预测引擎不可用")
            
            # 检查预测权限
            if not await self.check_prediction_limit(user_id):
                raise BusinessException.prediction_limit_exceeded("预测次数已达上限")
            
            # 验证算法是否支持
            if algorithm not in self.algorithm_mapping:
                raise BusinessException.algorithm_not_found(f"不支持的算法: {algorithm}")
            
            # 检查算法权限
            algo_config = self.db.query(AlgorithmConfig).filter(
                AlgorithmConfig.algorithm_name == algorithm,
                AlgorithmConfig.is_active == True
            ).first()
            
            if not algo_config:
                raise BusinessException.algorithm_not_found("算法配置不存在")
            
            has_permission = await self.membership_service.check_permission(
                user_id, algo_config.required_level
            )
            if not has_permission:
                raise BusinessException.insufficient_permission(f"需要{algo_config.required_level}会员权限")
            
            # 检查缓存
            cache_key = CacheKeyManager.prediction_key(algorithm, target_issue, periods, count)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                logger.debug(f"使用缓存预测结果: {algorithm}, {target_issue}")
                enriched_result = dict(cached_result)
                comparison_data = await self._build_comparison_data(
                    target_issue=target_issue,
                    predicted_numbers=enriched_result.get("predicted_numbers", []),
                )
                enriched_result.update(comparison_data)
                
                # 更新用户预测次数
                await self.membership_service.update_daily_prediction_count(user_id)
                
                # 记录预测历史（标记为缓存）
                await self._save_prediction_history(
                    user_id=user_id,
                    algorithm=algorithm,
                    target_issue=target_issue,
                    periods=periods,
                    count=count,
                    predicted_numbers=enriched_result["predicted_numbers"],
                    confidence_score=enriched_result.get("confidence_score"),
                    algorithm_params=params or {},
                    is_cached=True,
                    cache_key=cache_key,
                    execution_time=0.0,
                    actual_numbers=enriched_result.get("actual_numbers"),
                    hit_count=enriched_result.get("hit_count"),
                    hit_rate=enriched_result.get("hit_rate"),
                    is_hit=enriched_result.get("is_hit"),
                    analysis_data=enriched_result.get("analysis_data")
                )
                
                return enriched_result
            
            # 执行原始预测算法
            start_time = datetime.now()
            prediction_result = await self._execute_original_prediction(
                algorithm=algorithm,
                target_issue=target_issue,
                periods=periods,
                count=count,
                params=params or {}
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 添加执行时间到结果中
            prediction_result["execution_time"] = execution_time
            prediction_result["target_issue"] = target_issue
            prediction_result["periods"] = periods
            prediction_result["algorithm"] = algorithm
            prediction_result.update(
                await self._build_comparison_data(
                    target_issue=target_issue,
                    predicted_numbers=prediction_result.get("predicted_numbers", []),
                )
            )
            
            # 缓存结果
            await self.cache.set(
                cache_key, 
                prediction_result, 
                expire=3600  # 1小时
            )
            
            # 更新用户预测次数
            await self.membership_service.update_daily_prediction_count(user_id)
            
            # 记录预测历史
            await self._save_prediction_history(
                user_id=user_id,
                algorithm=algorithm,
                target_issue=target_issue,
                periods=periods,
                count=count,
                predicted_numbers=prediction_result["predicted_numbers"],
                confidence_score=prediction_result.get("confidence_score"),
                algorithm_params=params or {},
                is_cached=False,
                cache_key=cache_key,
                execution_time=execution_time,
                actual_numbers=prediction_result.get("actual_numbers"),
                hit_count=prediction_result.get("hit_count"),
                hit_rate=prediction_result.get("hit_rate"),
                is_hit=prediction_result.get("is_hit"),
                analysis_data=prediction_result.get("analysis_data")
            )
            
            # 更新算法统计
            await self._update_algorithm_stats(algo_config, execution_time)
            
            logger.info(f"原始算法预测完成: 用户={user_id}, 算法={algorithm}, 期号={target_issue}")
            return prediction_result
            
        except BusinessException:
            raise
        except Exception as e:
            logger.error(f"预测执行失败: {e}")
            raise BusinessException.prediction_failed(f"预测执行失败: {str(e)}")
    
    async def _execute_original_prediction(
        self,
        algorithm: str,
        target_issue: str,
        periods: int,
        count: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行原始Happy8预测算法"""
        
        # 获取历史数据
        historical_data = await self._get_historical_data(periods)
        
        if len(historical_data) < 10:
            raise BusinessException.insufficient_data("历史数据不足")
        
        # 获取映射的原始算法名
        original_algorithm = self.algorithm_mapping.get(algorithm)
        if not original_algorithm:
            raise BusinessException.algorithm_not_found(f"无法映射算法: {algorithm}")
        
        logger.info(f"执行原始算法: {algorithm} -> {original_algorithm}")
        
        # 根据算法类型调用相应的适配器方法
        try:
            if algorithm == "frequency":
                return await self.algorithm_adapter.frequency_analysis(historical_data, count, params)
            elif algorithm == "hot_cold":
                return await self.algorithm_adapter.hot_cold_analysis(historical_data, count, params)
            elif algorithm == "missing":
                return await self.algorithm_adapter.missing_analysis(historical_data, count, params)
            elif algorithm in ["markov", "markov_basic"]:
                return await self.algorithm_adapter.markov_analysis(historical_data, count, params)
            elif algorithm in ["ml_ensemble", "ensemble_basic"]:
                return await self.algorithm_adapter.ml_ensemble_analysis(historical_data, count, params)
            elif algorithm == "deep_learning":
                return await self.algorithm_adapter.deep_learning_analysis(historical_data, count, params)
            elif algorithm == "super_predictor":
                return await self.algorithm_adapter.super_predictor_analysis(historical_data, count, params)
            else:
                # 通用调用
                return await self.algorithm_adapter.execute_original_algorithm(original_algorithm, historical_data, count, params)
                
        except Exception as e:
            logger.error(f"原始算法 {algorithm} 执行失败: {e}")
            raise BusinessException.prediction_failed(f"算法 {algorithm} 执行失败: {str(e)}")
    
    async def _get_historical_data(self, periods: int) -> List[Dict[str, Any]]:
        """获取历史开奖数据"""
        try:
            results = self.db.query(LotteryResult).order_by(
                LotteryResult.draw_date.desc()
            ).limit(periods).all()
            
            data = []
            for result in results:
                data.append({
                    "issue": result.issue,
                    "date": result.draw_date.strftime("%Y-%m-%d"),
                    "numbers": result.numbers,
                    "sum_value": result.sum_value,
                    "odd_count": result.odd_count,
                    "big_count": result.big_count
                })
            
            return list(reversed(data))  # 按时间正序返回
            
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            raise BusinessException.data_not_found("获取历史数据失败")
    
    async def _save_prediction_history(
        self,
        user_id: int,
        algorithm: str,
        target_issue: str,
        periods: int,
        count: int,
        predicted_numbers: List[int],
        confidence_score: Optional[float],
        algorithm_params: Dict[str, Any],
        is_cached: bool,
        cache_key: str,
        execution_time: float,
        actual_numbers: Optional[List[int]] = None,
        hit_count: Optional[int] = None,
        hit_rate: Optional[float] = None,
        is_hit: Optional[bool] = None,
        analysis_data: Optional[Dict[str, Any]] = None,
    ):
        """保存预测历史记录"""
        try:
            history = PredictionHistory(
                user_id=user_id,
                algorithm=algorithm,
                target_issue=target_issue,
                periods=periods,
                count=count,
                predicted_numbers=predicted_numbers,
                confidence_score=confidence_score,
                algorithm_params=algorithm_params,
                actual_numbers=actual_numbers,
                hit_count=hit_count,
                hit_rate=hit_rate,
                is_hit=is_hit,
                analysis_data=analysis_data,
                is_cached=is_cached,
                cache_key=cache_key,
                execution_time=execution_time,
                created_at=datetime.now()
            )
            
            self.db.add(history)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"保存预测历史失败: {e}")

    @staticmethod
    def _parse_default_params(default_params: Any) -> Dict[str, Any]:
        """兼容JSON字符串和字典两种配置格式。"""
        if not default_params:
            return {}
        if isinstance(default_params, dict):
            return default_params
        if isinstance(default_params, str):
            try:
                return json.loads(default_params)
            except Exception:
                return {}
        return {}

    async def _build_comparison_data(
        self, target_issue: str, predicted_numbers: List[int]
    ) -> Dict[str, Any]:
        """如果目标期号已有开奖数据，计算命中统计。"""
        if not predicted_numbers:
            return {}

        actual_result = (
            self.db.query(LotteryResult).filter(LotteryResult.issue == target_issue).first()
        )
        if not actual_result:
            return {}

        actual_numbers = actual_result.numbers or []
        hit_numbers = sorted(set(predicted_numbers).intersection(actual_numbers))
        hit_count = len(hit_numbers)
        hit_rate = hit_count / len(predicted_numbers) if predicted_numbers else 0.0

        return {
            "actual_numbers": actual_numbers,
            "hit_numbers": hit_numbers,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "is_hit": hit_count > 0,
        }
    
    async def _update_algorithm_stats(self, algo_config: AlgorithmConfig, execution_time: float):
        """更新算法统计信息"""
        try:
            algo_config.usage_count = (algo_config.usage_count or 0) + 1
            
            # 更新平均执行时间
            if algo_config.avg_execution_time:
                algo_config.avg_execution_time = (
                    algo_config.avg_execution_time * (algo_config.usage_count - 1) + execution_time
                ) / algo_config.usage_count
            else:
                algo_config.avg_execution_time = execution_time
            
            algo_config.updated_at = datetime.now()
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新算法统计失败: {e}")
    
    async def get_prediction_history(
        self,
        user_id: int,
        algorithm: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取用户预测历史"""
        try:
            query = self.db.query(PredictionHistory).filter(
                PredictionHistory.user_id == user_id
            )
            
            if algorithm:
                query = query.filter(PredictionHistory.algorithm == algorithm)
            
            histories = query.order_by(
                PredictionHistory.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            result = []
            for history in histories:
                history_data = {
                    "id": history.id,
                    "algorithm": history.algorithm,
                    "algorithm_display_name": self.algorithm_display_names.get(history.algorithm, history.algorithm),
                    "target_issue": history.target_issue,
                    "periods": history.periods,
                    "count": history.count,
                    "predicted_numbers": history.predicted_numbers,
                    "confidence_score": history.confidence_score,
                    "actual_numbers": history.actual_numbers,
                    "hit_count": history.hit_count,
                    "hit_rate": history.hit_rate,
                    "is_hit": history.is_hit,
                    "execution_time": history.execution_time,
                    "is_cached": history.is_cached,
                    "created_at": history.created_at.isoformat(),
                    "algorithm_params": history.algorithm_params
                }
                result.append(history_data)
            
            return result
            
        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            raise BusinessException.data_not_found("获取预测历史失败")
    
    async def get_latest_lottery_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最新开奖结果"""
        try:
            # 先从缓存获取
            cache_key = CacheKeyManager.lottery_results_key(limit)
            cached_results = await self.cache.get(cache_key)
            
            if cached_results:
                return cached_results
            
            # 从数据库获取
            results = self.db.query(LotteryResult).order_by(
                LotteryResult.draw_date.desc()
            ).limit(limit).all()
            
            data = []
            for result in results:
                data.append({
                    "id": result.id,
                    "issue": result.issue,
                    "draw_date": result.draw_date.isoformat(),
                    "numbers": result.numbers,
                    "sum_value": result.sum_value,
                    "odd_count": result.odd_count,
                    "even_count": result.even_count,
                    "big_count": result.big_count,
                    "small_count": result.small_count,
                    "zone_distribution": result.zone_distribution
                })
            
            # 缓存结果
            await self.cache.set(cache_key, data, expire=1800)  # 30分钟
            
            return data
            
        except Exception as e:
            logger.error(f"获取开奖结果失败: {e}")
            raise BusinessException.data_not_found("获取开奖结果失败")
    
    async def get_algorithm_diagnostics(self) -> Dict[str, Any]:
        """获取算法诊断信息"""
        try:
            diagnostics = {
                "adapter_available": self.algorithm_adapter.is_original_available(),
                "original_algorithms": self.algorithm_adapter.get_all_available_algorithms(),
                "mapped_algorithms": self.algorithm_mapping,
                "algorithm_status": {}
            }
            
            # 检查每个算法的状态
            for api_algo, original_algo in self.algorithm_mapping.items():
                try:
                    algo_info = await self.algorithm_adapter.get_algorithm_info(original_algo)
                    diagnostics["algorithm_status"][api_algo] = {
                        "original_algorithm": original_algo,
                        "available": algo_info.get("available", False),
                        "error": algo_info.get("error"),
                        "complexity": algo_info.get("complexity"),
                        "data_requirements": algo_info.get("data_requirements")
                    }
                except Exception as e:
                    diagnostics["algorithm_status"][api_algo] = {
                        "original_algorithm": original_algo,
                        "available": False,
                        "error": str(e)
                    }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"获取算法诊断信息失败: {e}")
            return {"error": str(e)}
