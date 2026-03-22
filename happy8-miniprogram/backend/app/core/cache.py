"""Redis缓存管理"""

import json
import pickle
from typing import Any, Optional, Union, Dict, List
from datetime import timedelta
import redis.asyncio as aioredis
import redis
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("app.cache")


class RedisManager:
    """Redis管理器"""
    
    def __init__(self):
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._async_redis_pool: Optional[aioredis.ConnectionPool] = None
        self._redis_client: Optional[redis.Redis] = None
        self._async_redis_client: Optional[aioredis.Redis] = None
    
    def get_redis_pool(self) -> redis.ConnectionPool:
        """获取同步Redis连接池"""
        if self._redis_pool is None:
            self._redis_pool = redis.ConnectionPool.from_url(
                str(settings.REDIS_URL),
                max_connections=settings.REDIS_POOL_SIZE,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                health_check_interval=30,
                decode_responses=True
            )
        return self._redis_pool
    
    def get_async_redis_pool(self) -> aioredis.ConnectionPool:
        """获取异步Redis连接池"""
        if self._async_redis_pool is None:
            self._async_redis_pool = aioredis.ConnectionPool.from_url(
                str(settings.REDIS_URL),
                max_connections=settings.REDIS_POOL_SIZE,
                socket_timeout=settings.REDIS_TIMEOUT,
                socket_connect_timeout=settings.REDIS_TIMEOUT,
                health_check_interval=30,
                decode_responses=True
            )
        return self._async_redis_pool
    
    def get_redis_client(self) -> redis.Redis:
        """获取同步Redis客户端"""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                connection_pool=self.get_redis_pool()
            )
        return self._redis_client
    
    def get_async_redis_client(self) -> aioredis.Redis:
        """获取异步Redis客户端"""
        if self._async_redis_client is None:
            self._async_redis_client = aioredis.Redis(
                connection_pool=self.get_async_redis_pool()
            )
        return self._async_redis_client
    
    async def close(self):
        """关闭Redis连接"""
        if self._async_redis_client:
            await self._async_redis_client.close()
            logger.info("异步Redis客户端已关闭")
        
        if self._redis_client:
            self._redis_client.close()
            logger.info("同步Redis客户端已关闭")


# 全局Redis管理器实例
redis_manager = RedisManager()


class CacheService:
    """缓存服务"""
    
    def __init__(self):
        self.redis_client = redis_manager.get_async_redis_client()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # 尝试JSON反序列化
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # 如果JSON反序列化失败，尝试pickle
                try:
                    return pickle.loads(value.encode())
                except:
                    # 都失败则返回原始字符串
                    return value
                    
        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """设置缓存值"""
        try:
            # 序列化值
            if isinstance(value, (dict, list, tuple)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, str, bool)):
                serialized_value = json.dumps(value)
            else:
                # 复杂对象使用pickle
                serialized_value = pickle.dumps(value).decode('latin1')
            
            # 设置过期时间
            if isinstance(expire, timedelta):
                expire = int(expire.total_seconds())
            
            result = await self.redis_client.set(key, serialized_value, ex=expire)
            if result:
                logger.debug(f"缓存设置成功 {key}")
            return result
            
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            result = await self.redis_client.delete(key)
            if result:
                logger.debug(f"缓存删除成功 {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """设置缓存过期时间"""
        try:
            result = await self.redis_client.expire(key, seconds)
            return bool(result)
        except Exception as e:
            logger.error(f"设置缓存过期时间失败 {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """获取缓存剩余生存时间"""
        try:
            result = await self.redis_client.ttl(key)
            return result
        except Exception as e:
            logger.error(f"获取缓存TTL失败 {key}: {e}")
            return -1
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的缓存键"""
        try:
            result = await self.redis_client.keys(pattern)
            return result
        except Exception as e:
            logger.error(f"获取缓存键失败 {pattern}: {e}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        try:
            keys = await self.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                logger.info(f"清除缓存 {pattern}: {result}个键")
                return result
            return 0
        except Exception as e:
            logger.error(f"清除缓存模式失败 {pattern}: {e}")
            return 0
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"递增计数器失败 {key}: {e}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        try:
            result = await self.redis_client.decrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"递减计数器失败 {key}: {e}")
            return 0


class CacheKeyManager:
    """缓存键管理器"""
    
    # 键前缀
    USER_PREFIX = "user"
    PREDICTION_PREFIX = "prediction"
    LOTTERY_PREFIX = "lottery"
    SESSION_PREFIX = "session"
    RATE_LIMIT_PREFIX = "rate_limit"
    TEMP_PREFIX = "temp"
    
    @classmethod
    def user_key(cls, user_id: int) -> str:
        """用户缓存键"""
        return f"{cls.USER_PREFIX}:{user_id}"
    
    @classmethod
    def user_membership_key(cls, user_id: int) -> str:
        """用户会员信息缓存键"""
        return f"{cls.USER_PREFIX}:{user_id}:membership"
    
    @classmethod
    def prediction_key(cls, algorithm: str, target_issue: str, periods: int, count: int) -> str:
        """预测结果缓存键"""
        return f"{cls.PREDICTION_PREFIX}:{algorithm}:{target_issue}:{periods}:{count}"
    
    @classmethod
    def lottery_results_key(cls, limit: int = 100) -> str:
        """彩票结果缓存键"""
        return f"{cls.LOTTERY_PREFIX}:results:{limit}"
    
    @classmethod
    def session_key(cls, token: str) -> str:
        """会话缓存键"""
        return f"{cls.SESSION_PREFIX}:{token}"
    
    @classmethod
    def rate_limit_key(cls, identifier: str, action: str) -> str:
        """频率限制缓存键"""
        return f"{cls.RATE_LIMIT_PREFIX}:{identifier}:{action}"
    
    @classmethod
    def temp_key(cls, identifier: str) -> str:
        """临时数据缓存键"""
        return f"{cls.TEMP_PREFIX}:{identifier}"
    
    @classmethod
    def verification_code_key(cls, phone: str) -> str:
        """验证码缓存键"""
        return f"{cls.TEMP_PREFIX}:sms:{phone}"
    
    @classmethod
    def wechat_access_token_key(cls) -> str:
        """微信访问令牌缓存键"""
        return f"{cls.TEMP_PREFIX}:wechat:access_token"


class RateLimiter:
    """频率限制器"""
    
    def __init__(self, cache_service: CacheService):
        self.cache = cache_service
    
    async def is_allowed(
        self,
        identifier: str,
        action: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """检查是否允许访问"""
        key = CacheKeyManager.rate_limit_key(identifier, action)
        
        try:
            current_count = await self.cache.get(key) or 0
            
            if current_count >= max_requests:
                ttl = await self.cache.ttl(key)
                return False, ttl
            
            # 递增计数器
            new_count = await self.cache.incr(key)
            
            # 如果是第一次请求，设置过期时间
            if new_count == 1:
                await self.cache.expire(key, window_seconds)
            
            return True, window_seconds
            
        except Exception as e:
            logger.error(f"频率限制检查失败: {e}")
            # 出错时允许访问
            return True, 0


class HealthCheck:
    """Redis健康检查"""
    
    @staticmethod
    async def check_connection() -> bool:
        """检查Redis连接"""
        try:
            client = redis_manager.get_async_redis_client()
            result = await client.ping()
            return result
        except Exception as e:
            logger.error(f"Redis连接检查失败: {e}")
            return False
    
    @staticmethod
    async def get_info() -> Dict[str, Any]:
        """获取Redis信息"""
        try:
            client = redis_manager.get_async_redis_client()
            info = await client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        except Exception as e:
            logger.error(f"获取Redis信息失败: {e}")
            return {"error": str(e)}


# 全局缓存服务实例
cache_service = CacheService()


async def get_cache() -> CacheService:
    """获取缓存服务（用于依赖注入）"""
    return cache_service


def init_redis():
    """初始化Redis连接"""
    try:
        # 初始化连接池
        redis_manager.get_redis_pool()
        redis_manager.get_async_redis_pool()
        logger.info("Redis连接池初始化成功")
    except Exception as e:
        logger.error(f"Redis初始化失败: {e}")
        raise


async def close_redis():
    """关闭Redis连接"""
    await redis_manager.close()