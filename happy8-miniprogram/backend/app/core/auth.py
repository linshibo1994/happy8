"""JWT认证工具"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.exceptions import BusinessException
from app.core.logging import get_logger

logger = get_logger("app.auth")

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class JWTManager:
    """JWT管理器"""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建访问令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"创建访问令牌成功，用户ID: {data.get('sub')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"创建访问令牌失败: {e}")
            raise BusinessException.validation_error("令牌创建失败")
    
    def create_refresh_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """创建刷新令牌"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            # 刷新令牌有效期为30天
            expire = datetime.utcnow() + timedelta(days=30)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            logger.debug(f"创建刷新令牌成功，用户ID: {data.get('sub')}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"创建刷新令牌失败: {e}")
            raise BusinessException.validation_error("刷新令牌创建失败")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # 检查令牌类型
            token_type = payload.get("type")
            if not token_type:
                raise BusinessException.token_invalid("令牌类型无效")
            
            # 检查过期时间
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise BusinessException.token_expired("令牌已过期")
            
            logger.debug(f"令牌验证成功，用户ID: {payload.get('sub')}")
            return payload
            
        except JWTError as e:
            logger.warning(f"令牌验证失败: {e}")
            raise BusinessException.token_invalid("令牌无效")
        except Exception as e:
            logger.error(f"令牌验证异常: {e}")
            raise BusinessException.token_invalid("令牌验证失败")
    
    def get_user_id_from_token(self, token: str) -> int:
        """从令牌中获取用户ID"""
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise BusinessException.token_invalid("令牌中缺少用户信息")
        
        try:
            return int(user_id)
        except ValueError:
            raise BusinessException.token_invalid("用户ID格式无效")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """使用刷新令牌获取新的访问令牌"""
        payload = self.verify_token(refresh_token)
        
        # 检查是否为刷新令牌
        if payload.get("type") != "refresh":
            raise BusinessException.token_invalid("不是有效的刷新令牌")
        
        user_id = payload.get("sub")
        if not user_id:
            raise BusinessException.token_invalid("刷新令牌中缺少用户信息")
        
        # 创建新的访问令牌
        access_token_data = {"sub": user_id}
        return self.create_access_token(access_token_data)


class PasswordManager:
    """密码管理器"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """加密密码"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_random_password(length: int = 12) -> str:
        """生成随机密码"""
        import random
        import string
        
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(random.choice(characters) for _ in range(length))


class TokenBlacklist:
    """令牌黑名单管理"""
    
    def __init__(self, cache_service):
        self.cache = cache_service
        self.blacklist_prefix = "token_blacklist"
    
    async def add_to_blacklist(self, token: str, expire_time: Optional[int] = None) -> bool:
        """将令牌添加到黑名单"""
        key = f"{self.blacklist_prefix}:{token}"
        
        if not expire_time:
            # 默认过期时间为令牌的剩余有效期
            try:
                jwt_manager = JWTManager()
                payload = jwt_manager.verify_token(token)
                exp = payload.get("exp")
                if exp:
                    expire_time = max(0, int(exp - datetime.utcnow().timestamp()))
                else:
                    expire_time = 3600  # 默认1小时
            except:
                expire_time = 3600
        
        result = await self.cache.set(key, "blacklisted", expire=expire_time)
        if result:
            logger.info(f"令牌已加入黑名单: {token[:20]}...")
        return result
    
    async def is_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        key = f"{self.blacklist_prefix}:{token}"
        return await self.cache.exists(key)
    
    async def remove_from_blacklist(self, token: str) -> bool:
        """从黑名单中移除令牌"""
        key = f"{self.blacklist_prefix}:{token}"
        result = await self.cache.delete(key)
        if result:
            logger.info(f"令牌已从黑名单移除: {token[:20]}...")
        return result


# 全局JWT管理器实例
jwt_manager = JWTManager()


def get_token_blacklist():
    """获取令牌黑名单实例（延迟初始化以避免循环导入）"""
    from app.core.cache import cache_service
    return TokenBlacklist(cache_service)


# 全局令牌黑名单实例（延迟初始化）
token_blacklist = None


def init_token_blacklist():
    """初始化令牌黑名单"""
    global token_blacklist
    if token_blacklist is None:
        from app.core.cache import cache_service
        token_blacklist = TokenBlacklist(cache_service)


def create_token_pair(user_id: int) -> Dict[str, str]:
    """创建令牌对（访问令牌 + 刷新令牌）"""
    token_data = {"sub": str(user_id)}
    
    access_token = jwt_manager.create_access_token(token_data)
    refresh_token = jwt_manager.create_refresh_token(token_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

def create_access_token(data: Dict[str, Any]) -> str:
    """创建访问令牌（便捷函数）"""
    return jwt_manager.create_access_token(data)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """创建刷新令牌（便捷函数）"""
    return jwt_manager.create_refresh_token(data)

def verify_token(token: str) -> Dict[str, Any]:
    """验证令牌（便捷函数）"""
    return jwt_manager.verify_token(token)

def verify_refresh_token(refresh_token: str) -> Dict[str, Any]:
    """验证刷新令牌"""
    payload = jwt_manager.verify_token(refresh_token)
    
    # 检查是否为刷新令牌
    if payload.get("type") != "refresh":
        raise BusinessException.token_invalid("不是有效的刷新令牌")
    
    return payload

def get_user_id_from_token(token: str) -> int:
    """从令牌中获取用户ID（便捷函数）"""
    return jwt_manager.get_user_id_from_token(token)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（便捷函数）"""
    return PasswordManager.verify_password(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """加密密码（便捷函数）"""
    return PasswordManager.hash_password(password)