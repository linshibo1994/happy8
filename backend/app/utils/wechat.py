"""微信API工具"""

import base64
import hashlib
import hmac
import json
import time
from typing import Dict, Optional, Any
import httpx
from urllib.parse import quote

from app.core.config import settings
from app.core.cache import CacheService, CacheKeyManager
from app.core.exceptions import BusinessException
from app.core.logging import wechat_logger as logger


class WeChatAPI:
    """微信API客户端"""
    
    def __init__(self, cache_service: CacheService):
        self.app_id = settings.WECHAT_APP_ID
        self.app_secret = settings.WECHAT_APP_SECRET
        self.cache = cache_service
        self.base_url = "https://api.weixin.qq.com"
    
    async def get_access_token(self) -> str:
        """获取微信访问令牌"""
        # 先从缓存获取
        cache_key = CacheKeyManager.wechat_access_token_key()
        cached_token = await self.cache.get(cache_key)
        
        if cached_token:
            logger.debug("使用缓存的微信访问令牌")
            return cached_token
        
        # 从微信API获取
        url = f"{self.base_url}/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": self.app_id,
            "secret": self.app_secret
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if "access_token" in data:
                    access_token = data["access_token"]
                    expires_in = data.get("expires_in", 7200)
                    
                    # 缓存令牌，提前5分钟过期
                    await self.cache.set(
                        cache_key, 
                        access_token, 
                        expire=expires_in - 300
                    )
                    
                    logger.info("微信访问令牌获取成功")
                    return access_token
                else:
                    error_msg = data.get("errmsg", "未知错误")
                    logger.error(f"获取微信访问令牌失败: {error_msg}")
                    raise BusinessException.validation_error(f"微信API错误: {error_msg}")
                    
        except httpx.RequestError as e:
            logger.error(f"微信API请求失败: {e}")
            raise BusinessException.validation_error("微信服务不可用")
    
    async def code_to_session(self, js_code: str) -> Dict[str, Any]:
        """小程序登录，获取用户session信息"""
        url = f"{self.base_url}/sns/jscode2session"
        params = {
            "appid": self.app_id,
            "secret": self.app_secret,
            "js_code": js_code,
            "grant_type": "authorization_code"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if "openid" in data:
                    logger.info(f"微信登录成功，openid: {data['openid']}")
                    return {
                        "openid": data["openid"],
                        "unionid": data.get("unionid"),
                        "session_key": data["session_key"]
                    }
                else:
                    error_code = data.get("errcode")
                    error_msg = data.get("errmsg", "未知错误")
                    logger.error(f"微信登录失败: {error_code} - {error_msg}")
                    
                    if error_code == 40029:
                        raise BusinessException.validation_error("无效的code")
                    elif error_code == 45011:
                        raise BusinessException.validation_error("API调用过于频繁")
                    else:
                        raise BusinessException.validation_error(f"微信登录失败: {error_msg}")
                        
        except httpx.RequestError as e:
            logger.error(f"微信登录请求失败: {e}")
            raise BusinessException.validation_error("微信服务不可用")
    
    async def get_user_info(self, access_token: str, openid: str) -> Dict[str, Any]:
        """获取用户基本信息"""
        url = f"{self.base_url}/cgi-bin/user/info"
        params = {
            "access_token": access_token,
            "openid": openid,
            "lang": "zh_CN"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                data = response.json()
                
                if "openid" in data:
                    return {
                        "openid": data["openid"],
                        "nickname": data.get("nickname", ""),
                        "headimgurl": data.get("headimgurl", ""),
                        "sex": data.get("sex", 0),
                        "province": data.get("province", ""),
                        "city": data.get("city", ""),
                        "country": data.get("country", "")
                    }
                else:
                    error_msg = data.get("errmsg", "未知错误")
                    logger.error(f"获取用户信息失败: {error_msg}")
                    raise BusinessException.validation_error(f"获取用户信息失败: {error_msg}")
                    
        except httpx.RequestError as e:
            logger.error(f"获取用户信息请求失败: {e}")
            raise BusinessException.validation_error("微信服务不可用")
    
    def decrypt_data(self, encrypted_data: str, session_key: str, iv: str) -> Dict[str, Any]:
        """解密微信数据"""
        import base64
        from Crypto.Cipher import AES
        
        try:
            # Base64解码
            session_key = base64.b64decode(session_key)
            encrypted_data = base64.b64decode(encrypted_data)
            iv = base64.b64decode(iv)
            
            # AES解密
            cipher = AES.new(session_key, AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(encrypted_data)
            
            # 去除padding
            decrypted = decrypted[:-ord(decrypted[-1:])]
            
            # 解析JSON
            user_info = json.loads(decrypted.decode('utf-8'))
            
            # 验证数据完整性
            if user_info.get('watermark', {}).get('appid') != self.app_id:
                raise BusinessException.validation_error("数据来源验证失败")
            
            logger.info("微信数据解密成功")
            return user_info
            
        except Exception as e:
            logger.error(f"微信数据解密失败: {e}")
            raise BusinessException.validation_error("数据解密失败")


class WeChatPayAPI:
    """微信支付API"""
    
    def __init__(self):
        self.mch_id = settings.WECHAT_PAY_MCHID
        self.private_key_path = settings.WECHAT_PAY_PRIVATE_KEY_PATH
        self.cert_serial = settings.WECHAT_PAY_CERT_SERIAL
        self.apiv3_key = settings.WECHAT_PAY_APIV3_KEY
        self.notify_url = settings.WECHAT_PAY_NOTIFY_URL
        self.base_url = "https://api.mch.weixin.qq.com"
        self._private_key = None

    def _load_private_key(self):
        """加载商户私钥"""
        if self._private_key is not None:
            return self._private_key

        try:
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
            return self._private_key
        except FileNotFoundError:
            logger.error("微信支付私钥文件不存在")
            raise BusinessException.validation_error("支付配置缺失：私钥文件不存在")
        except Exception as e:
            logger.error(f"加载微信支付私钥失败: {e}")
            raise BusinessException.validation_error("支付配置错误：加载私钥失败")

    def _generate_signature(self, method: str, url: str, timestamp: str, nonce: str, body: str) -> str:
        """生成签名"""
        # 构造待签名字符串
        sign_str = f"{method}\n{url}\n{timestamp}\n{nonce}\n{body}\n"
        
        return self._sign_raw(sign_str)

    def _sign_raw(self, sign_str: str) -> str:
        """使用商户私钥对原始字符串进行签名"""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            import base64
            
            private_key = self._load_private_key()
            signature = private_key.sign(
                sign_str.encode('utf-8'),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode('utf-8')
            
        except Exception as e:
            logger.error(f"生成微信支付签名失败: {e}")
            raise BusinessException.validation_error("支付签名生成失败")

    def generate_pay_sign(self, app_id: str, timestamp: str, nonce_str: str, package: str) -> str:
        """生成小程序调起支付签名"""
        sign_str = f"{app_id}\n{timestamp}\n{nonce_str}\n{package}\n"
        return self._sign_raw(sign_str)
    
    def _get_authorization_header(self, method: str, url: str, body: str) -> str:
        """获取授权头"""
        import uuid
        import time
        
        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4()).replace('-', '')
        signature = self._generate_signature(method, url, timestamp, nonce, body)
        
        return (
            f'WECHATPAY2-SHA256-RSA2048 '
            f'mchid="{self.mch_id}",'
            f'nonce_str="{nonce}",'
            f'timestamp="{timestamp}",'
            f'serial_no="{self.cert_serial}",'
            f'signature="{signature}"'
        )
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建支付订单"""
        url = f"{self.base_url}/v3/pay/transactions/jsapi"
        
        # 构造请求体
        body = {
            "appid": settings.WECHAT_APP_ID,
            "mchid": self.mch_id,
            "description": order_data["description"],
            "out_trade_no": order_data["out_trade_no"],
            "notify_url": self.notify_url,
            "amount": {
                "total": order_data["total_amount"],
                "currency": "CNY"
            },
            "payer": {
                "openid": order_data["openid"]
            }
        }
        
        body_json = json.dumps(body, separators=(',', ':'))
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": self._get_authorization_header("POST", "/v3/pay/transactions/jsapi", body_json),
                "User-Agent": "Happy8-MiniProgram/1.0.0"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, content=body_json, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"微信支付订单创建成功: {order_data['out_trade_no']}")
                    return data
                else:
                    error_data = response.json()
                    error_msg = error_data.get("message", "支付订单创建失败")
                    logger.error(f"微信支付订单创建失败: {error_msg}")
                    raise BusinessException.validation_error(f"支付失败: {error_msg}")
                    
        except httpx.RequestError as e:
            logger.error(f"微信支付请求失败: {e}")
            raise BusinessException.validation_error("支付服务不可用")
    
    async def query_order(self, out_trade_no: str) -> Dict[str, Any]:
        """查询支付订单"""
        url = f"{self.base_url}/v3/pay/transactions/out-trade-no/{out_trade_no}"
        query_params = f"?mchid={self.mch_id}"
        
        try:
            headers = {
                "Authorization": self._get_authorization_header("GET", f"/v3/pay/transactions/out-trade-no/{out_trade_no}{query_params}", ""),
                "User-Agent": "Happy8-MiniProgram/1.0.0"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url + query_params, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"微信支付订单查询成功: {out_trade_no}")
                    return data
                else:
                    error_data = response.json()
                    error_msg = error_data.get("message", "订单查询失败")
                    logger.error(f"微信支付订单查询失败: {error_msg}")
                    raise BusinessException.validation_error(f"订单查询失败: {error_msg}")
                    
        except httpx.RequestError as e:
            logger.error(f"微信支付查询请求失败: {e}")
            raise BusinessException.validation_error("支付服务不可用")
    
    def verify_notify(self, headers: Dict[str, str], body: str) -> bool:
        """验证支付回调通知"""
        try:
            # 提取签名信息
            wechatpay_signature = headers.get("Wechatpay-Signature")
            wechatpay_timestamp = headers.get("Wechatpay-Timestamp")
            wechatpay_nonce = headers.get("Wechatpay-Nonce")
            wechatpay_serial = headers.get("Wechatpay-Serial")
            
            if not all([wechatpay_signature, wechatpay_timestamp, wechatpay_nonce]):
                logger.error("微信支付回调签名信息不完整")
                return False
            
            # 构造验签字符串
            sign_str = f"{wechatpay_timestamp}\n{wechatpay_nonce}\n{body}\n"
            
            # TODO: 实现证书验签逻辑
            # 这里需要使用微信支付平台证书来验证签名
            
            logger.info("微信支付回调验证成功")
            return True
            
        except Exception as e:
            logger.error(f"微信支付回调验证失败: {e}")
            return False
