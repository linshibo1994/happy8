"""支付服务"""

import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.membership import MembershipOrder, OrderStatus, PaymentMethod
from app.core.exceptions import BusinessException
from app.core.cache import CacheService
from app.core.logging import payment_logger as logger
from app.utils.wechat import WeChatPayAPI
from app.services.membership_service import MembershipService


class PaymentService:
    """支付服务"""
    
    def __init__(self, db: Session, cache: CacheService):
        self.db = db
        self.cache = cache
        self.wechat_pay = WeChatPayAPI()
        self.membership_service = MembershipService(db, cache)
    
    async def create_wechat_payment(
        self,
        order_no: str,
        openid: str,
        description: str = None
    ) -> Dict[str, Any]:
        """创建微信支付"""
        try:
            # 获取订单信息
            order = await self.membership_service.get_order_by_no(order_no)
            if not order:
                raise BusinessException.order_not_found("订单不存在")
            
            # 检查订单状态
            if order.status != OrderStatus.PENDING:
                raise BusinessException.validation_error("订单状态不允许支付")
            
            # 检查订单是否过期
            if order.expire_at and order.expire_at < datetime.now():
                # 标记为过期
                order.status = OrderStatus.EXPIRED
                order.updated_at = datetime.now()
                self.db.commit()
                raise BusinessException.validation_error("订单已过期")
            
            # 构造支付请求数据
            pay_data = {
                "out_trade_no": order_no,
                "total_amount": order.amount,
                "description": description or f"Happy8会员-{order.plan.name if order.plan else '套餐'}",
                "openid": openid
            }
            
            # 调用微信支付API
            wechat_result = await self.wechat_pay.create_order(pay_data)
            
            # 更新订单支付方式
            order.pay_method = PaymentMethod.WECHAT_PAY
            order.updated_at = datetime.now()
            self.db.commit()
            
            # 记录支付事件
            from app.core.logging import log_payment_event
            log_payment_event(
                user_id=order.user_id,
                order_no=order_no,
                amount=order.amount,
                event_type="payment_created",
                details={"openid": openid}
            )
            
            return {
                "prepay_id": wechat_result.get("prepay_id"),
                "order_no": order_no,
                "amount": order.amount,
                "currency": "CNY",
                "expire_time": order.expire_at.isoformat() if order.expire_at else None
            }
            
        except Exception as e:
            logger.error(f"创建微信支付失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("创建支付失败")
    
    async def handle_wechat_notify(self, headers: Dict[str, str], body: str) -> bool:
        """处理微信支付回调通知"""
        try:
            # 验证签名
            if not self.wechat_pay.verify_notify(headers, body):
                logger.error("微信支付回调签名验证失败")
                return False
            
            # 解析回调数据
            notify_data = json.loads(body)
            resource = notify_data.get("resource", {})
            
            # 解密回调数据（实际项目中需要实现解密逻辑）
            # 这里简化处理，假设已解密
            decrypt_data = resource  # 实际应该是解密后的数据
            
            transaction_id = decrypt_data.get("transaction_id")
            out_trade_no = decrypt_data.get("out_trade_no")
            trade_state = decrypt_data.get("trade_state")
            
            if not out_trade_no:
                logger.error("微信支付回调缺少订单号")
                return False
            
            # 获取订单
            order = await self.membership_service.get_order_by_no(out_trade_no)
            if not order:
                logger.error(f"微信支付回调订单不存在: {out_trade_no}")
                return False
            
            # 处理支付结果
            if trade_state == "SUCCESS":
                await self._handle_payment_success(order, transaction_id, decrypt_data)
            elif trade_state in ["CLOSED", "REVOKED", "PAYERROR"]:
                await self._handle_payment_failure(order, trade_state)
            
            return True
            
        except Exception as e:
            logger.error(f"处理微信支付回调失败: {e}")
            return False
    
    async def _handle_payment_success(
        self,
        order: MembershipOrder,
        transaction_id: str,
        pay_data: Dict[str, Any]
    ):
        """处理支付成功"""
        try:
            # 检查订单状态，避免重复处理
            if order.status == OrderStatus.PAID:
                logger.warning(f"订单已支付，跳过处理: {order.order_no}")
                return
            
            # 更新订单状态
            order.status = OrderStatus.PAID
            order.transaction_id = transaction_id
            order.paid_at = datetime.now()
            order.updated_at = datetime.now()
            
            # 升级用户会员
            plan = await self.membership_service.get_plan_by_id(order.plan_id)
            await self.membership_service.upgrade_membership(
                user_id=order.user_id,
                plan=plan,
                order=order
            )
            
            # 记录支付成功事件
            from app.core.logging import log_payment_event
            log_payment_event(
                user_id=order.user_id,
                order_no=order.order_no,
                amount=order.amount,
                event_type="payment_success",
                details={
                    "transaction_id": transaction_id,
                    "plan_name": plan.name,
                    "plan_level": plan.level.value
                }
            )
            
            logger.info(f"支付成功处理完成: {order.order_no}")
            
        except Exception as e:
            logger.error(f"处理支付成功失败: {e}")
            raise
    
    async def _handle_payment_failure(
        self,
        order: MembershipOrder,
        trade_state: str
    ):
        """处理支付失败"""
        try:
            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self.db.commit()
            
            # 记录支付失败事件
            from app.core.logging import log_payment_event
            log_payment_event(
                user_id=order.user_id,
                order_no=order.order_no,
                amount=order.amount,
                event_type="payment_failed",
                details={"trade_state": trade_state}
            )
            
            logger.info(f"支付失败处理完成: {order.order_no}, 状态: {trade_state}")
            
        except Exception as e:
            logger.error(f"处理支付失败失败: {e}")
            raise
    
    async def query_payment_status(self, order_no: str) -> Dict[str, Any]:
        """查询支付状态"""
        try:
            # 从数据库获取订单状态
            order = await self.membership_service.get_order_by_no(order_no)
            if not order:
                raise BusinessException.order_not_found("订单不存在")
            
            # 如果订单已支付，直接返回状态
            if order.status == OrderStatus.PAID:
                return {
                    "order_no": order_no,
                    "status": "paid",
                    "transaction_id": order.transaction_id,
                    "paid_at": order.paid_at.isoformat() if order.paid_at else None
                }
            
            # 如果订单状态为待支付，查询微信支付状态
            if order.status == OrderStatus.PENDING:
                try:
                    wechat_result = await self.wechat_pay.query_order(order_no)
                    trade_state = wechat_result.get("trade_state")
                    
                    if trade_state == "SUCCESS":
                        # 微信显示支付成功，但本地状态未更新，触发状态同步
                        transaction_id = wechat_result.get("transaction_id")
                        await self._handle_payment_success(order, transaction_id, wechat_result)
                        
                        return {
                            "order_no": order_no,
                            "status": "paid",
                            "transaction_id": transaction_id
                        }
                    elif trade_state in ["CLOSED", "REVOKED", "PAYERROR"]:
                        # 支付失败，更新状态
                        await self._handle_payment_failure(order, trade_state)
                        
                        return {
                            "order_no": order_no,
                            "status": "failed",
                            "reason": trade_state
                        }
                    else:
                        # 支付进行中
                        return {
                            "order_no": order_no,
                            "status": "pending",
                            "trade_state": trade_state
                        }
                        
                except Exception as e:
                    logger.warning(f"查询微信支付状态失败: {e}")
                    # 微信查询失败，返回本地状态
                    pass
            
            # 返回本地订单状态
            status_map = {
                OrderStatus.PENDING: "pending",
                OrderStatus.PAID: "paid",
                OrderStatus.CANCELLED: "cancelled",
                OrderStatus.REFUNDED: "refunded",
                OrderStatus.EXPIRED: "expired"
            }
            
            return {
                "order_no": order_no,
                "status": status_map.get(order.status, "unknown"),
                "created_at": order.created_at.isoformat(),
                "expire_at": order.expire_at.isoformat() if order.expire_at else None
            }
            
        except Exception as e:
            logger.error(f"查询支付状态失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("查询支付状态失败")
    
    async def cancel_payment(self, order_no: str, user_id: int) -> bool:
        """取消支付"""
        try:
            # 获取订单
            order = await self.membership_service.get_order_by_no(order_no)
            if not order:
                raise BusinessException.order_not_found("订单不存在")
            
            # 检查订单所有权
            if order.user_id != user_id:
                raise BusinessException.permission_denied("无权操作此订单")
            
            # 检查订单状态
            if order.status != OrderStatus.PENDING:
                raise BusinessException.validation_error("订单状态不允许取消")
            
            # 更新订单状态
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            self.db.commit()
            
            # 记录取消事件
            from app.core.logging import log_payment_event
            log_payment_event(
                user_id=user_id,
                order_no=order_no,
                amount=order.amount,
                event_type="payment_cancelled"
            )
            
            logger.info(f"订单取消成功: {order_no}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"取消支付失败: {e}")
            if isinstance(e, BusinessException):
                raise
            raise BusinessException.validation_error("取消支付失败")
    
    async def get_payment_methods(self) -> Dict[str, Any]:
        """获取支持的支付方式"""
        return {
            "methods": [
                {
                    "id": "wechat_pay",
                    "name": "微信支付",
                    "icon": "wechat",
                    "enabled": True,
                    "description": "使用微信支付，安全便捷"
                }
            ],
            "default": "wechat_pay"
        }