"""支付系统API路由"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.cache import get_cache, CacheService
from app.core.dependencies import get_current_active_user
from app.core.exceptions import create_success_response
from app.models.user import User
from app.services.payment_service import PaymentService
from app.api.schemas.payment_schemas import (
    CreatePaymentRequest,
    PaymentResponse,
    PaymentStatusResponse,
    CancelPaymentRequest,
    PaymentMethodResponse
)

router = APIRouter(prefix="/payments", tags=["支付系统"])


@router.get("/methods", response_model=Dict[str, Any])
async def get_payment_methods(
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """获取支持的支付方式"""
    payment_service = PaymentService(db, cache)
    
    methods = await payment_service.get_payment_methods()
    
    return create_success_response(data=methods, message="获取支付方式成功")


@router.post("/create", response_model=Dict[str, Any])
async def create_payment(
    request: CreatePaymentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """创建支付"""
    payment_service = PaymentService(db, cache)
    
    # 目前只支持微信支付
    if request.payment_method == "wechat_pay":
        result = await payment_service.create_wechat_payment(
            order_no=request.order_no,
            openid=current_user.openid
        )
        
        # 为小程序构造支付参数
        import time
        import secrets
        from app.core.config import settings
        
        # 生成小程序支付参数
        timeStamp = str(int(time.time()))
        nonceStr = secrets.token_hex(16)
        package = f"prepay_id={result['prepay_id']}"
        signType = "RSA"
        
        paySign = payment_service.wechat_pay.generate_pay_sign(
            app_id=settings.WECHAT_APP_ID,
            timestamp=timeStamp,
            nonce_str=nonceStr,
            package=package
        )
        
        payment_params = {
            "timeStamp": timeStamp,
            "nonceStr": nonceStr,
            "package": package,
            "signType": signType,
            "paySign": paySign
        }
        
        response_data = {
            **result,
            "payment_params": payment_params
        }
        
        return create_success_response(data=response_data, message="支付创建成功")
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的支付方式"
        )


@router.get("/status/{order_no}", response_model=Dict[str, Any])
async def get_payment_status(
    order_no: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """查询支付状态"""
    payment_service = PaymentService(db, cache)
    
    # 验证订单所有权
    from app.services.membership_service import MembershipService
    membership_service = MembershipService(db, cache)
    order = await membership_service.get_order_by_no(order_no)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="订单不存在"
        )
    
    if order.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权查看此订单"
        )
    
    # 查询支付状态
    status_info = await payment_service.query_payment_status(order_no)
    
    return create_success_response(data=status_info, message="查询支付状态成功")


@router.post("/cancel", response_model=Dict[str, Any])
async def cancel_payment(
    request: CancelPaymentRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """取消支付"""
    payment_service = PaymentService(db, cache)
    
    success = await payment_service.cancel_payment(
        order_no=request.order_no,
        user_id=current_user.id
    )
    
    if success:
        return create_success_response(message="支付取消成功")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="取消支付失败"
        )


@router.post("/notify/wechat")
async def wechat_pay_notify(
    request: Request,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """微信支付回调通知"""
    payment_service = PaymentService(db, cache)
    
    # 获取请求头和请求体
    headers = dict(request.headers)
    body = await request.body()
    body_str = body.decode('utf-8')
    
    # 处理回调通知
    success = await payment_service.handle_wechat_notify(headers, body_str)
    
    if success:
        # 微信要求返回特定格式的成功响应
        return {"code": "SUCCESS", "message": "成功"}
    else:
        # 返回失败响应，微信会重新发送通知
        return {"code": "FAIL", "message": "失败"}


@router.get("/test/notify")
async def test_payment_notify(
    order_no: str,
    db: Session = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """测试支付回调（仅开发环境）"""
    from app.core.config import settings
    
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="接口不存在"
        )
    
    # 模拟微信支付成功回调
    mock_headers = {
        "wechatpay-signature": "mock_signature",
        "wechatpay-timestamp": "1234567890",
        "wechatpay-nonce": "mock_nonce",
        "wechatpay-serial": "mock_serial"
    }
    
    mock_body = f'''{{
        "id": "mock_notification_id",
        "create_time": "2025-01-01T00:00:00+08:00",
        "resource_type": "encrypt-resource",
        "event_type": "TRANSACTION.SUCCESS",
        "resource": {{
            "original_type": "transaction",
            "algorithm": "AEAD_AES_256_GCM",
            "ciphertext": "mock_encrypted_data",
            "associated_data": "transaction",
            "nonce": "mock_nonce"
        }},
        "summary": "支付成功"
    }}'''
    
    # 由于这是测试接口，直接模拟成功的处理
    payment_service = PaymentService(db, cache)
    
    # 获取订单并直接标记为已支付（仅测试用）
    from app.services.membership_service import MembershipService
    membership_service = MembershipService(db, cache)
    
    order = await membership_service.get_order_by_no(order_no)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="订单不存在"
        )
    
    # 模拟支付成功
    await payment_service._handle_payment_success(
        order=order,
        transaction_id=f"mock_transaction_{order_no}",
        pay_data={"trade_state": "SUCCESS"}
    )
    
    return create_success_response(message="测试支付回调成功")
