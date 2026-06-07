"""创建初始数据库结构

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级数据库结构"""
    # 创建用户表
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False, comment='用户ID'),
        sa.Column('openid', sa.String(length=64), nullable=False, comment='微信openid'),
        sa.Column('unionid', sa.String(length=64), nullable=True, comment='微信unionid'),
        sa.Column('nickname', sa.String(length=100), nullable=False, comment='用户昵称'),
        sa.Column('avatar_url', sa.String(length=500), nullable=True, comment='头像URL'),
        sa.Column('phone', sa.String(length=20), nullable=True, comment='手机号'),
        sa.Column('email', sa.String(length=100), nullable=True, comment='邮箱'),
        sa.Column('is_active', sa.Boolean(), nullable=False, comment='是否激活'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('openid'),
        sa.UniqueConstraint('unionid')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_openid'), 'users', ['openid'], unique=False)
    op.create_index(op.f('ix_users_unionid'), 'users', ['unionid'], unique=False)
    
    # 创建用户资料表
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.Integer(), nullable=False, comment='资料ID'),
        sa.Column('user_id', sa.Integer(), nullable=False, comment='用户ID'),
        sa.Column('real_name', sa.String(length=50), nullable=True, comment='真实姓名'),
        sa.Column('id_card', sa.String(length=200), nullable=True, comment='身份证号(加密)'),
        sa.Column('gender', sa.String(length=10), nullable=True, comment='性别'),
        sa.Column('birthday', sa.DateTime(), nullable=True, comment='生日'),
        sa.Column('address', sa.Text(), nullable=True, comment='地址'),
        sa.Column('preferences', sa.Text(), nullable=True, comment='用户偏好设置(JSON)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_user_profiles_id'), 'user_profiles', ['id'], unique=False)
    
    # 创建会员套餐表
    op.create_table(
        'membership_plans',
        sa.Column('id', sa.Integer(), nullable=False, comment='套餐ID'),
        sa.Column('name', sa.String(length=50), nullable=False, comment='套餐名称'),
        sa.Column('level', sa.Enum('free', 'vip', 'premium', name='membershiplevel'), nullable=False, comment='会员等级'),
        sa.Column('duration_days', sa.Integer(), nullable=False, comment='有效期天数'),
        sa.Column('price', sa.Integer(), nullable=False, comment='价格(分)'),
        sa.Column('original_price', sa.Integer(), nullable=True, comment='原价(分)'),
        sa.Column('features', sa.Text(), nullable=False, comment='特权列表(JSON)'),
        sa.Column('max_predictions_per_day', sa.Integer(), nullable=True, comment='每日预测次数限制'),
        sa.Column('available_algorithms', sa.Text(), nullable=True, comment='可用算法列表(JSON)'),
        sa.Column('is_active', sa.Boolean(), nullable=False, comment='是否启用'),
        sa.Column('sort_order', sa.Integer(), nullable=True, comment='排序权重'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_membership_plans_id'), 'membership_plans', ['id'], unique=False)
    
    # 创建用户会员信息表
    op.create_table(
        'memberships',
        sa.Column('id', sa.Integer(), nullable=False, comment='会员ID'),
        sa.Column('user_id', sa.Integer(), nullable=False, comment='用户ID'),
        sa.Column('level', sa.Enum('free', 'vip', 'premium', name='membershiplevel'), nullable=False, comment='当前会员等级'),
        sa.Column('expire_date', sa.DateTime(), nullable=True, comment='到期时间'),
        sa.Column('auto_renew', sa.Boolean(), nullable=True, comment='是否自动续费'),
        sa.Column('predictions_today', sa.Integer(), nullable=True, comment='今日已用预测次数'),
        sa.Column('predictions_total', sa.Integer(), nullable=True, comment='总预测次数'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_memberships_id'), 'memberships', ['id'], unique=False)
    
    # 创建会员订单表
    op.create_table(
        'membership_orders',
        sa.Column('id', sa.Integer(), nullable=False, comment='订单ID'),
        sa.Column('order_no', sa.String(length=32), nullable=False, comment='订单号'),
        sa.Column('user_id', sa.Integer(), nullable=False, comment='用户ID'),
        sa.Column('plan_id', sa.Integer(), nullable=False, comment='套餐ID'),
        sa.Column('amount', sa.Integer(), nullable=False, comment='实际支付金额(分)'),
        sa.Column('original_amount', sa.Integer(), nullable=True, comment='原价(分)'),
        sa.Column('discount_amount', sa.Integer(), nullable=True, comment='优惠金额(分)'),
        sa.Column('status', sa.Enum('pending', 'paid', 'cancelled', 'refunded', 'expired', name='orderstatus'), nullable=False, comment='订单状态'),
        sa.Column('pay_method', sa.Enum('wechat_pay', 'alipay', name='paymentmethod'), nullable=True, comment='支付方式'),
        sa.Column('transaction_id', sa.String(length=64), nullable=True, comment='第三方交易号'),
        sa.Column('expire_at', sa.DateTime(), nullable=True, comment='订单过期时间'),
        sa.Column('paid_at', sa.DateTime(), nullable=True, comment='支付时间'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.ForeignKeyConstraint(['plan_id'], ['membership_plans.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_no')
    )
    op.create_index(op.f('ix_membership_orders_id'), 'membership_orders', ['id'], unique=False)
    op.create_index(op.f('ix_membership_orders_order_no'), 'membership_orders', ['order_no'], unique=False)
    
    # 创建彩票开奖结果表
    op.create_table(
        'lottery_results',
        sa.Column('id', sa.Integer(), nullable=False, comment='结果ID'),
        sa.Column('issue', sa.String(length=20), nullable=False, comment='期号'),
        sa.Column('draw_date', sa.DateTime(), nullable=False, comment='开奖日期'),
        sa.Column('numbers', sa.JSON(), nullable=False, comment='开奖号码(JSON)'),
        sa.Column('sum_value', sa.Integer(), nullable=False, comment='号码总和'),
        sa.Column('odd_count', sa.Integer(), nullable=False, comment='奇数个数'),
        sa.Column('even_count', sa.Integer(), nullable=False, comment='偶数个数'),
        sa.Column('big_count', sa.Integer(), nullable=False, comment='大号个数(41-80)'),
        sa.Column('small_count', sa.Integer(), nullable=False, comment='小号个数(1-40)'),
        sa.Column('zone_distribution', sa.JSON(), nullable=True, comment='区间分布统计(JSON)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('issue')
    )
    op.create_index(op.f('ix_lottery_results_id'), 'lottery_results', ['id'], unique=False)
    op.create_index(op.f('ix_lottery_results_issue'), 'lottery_results', ['issue'], unique=False)
    op.create_index(op.f('ix_lottery_results_draw_date'), 'lottery_results', ['draw_date'], unique=False)
    
    # 创建预测历史记录表
    op.create_table(
        'prediction_history',
        sa.Column('id', sa.Integer(), nullable=False, comment='预测ID'),
        sa.Column('user_id', sa.Integer(), nullable=False, comment='用户ID'),
        sa.Column('algorithm', sa.String(length=50), nullable=False, comment='预测算法'),
        sa.Column('target_issue', sa.String(length=20), nullable=False, comment='目标期号'),
        sa.Column('periods', sa.Integer(), nullable=True, comment='分析期数'),
        sa.Column('count', sa.Integer(), nullable=True, comment='预测号码个数'),
        sa.Column('predicted_numbers', sa.JSON(), nullable=False, comment='预测号码(JSON)'),
        sa.Column('confidence_score', sa.Float(), nullable=True, comment='置信度'),
        sa.Column('algorithm_params', sa.JSON(), nullable=True, comment='算法参数(JSON)'),
        sa.Column('actual_numbers', sa.JSON(), nullable=True, comment='实际开奖号码(JSON)'),
        sa.Column('hit_count', sa.Integer(), nullable=True, comment='命中个数'),
        sa.Column('hit_rate', sa.Float(), nullable=True, comment='命中率'),
        sa.Column('is_hit', sa.Boolean(), nullable=True, comment='是否命中'),
        sa.Column('analysis_data', sa.JSON(), nullable=True, comment='分析过程数据(JSON)'),
        sa.Column('execution_time', sa.Float(), nullable=True, comment='执行时间(秒)'),
        sa.Column('is_cached', sa.Boolean(), nullable=True, comment='是否来自缓存'),
        sa.Column('cache_key', sa.String(length=100), nullable=True, comment='缓存键'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_prediction_history_id'), 'prediction_history', ['id'], unique=False)
    op.create_index(op.f('ix_prediction_history_user_id'), 'prediction_history', ['user_id'], unique=False)
    op.create_index(op.f('ix_prediction_history_algorithm'), 'prediction_history', ['algorithm'], unique=False)
    op.create_index(op.f('ix_prediction_history_target_issue'), 'prediction_history', ['target_issue'], unique=False)
    op.create_index(op.f('ix_prediction_history_cache_key'), 'prediction_history', ['cache_key'], unique=False)
    
    # 创建算法配置表
    op.create_table(
        'algorithm_configs',
        sa.Column('id', sa.Integer(), nullable=False, comment='配置ID'),
        sa.Column('algorithm_name', sa.String(length=50), nullable=False, comment='算法名称'),
        sa.Column('display_name', sa.String(length=100), nullable=False, comment='显示名称'),
        sa.Column('description', sa.Text(), nullable=True, comment='算法描述'),
        sa.Column('default_params', sa.JSON(), nullable=True, comment='默认参数(JSON)'),
        sa.Column('required_level', sa.String(length=20), nullable=True, comment='所需会员等级'),
        sa.Column('is_active', sa.Boolean(), nullable=True, comment='是否启用'),
        sa.Column('sort_order', sa.Integer(), nullable=True, comment='排序权重'),
        sa.Column('avg_execution_time', sa.Float(), nullable=True, comment='平均执行时间'),
        sa.Column('success_rate', sa.Float(), nullable=True, comment='成功率'),
        sa.Column('usage_count', sa.Integer(), nullable=True, comment='使用次数'),
        sa.Column('created_at', sa.DateTime(), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('algorithm_name')
    )
    op.create_index(op.f('ix_algorithm_configs_id'), 'algorithm_configs', ['id'], unique=False)


def downgrade() -> None:
    """降级数据库结构"""
    # 删除所有表
    op.drop_index(op.f('ix_algorithm_configs_id'), table_name='algorithm_configs')
    op.drop_table('algorithm_configs')
    
    op.drop_index(op.f('ix_prediction_history_cache_key'), table_name='prediction_history')
    op.drop_index(op.f('ix_prediction_history_target_issue'), table_name='prediction_history')
    op.drop_index(op.f('ix_prediction_history_algorithm'), table_name='prediction_history')
    op.drop_index(op.f('ix_prediction_history_user_id'), table_name='prediction_history')
    op.drop_index(op.f('ix_prediction_history_id'), table_name='prediction_history')
    op.drop_table('prediction_history')
    
    op.drop_index(op.f('ix_lottery_results_draw_date'), table_name='lottery_results')
    op.drop_index(op.f('ix_lottery_results_issue'), table_name='lottery_results')
    op.drop_index(op.f('ix_lottery_results_id'), table_name='lottery_results')
    op.drop_table('lottery_results')
    
    op.drop_index(op.f('ix_membership_orders_order_no'), table_name='membership_orders')
    op.drop_index(op.f('ix_membership_orders_id'), table_name='membership_orders')
    op.drop_table('membership_orders')
    
    op.drop_index(op.f('ix_memberships_id'), table_name='memberships')
    op.drop_table('memberships')
    
    op.drop_index(op.f('ix_membership_plans_id'), table_name='membership_plans')
    op.drop_table('membership_plans')
    
    op.drop_index(op.f('ix_user_profiles_id'), table_name='user_profiles')
    op.drop_table('user_profiles')
    
    op.drop_index(op.f('ix_users_unionid'), table_name='users')
    op.drop_index(op.f('ix_users_openid'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')