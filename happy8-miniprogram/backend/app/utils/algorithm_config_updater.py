"""更新算法配置数据 - 添加所有17种原始Happy8算法"""

from datetime import datetime
from sqlalchemy.orm import Session
from app.models.prediction import AlgorithmConfig
from app.core.database import get_db

def update_algorithm_configs():
    """更新算法配置，确保包含所有17种原始算法"""
    
    # 完整的算法配置列表
    algorithm_configs = [
        # 基础统计算法
        {
            "algorithm_name": "frequency",
            "display_name": "频率分析",
            "description": "基于历史开奖频率的统计分析算法，分析各号码的出现频率趋势",
            "required_level": "free",
            "sort_order": 1,
            "default_params": '{"weight_recent": 0.7, "min_frequency": 0.01}'
        },
        {
            "algorithm_name": "hot_cold",
            "display_name": "冷热分析", 
            "description": "分析号码的冷热趋势，识别当前热门和冷门号码",
            "required_level": "free",
            "sort_order": 2,
            "default_params": '{"hot_threshold": 0.6, "cold_threshold": 0.3}'
        },
        {
            "algorithm_name": "missing",
            "display_name": "遗漏分析",
            "description": "基于号码遗漏期数的分析算法，预测遗漏较久的号码回补",
            "required_level": "free", 
            "sort_order": 3,
            "default_params": '{"max_missing_weight": 0.8}'
        },
        
        # 马尔可夫链系列
        {
            "algorithm_name": "markov",
            "display_name": "基础马尔可夫链",
            "description": "基于一阶马尔可夫链的状态转移预测算法",
            "required_level": "vip",
            "sort_order": 4,
            "default_params": '{"transition_window": 20}'
        },
        {
            "algorithm_name": "markov_2nd",
            "display_name": "二阶马尔可夫链",
            "description": "基于二阶马尔可夫链的高级状态转移预测算法",
            "required_level": "vip",
            "sort_order": 5,
            "default_params": '{"order": 2, "transition_window": 30}'
        },
        {
            "algorithm_name": "markov_3rd", 
            "display_name": "三阶马尔可夫链",
            "description": "基于三阶马尔可夫链的超高级状态转移预测算法",
            "required_level": "premium",
            "sort_order": 6,
            "default_params": '{"order": 3, "transition_window": 50}'
        },
        {
            "algorithm_name": "adaptive_markov",
            "display_name": "自适应马尔可夫链",
            "description": "自适应调整阶数的智能马尔可夫链预测算法",
            "required_level": "premium",
            "sort_order": 7,
            "default_params": '{"max_order": 3, "adaptation_threshold": 0.1}'
        },
        
        # 机器学习系列
        {
            "algorithm_name": "ensemble",
            "display_name": "基础集成学习",
            "description": "集成多种机器学习算法的预测结果",
            "required_level": "vip",
            "sort_order": 8,
            "default_params": '{"algorithms": ["frequency", "hot_cold", "missing"]}'
        },
        {
            "algorithm_name": "advanced_ensemble",
            "display_name": "高级集成学习",
            "description": "使用高级机器学习技术的智能集成预测算法",
            "required_level": "premium",
            "sort_order": 9,
            "default_params": '{"meta_learner": "gradient_boosting", "cv_folds": 5}'
        },
        {
            "algorithm_name": "clustering",
            "display_name": "聚类分析",
            "description": "基于聚类分析的模式识别预测算法",
            "required_level": "vip",
            "sort_order": 10,
            "default_params": '{"n_clusters": 8, "cluster_method": "kmeans"}'
        },
        {
            "algorithm_name": "monte_carlo",
            "display_name": "蒙特卡洛模拟",
            "description": "使用蒙特卡洛方法进行随机模拟预测",
            "required_level": "vip",
            "sort_order": 11,
            "default_params": '{"simulations": 10000, "confidence_level": 0.95}'
        },
        
        # 深度学习系列
        {
            "algorithm_name": "lstm",
            "display_name": "LSTM深度学习",
            "description": "基于长短期记忆网络的深度学习预测算法",
            "required_level": "premium",
            "sort_order": 12,
            "default_params": '{"lstm_units": 64, "epochs": 100, "sequence_length": 20}'
        },
        {
            "algorithm_name": "transformer",
            "display_name": "Transformer深度学习",
            "description": "基于Transformer架构的先进深度学习预测算法", 
            "required_level": "premium",
            "sort_order": 13,
            "default_params": '{"attention_heads": 8, "layers": 6, "d_model": 128}'
        },
        {
            "algorithm_name": "gnn",
            "display_name": "图神经网络",
            "description": "基于图神经网络的复杂关系建模预测算法",
            "required_level": "premium",
            "sort_order": 14,
            "default_params": '{"graph_layers": 3, "node_features": 32}'
        },
        
        # 高级算法
        {
            "algorithm_name": "bayesian",
            "display_name": "贝叶斯推理",
            "description": "基于贝叶斯统计推理的概率预测算法",
            "required_level": "premium",
            "sort_order": 15,
            "default_params": '{"prior_strength": 1.0, "posterior_samples": 1000}'
        },
        {
            "algorithm_name": "high_confidence",
            "display_name": "高置信度预测器",
            "description": "专注于高置信度预测的保守型算法",
            "required_level": "premium",
            "sort_order": 16,
            "default_params": '{"confidence_threshold": 0.8, "conservative_factor": 1.2}'
        },
        
        # 超级算法
        {
            "algorithm_name": "super_predictor",
            "display_name": "超级预测器",
            "description": "融合所有算法的终极预测器，使用智能权重分配",
            "required_level": "premium",
            "sort_order": 17,
            "default_params": '{"fusion_strategy": "dynamic_weighting", "quality_threshold": 0.7}'
        }
    ]
    
    return algorithm_configs

def insert_algorithm_configs(db: Session):
    """插入算法配置到数据库"""
    
    algorithm_configs = update_algorithm_configs()
    
    for config_data in algorithm_configs:
        # 检查算法是否已存在
        existing = db.query(AlgorithmConfig).filter(
            AlgorithmConfig.algorithm_name == config_data["algorithm_name"]
        ).first()
        
        if existing:
            # 更新现有配置
            for key, value in config_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            existing.updated_at = datetime.now()
        else:
            # 创建新配置
            new_config = AlgorithmConfig(
                algorithm_name=config_data["algorithm_name"],
                display_name=config_data["display_name"],
                description=config_data["description"],
                required_level=config_data["required_level"],
                sort_order=config_data["sort_order"],
                default_params=config_data["default_params"],
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            db.add(new_config)
    
    try:
        db.commit()
        print(f"✅ 成功更新 {len(algorithm_configs)} 个算法配置")
    except Exception as e:
        db.rollback()
        print(f"❌ 更新算法配置失败: {e}")
        raise

if __name__ == "__main__":
    # 如果直接运行此脚本，则执行更新
    print("更新算法配置...")
    configs = update_algorithm_configs()
    for config in configs:
        print(f"- {config['algorithm_name']}: {config['display_name']} ({config['required_level']})")
    print(f"\n总共配置 {len(configs)} 个算法")
