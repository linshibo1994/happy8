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
            "description": "基于历史相对频率生成号码排序分，结果不代表下一期真实概率提升",
            "required_level": "free",
            "sort_order": 1,
            "default_params": '{"weight_recent": 0.7, "min_frequency": 0.01}'
        },
        {
            "algorithm_name": "hot_cold",
            "display_name": "冷热分析",
            "description": "基于近期热号与当前遗漏冷号生成启发式排序分",
            "required_level": "free",
            "sort_order": 2,
            "default_params": '{"hot_threshold": 0.6, "cold_threshold": 0.3}'
        },
        {
            "algorithm_name": "missing",
            "display_name": "遗漏分析",
            "description": "基于当前遗漏期数生成启发式排序分，不将遗漏解释为理论回补概率",
            "required_level": "free",
            "sort_order": 3,
            "default_params": '{"max_missing_weight": 0.8}'
        },

        # 马尔可夫链系列
        {
            "algorithm_name": "markov",
            "display_name": "基础马尔可夫链",
            "description": "基于单号上一期出现状态的一阶跨期转移排序模型",
            "required_level": "vip",
            "sort_order": 4,
            "default_params": '{"transition_window": 20}'
        },
        {
            "algorithm_name": "markov_2nd",
            "display_name": "二阶马尔可夫链",
            "description": "基于单号前两期出现状态的二阶跨期转移排序模型",
            "required_level": "vip",
            "sort_order": 5,
            "default_params": '{"order": 2, "transition_window": 30}'
        },
        {
            "algorithm_name": "markov_3rd",
            "display_name": "三阶马尔可夫链",
            "description": "基于单号前三期出现状态的三阶跨期转移排序模型",
            "required_level": "premium",
            "sort_order": 6,
            "default_params": '{"order": 3, "transition_window": 50}'
        },
        {
            "algorithm_name": "adaptive_markov",
            "display_name": "自适应马尔可夫链",
            "description": "按数据量融合一至三阶跨期状态转移评分的排序模型",
            "required_level": "premium",
            "sort_order": 7,
            "default_params": '{"max_order": 3, "adaptation_threshold": 0.1}'
        },

        # 机器学习系列
        {
            "algorithm_name": "ensemble",
            "display_name": "基础集成学习",
            "description": "固定权重融合统计、遗漏和跨期状态转移评分",
            "required_level": "vip",
            "sort_order": 8,
            "default_params": '{"algorithms": ["frequency", "hot_cold", "missing"]}'
        },
        {
            "algorithm_name": "advanced_ensemble",
            "display_name": "高级集成学习",
            "description": "使用时间后段验证集加权的多输出机器学习排序模型",
            "required_level": "premium",
            "sort_order": 9,
            "default_params": '{"meta_learner": "gradient_boosting", "cv_folds": 5}'
        },
        {
            "algorithm_name": "clustering",
            "display_name": "聚类分析",
            "description": "基于KMeans聚类的历史开奖结构相似性排序模型",
            "required_level": "vip",
            "sort_order": 10,
            "default_params": '{"n_clusters": 8, "cluster_method": "kmeans"}'
        },
        {
            "algorithm_name": "monte_carlo",
            "display_name": "蒙特卡洛模拟",
            "description": "按显式抽样假设进行1-80无放回蒙特卡洛模拟，输出入选排序分",
            "required_level": "vip",
            "sort_order": 11,
            "default_params": '{"simulations": 10000, "confidence_level": 0.95}'
        },

        # 深度学习系列
        {
            "algorithm_name": "lstm",
            "display_name": "LSTM深度学习",
            "description": "基于时间升序窗口的LSTM多标签排序模型，依赖TensorFlow可用性",
            "required_level": "premium",
            "sort_order": 12,
            "default_params": '{"lstm_units": 64, "epochs": 100, "sequence_length": 20}'
        },
        {
            "algorithm_name": "transformer",
            "display_name": "Transformer深度学习",
            "description": "基于时间升序窗口的Transformer多标签排序模型，依赖PyTorch可用性",
            "required_level": "premium",
            "sort_order": 13,
            "default_params": '{"attention_heads": 8, "layers": 6, "d_model": 128}'
        },
        {
            "algorithm_name": "gnn",
            "display_name": "图神经网络",
            "description": "基于号码共现图传播与时间样本权重的确定性排序模型",
            "required_level": "premium",
            "sort_order": 14,
            "default_params": '{"graph_layers": 3, "node_features": 32}'
        },

        # 高级算法
        {
            "algorithm_name": "bayesian",
            "display_name": "贝叶斯推理",
            "description": "基于Dirichlet后验均值的贝叶斯排序评分模型",
            "required_level": "premium",
            "sort_order": 15,
            "default_params": '{"prior_strength": 1.0, "posterior_samples": 1000}'
        },
        {
            "algorithm_name": "high_confidence",
            "display_name": "质量门控预测器",
            "description": "基于输出结构和数据质量评分的门控预测器，不代表真实命中概率",
            "required_level": "premium",
            "sort_order": 16,
            "default_params": '{"confidence_threshold": 0.8, "conservative_factor": 1.2}'
        },

        # 超级算法
        {
            "algorithm_name": "super_predictor",
            "display_name": "综合排序融合器",
            "description": "按算法信号组融合多种排序分，避免同类模型重复放大",
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
