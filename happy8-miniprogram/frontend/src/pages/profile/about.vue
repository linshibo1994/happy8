<template>
  <view class="about">
    <!-- App Logo -->
    <view class="logo-section">
      <view class="container">
        <view class="logo-card">
          <image class="app-logo" :src="appLogo" mode="aspectFit" />
          <text class="app-name">Happy8智能预测</text>
          <text class="app-version">v{{ appVersion }}</text>
          <text class="app-slogan">专业的快乐8号码预测平台</text>
        </view>
      </view>
    </view>

    <!-- 产品介绍 -->
    <view class="intro-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">产品介绍</text>
        </view>
        
        <view class="intro-card card">
          <text class="intro-text">
            Happy8智能预测是一款专业的快乐8号码预测应用，采用17种先进的算法模型，包括频率分析、热冷号分析、马尔可夫链、深度学习等技术，为用户提供科学、准确的号码预测服务。
          </text>
        </view>
      </view>
    </view>

    <!-- 核心功能 -->
    <view class="features-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">核心功能</text>
        </view>
        
        <view class="features-grid">
          <view 
            v-for="feature in features" 
            :key="feature.title"
            class="feature-item card"
          >
            <text class="feature-icon">{{ feature.icon }}</text>
            <text class="feature-title">{{ feature.title }}</text>
            <text class="feature-desc">{{ feature.description }}</text>
          </view>
        </view>
      </view>
    </view>

    <!-- 算法介绍 -->
    <view class="algorithms-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">算法技术</text>
        </view>
        
        <view class="algorithms-list">
          <view 
            v-for="algorithm in algorithms" 
            :key="algorithm.name"
            class="algorithm-item card"
          >
            <view class="algorithm-header">
              <text class="algorithm-name">{{ algorithm.name }}</text>
              <view class="algorithm-badge" :class="algorithm.level.toLowerCase()">
                <text class="badge-text">{{ getLevelName(algorithm.level) }}</text>
              </view>
            </view>
            <text class="algorithm-desc">{{ algorithm.description }}</text>
          </view>
        </view>
      </view>
    </view>

    <!-- 数据统计 -->
    <view class="stats-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">平台数据</text>
        </view>
        
        <view class="stats-grid">
          <view 
            v-for="stat in stats" 
            :key="stat.label"
            class="stat-item card"
          >
            <text class="stat-value">{{ stat.value }}</text>
            <text class="stat-label">{{ stat.label }}</text>
          </view>
        </view>
      </view>
    </view>

    <!-- 团队介绍 -->
    <view class="team-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">开发团队</text>
        </view>
        
        <view class="team-card card">
          <text class="team-intro">
            我们是一支专注于人工智能和数据分析的技术团队，致力于为用户提供最优质的预测服务。团队成员具有丰富的机器学习、数据挖掘和软件开发经验。
          </text>
          
          <view class="team-contact">
            <text class="contact-title">联系我们</text>
            <view class="contact-info">
              <text class="contact-item">📧 邮箱：support@happy8.com</text>
              <text class="contact-item">💬 微信：happy8service</text>
              <text class="contact-item">🌐 官网：www.happy8.com</text>
            </view>
          </view>
        </view>
      </view>
    </view>

    <!-- 版权信息 -->
    <view class="copyright-section">
      <view class="container">
        <view class="copyright-card card">
          <text class="copyright-text">
            Copyright © {{ currentYear }} Happy8智能预测
          </text>
          <text class="copyright-text">
            All Rights Reserved
          </text>
          
          <view class="legal-links">
            <text class="legal-link" @click="viewPrivacyPolicy">隐私政策</text>
            <text class="legal-divider">|</text>
            <text class="legal-link" @click="viewUserAgreement">用户协议</text>
            <text class="legal-divider">|</text>
            <text class="legal-link" @click="viewDisclaimer">免责声明</text>
          </view>
        </view>
      </view>
    </view>

    <!-- 版本更新日志 -->
    <view class="changelog-section">
      <view class="container">
        <view class="section-header">
          <text class="section-title">更新日志</text>
        </view>
        
        <view class="changelog-list">
          <view 
            v-for="log in changeLogs" 
            :key="log.version"
            class="changelog-item card"
          >
            <view class="log-header">
              <text class="log-version">v{{ log.version }}</text>
              <text class="log-date">{{ log.date }}</text>
            </view>
            <view class="log-changes">
              <text 
                v-for="change in log.changes" 
                :key="change"
                class="log-change"
              >
                • {{ change }}
              </text>
            </view>
          </view>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import appLogo from '@/static/images/logo.png'

const appVersion = ref('1.0.0')

const currentYear = computed(() => new Date().getFullYear())

const features = [
  {
    icon: '🤖',
    title: '智能预测',
    description: '17种先进算法，科学分析历史数据，提供精准预测'
  },
  {
    icon: '📊',
    title: '数据分析',
    description: '实时统计分析，多维度展示号码趋势和规律'
  },
  {
    icon: '💎',
    title: '会员体系',
    description: '三级会员制度，解锁更多高级算法和功能'
  },
  {
    icon: '📱',
    title: '便捷操作',
    description: '简洁易用的界面，一键预测，快速获取结果'
  }
]

const algorithms = [
  {
    name: '频率分析',
    level: 'FREE',
    description: '基于历史开奖频率的统计分析算法'
  },
  {
    name: '热冷分析',
    level: 'FREE',
    description: '分析号码的热度和冷度趋势'
  },
  {
    name: '马尔可夫链',
    level: 'VIP',
    description: '基于状态转移概率的预测模型'
  },
  {
    name: 'LSTM神经网络',
    level: 'PREMIUM',
    description: '长短期记忆网络，捕捉序列中的长期依赖'
  },
  {
    name: 'Transformer',
    level: 'PREMIUM',
    description: '最新的注意力机制模型，专业级预测'
  },
  {
    name: '图神经网络',
    level: 'PREMIUM',
    description: '基于图结构的深度学习预测模型'
  }
]

const stats = [
  {
    label: '累计预测',
    value: '100万+'
  },
  {
    label: '注册用户',
    value: '50万+'
  },
  {
    label: '平均准确率',
    value: '85%'
  },
  {
    label: '算法模型',
    value: '17种'
  }
]

const changeLogs = [
  {
    version: '1.0.0',
    date: '2024-01-15',
    changes: [
      '正式发布Happy8智能预测应用',
      '集成17种预测算法',
      '支持三级会员体系',
      '完善的预测历史管理',
      '微信小程序版本上线'
    ]
  },
  {
    version: '0.9.0',
    date: '2024-01-01',
    changes: [
      '新增深度学习算法模块',
      '优化预测准确率',
      '改进用户界面设计',
      '修复已知问题'
    ]
  }
]

const getLevelName = (level: string) => {
  const levelMap = {
    'FREE': '免费',
    'VIP': 'VIP',
    'PREMIUM': '高级'
  }
  return levelMap[level as keyof typeof levelMap] || level
}

const viewPrivacyPolicy = () => {
  uni.showModal({
    title: '隐私政策',
    content: '隐私政策详情页面正在开发中',
    showCancel: false
  })
}

const viewUserAgreement = () => {
  uni.showModal({
    title: '用户协议',
    content: '用户协议详情页面正在开发中',
    showCancel: false
  })
}

const viewDisclaimer = () => {
  uni.showModal({
    title: '免责声明',
    content: '本应用仅供娱乐参考，预测结果不构成投资建议。请理性对待，谨慎决策。',
    showCancel: false
  })
}
</script>

<style lang="scss" scoped>
.about {
  min-height: 100vh;
  background: $background-color;
  padding-bottom: $spacing-lg;
}

// Logo区域
.logo-section {
  background: linear-gradient(135deg, $primary-color, lighten($primary-color, 10%));
  color: white;
  margin-bottom: $spacing-md;
  
  .logo-card {
    @include flex-center;
    flex-direction: column;
    padding: $spacing-xl;
    
    .app-logo {
      width: 120rpx;
      height: 120rpx;
      margin-bottom: $spacing-md;
      border-radius: $border-radius-lg;
    }
    
    .app-name {
      font-size: $font-size-xl;
      font-weight: $font-weight-bold;
      margin-bottom: $spacing-sm;
    }
    
    .app-version {
      font-size: $font-size-md;
      opacity: 0.8;
      margin-bottom: $spacing-sm;
    }
    
    .app-slogan {
      font-size: $font-size-sm;
      opacity: 0.7;
    }
  }
}

// 产品介绍
.intro-section {
  margin-bottom: $spacing-md;
  
  .intro-card {
    padding: $spacing-md;
    
    .intro-text {
      font-size: $font-size-md;
      color: $text-primary;
      line-height: $line-height-lg;
    }
  }
}

// 核心功能
.features-section {
  margin-bottom: $spacing-md;
  
  .features-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: $spacing-md;
    
    .feature-item {
      @include flex-center;
      flex-direction: column;
      padding: $spacing-lg;
      text-align: center;
      
      .feature-icon {
        font-size: 60rpx;
        margin-bottom: $spacing-md;
      }
      
      .feature-title {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $text-primary;
        margin-bottom: $spacing-sm;
      }
      
      .feature-desc {
        font-size: $font-size-sm;
        color: $text-secondary;
        line-height: $line-height-md;
      }
    }
  }
}

// 算法技术
.algorithms-section {
  margin-bottom: $spacing-md;
  
  .algorithms-list {
    .algorithm-item {
      padding: $spacing-md;
      margin-bottom: $spacing-sm;
      
      .algorithm-header {
        @include flex-between;
        margin-bottom: $spacing-sm;
        
        .algorithm-name {
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
          color: $text-primary;
        }
        
        .algorithm-badge {
          padding: 4rpx 8rpx;
          border-radius: $border-radius-sm;
          font-size: $font-size-xs;
          
          &.free {
            background: rgba($secondary-color, 0.1);
            
            .badge-text { color: $secondary-color; }
          }
          
          &.vip {
            background: rgba($warning-color, 0.1);
            
            .badge-text { color: $warning-color; }
          }
          
          &.premium {
            background: rgba($primary-color, 0.1);
            
            .badge-text { color: $primary-color; }
          }
        }
      }
      
      .algorithm-desc {
        font-size: $font-size-sm;
        color: $text-secondary;
        line-height: $line-height-md;
      }
    }
  }
}

// 数据统计
.stats-section {
  margin-bottom: $spacing-md;
  
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: $spacing-md;
    
    .stat-item {
      @include flex-center;
      flex-direction: column;
      padding: $spacing-lg;
      
      .stat-value {
        font-size: 48rpx;
        font-weight: $font-weight-bold;
        color: $primary-color;
        margin-bottom: $spacing-sm;
      }
      
      .stat-label {
        font-size: $font-size-sm;
        color: $text-secondary;
      }
    }
  }
}

// 团队介绍
.team-section {
  margin-bottom: $spacing-md;
  
  .team-card {
    padding: $spacing-md;
    
    .team-intro {
      font-size: $font-size-md;
      color: $text-primary;
      line-height: $line-height-lg;
      margin-bottom: $spacing-lg;
    }
    
    .team-contact {
      .contact-title {
        font-size: $font-size-md;
        font-weight: $font-weight-bold;
        color: $text-primary;
        display: block;
        margin-bottom: $spacing-md;
      }
      
      .contact-info {
        .contact-item {
          font-size: $font-size-sm;
          color: $text-secondary;
          display: block;
          margin-bottom: $spacing-sm;
          
          &:last-child {
            margin-bottom: 0;
          }
        }
      }
    }
  }
}

// 版权信息
.copyright-section {
  margin-bottom: $spacing-md;
  
  .copyright-card {
    @include flex-center;
    flex-direction: column;
    padding: $spacing-lg;
    text-align: center;
    
    .copyright-text {
      font-size: $font-size-sm;
      color: $text-secondary;
      margin-bottom: $spacing-xs;
    }
    
    .legal-links {
      @include flex-center;
      margin-top: $spacing-md;
      
      .legal-link {
        font-size: $font-size-xs;
        color: $primary-color;
        text-decoration: underline;
      }
      
      .legal-divider {
        font-size: $font-size-xs;
        color: $text-disabled;
        margin: 0 $spacing-sm;
      }
    }
  }
}

// 更新日志
.changelog-section {
  .changelog-list {
    .changelog-item {
      padding: $spacing-md;
      margin-bottom: $spacing-sm;
      
      .log-header {
        @include flex-between;
        margin-bottom: $spacing-sm;
        
        .log-version {
          font-size: $font-size-md;
          font-weight: $font-weight-bold;
          color: $primary-color;
        }
        
        .log-date {
          font-size: $font-size-sm;
          color: $text-secondary;
        }
      }
      
      .log-changes {
        .log-change {
          font-size: $font-size-sm;
          color: $text-primary;
          line-height: $line-height-lg;
          display: block;
          margin-bottom: $spacing-xs;
          
          &:last-child {
            margin-bottom: 0;
          }
        }
      }
    }
  }
}

.section-header {
  padding: $spacing-md 0 $spacing-sm;
  
  .section-title {
    font-size: $font-size-lg;
    font-weight: $font-weight-bold;
    color: $text-primary;
  }
}
</style>
