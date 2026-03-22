# Happy8 小程序前端开发文档

## 项目概述

Happy8 智能预测小程序是一个基于 uni-app 开发的微信小程序，集成了17种不同的彩票预测算法，提供完整的用户认证、会员体系和支付功能。

## 技术栈

- **框架**: uni-app (Vue 3 + Composition API)
- **语言**: TypeScript
- **UI库**: Wot Design Uni
- **状态管理**: Pinia
- **样式**: SCSS
- **构建工具**: Vite

## 项目结构

```
frontend/
├── src/
│   ├── components/          # 自定义组件
│   │   ├── Loading.vue     # 加载组件
│   │   ├── EmptyState.vue  # 空状态组件
│   │   ├── NumberBall.vue  # 号码球组件
│   │   └── StatCard.vue    # 统计卡片组件
│   ├── pages/              # 页面文件
│   │   ├── index/          # 首页
│   │   ├── predict/        # 预测页面
│   │   ├── history/        # 历史页面
│   │   ├── member/         # 会员页面
│   │   └── profile/        # 个人中心
│   ├── services/           # API服务层
│   │   ├── api.ts         # 基础API服务
│   │   ├── user.ts        # 用户相关API
│   │   ├── membership.ts  # 会员相关API
│   │   ├── algorithm.ts   # 算法相关API
│   │   ├── prediction.ts  # 预测相关API
│   │   └── lottery.ts     # 开奖相关API
│   ├── stores/            # 状态管理
│   │   ├── app.ts        # 应用全局状态
│   │   ├── user.ts       # 用户状态
│   │   ├── member.ts     # 会员状态
│   │   └── predict.ts    # 预测状态
│   ├── types/            # TypeScript类型定义
│   │   └── index.ts      # 全局类型
│   ├── constants/        # 常量配置
│   │   └── index.ts      # 全局常量
│   ├── uni.scss          # 全局样式变量
│   ├── App.vue           # 应用根组件
│   └── main.ts           # 应用入口
├── pages.json            # 页面配置
├── package.json          # 依赖配置
└── tsconfig.json         # TypeScript配置
```

## 核心功能

### 1. 用户认证系统
- 微信登录集成
- JWT token管理
- 自动token刷新
- 登录状态持久化

### 2. 会员体系
- 三级会员系统（FREE/VIP/PREMIUM）
- 微信支付集成
- 使用权限控制
- 预测次数限制

### 3. 预测系统
- 17种预测算法
- 算法权限管理
- 预测参数配置
- 结果置信度显示

### 4. 历史管理
- 预测历史记录
- 筛选和统计
- 结果分析
- 命中率计算

## 开发指南

### 环境要求

- Node.js >= 16.0.0
- npm >= 7.0.0
- HBuilderX (推荐)

### 本地开发

1. 安装依赖
```bash
npm install
```

2. 启动开发服务器
```bash
npm run dev:mp-weixin
```

3. 使用微信开发者工具打开 `dist/dev/mp-weixin` 目录

### 项目配置

#### API配置
在 `src/constants/index.ts` 中配置API地址：

```typescript
export const API_CONFIG = {
  BASE_URL: 'https://api.your-domain.com',
  TIMEOUT: 10000,
  MAX_RETRY: 3
}
```

#### 微信小程序配置
在 `manifest.json` 中配置小程序基本信息：

```json
{
  "mp-weixin": {
    "appid": "your-app-id",
    "setting": {
      "urlCheck": false
    }
  }
}
```

### 状态管理

使用 Pinia 进行状态管理，主要包含：

- **useAppStore**: 应用全局状态（主题、网络状态等）
- **useUserStore**: 用户状态（登录信息、个人资料）
- **useMembershipStore**: 会员状态（会员信息、订单等）
- **usePredictStore**: 预测状态（算法、历史记录等）

### API服务

所有API调用通过服务层封装，支持：

- 自动认证头添加
- Token自动刷新
- 统一错误处理
- 请求重试机制

### 组件开发

#### 自定义组件
- **Loading**: 加载状态组件
- **EmptyState**: 空状态展示
- **NumberBall**: 彩票号码显示
- **StatCard**: 统计数据卡片

#### 组件使用示例

```vue
<template>
  <NumberBall 
    :number="8" 
    size="large" 
    type="zone"
    @click="selectNumber"
  />
</template>
```

### 样式规范

#### SCSS变量
使用全局SCSS变量保持设计一致性：

```scss
// 颜色
$primary-color: #d32f2f;
$secondary-color: #666666;

// 间距
$spacing-sm: 20rpx;
$spacing-md: 30rpx;
$spacing-lg: 40rpx;

// 字体
$font-size-sm: 28rpx;
$font-size-md: 32rpx;
$font-size-lg: 36rpx;
```

#### 混入使用
使用预定义混入提高开发效率：

```scss
.card {
  @include card;
  @include flex-center;
}
```

### 类型定义

完整的TypeScript类型定义，包含：

- API响应类型
- 业务数据模型
- 组件Props类型
- Store状态类型

### 错误处理

统一的错误处理机制：

```typescript
try {
  const result = await api.predict(params)
} catch (error) {
  console.error('预测失败:', error)
  uni.showToast({
    title: '操作失败，请重试',
    icon: 'error'
  })
}
```

## 构建部署

### 测试环境
```bash
npm run build:mp-weixin
```

### 生产环境
```bash
npm run build:mp-weixin --mode production
```

### 代码检查
```bash
npm run lint
npm run type-check
```

## 性能优化

### 图片优化
- 使用WebP格式
- 图片懒加载
- 适当的图片尺寸

### 代码分割
- 路由级别的代码分割
- 按需加载组件
- 第三方库按需引入

### 缓存策略
- API响应缓存
- 静态资源缓存
- 用户数据本地存储

## 调试技巧

### 开发工具
- 使用微信开发者工具的调试功能
- Vue DevTools（H5平台）
- 网络请求监控

### 日志管理
```typescript
if (process.env.NODE_ENV === 'development') {
  console.log('调试信息:', data)
}
```

### 真机调试
- 开启真机调试模式
- 使用vconsole进行移动端调试

## 注意事项

### 微信小程序限制
- 包大小限制（主包2MB，分包2MB）
- 网络请求域名需要配置
- 某些API需要用户授权

### 兼容性
- 支持微信版本 >= 7.0.0
- 基础库版本 >= 2.12.0
- iOS/Android兼容性测试

### 安全考虑
- 敏感数据加密传输
- 用户隐私保护
- 防止XSS攻击

## 常见问题

### Q: 如何添加新的预测算法？
A: 在backend添加算法实现，然后在frontend的算法列表中配置即可。

### Q: 如何修改会员权限？
A: 在 `src/constants/index.ts` 中的 MEMBERSHIP 配置中修改。

### Q: 如何添加新的支付方式？
A: 在membership store中扩展支付方法，并在后端配置相应的支付接口。

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 完整的预测功能
- 会员系统集成
- 微信支付支持