// 全局配置
export const wotConfig = {
  // 主题配置
  theme: {
    // 主色调
    colorPrimary: '#d32f2f',
    colorSuccess: '#4caf50',
    colorWarning: '#ff9800',
    colorDanger: '#f44336',
    colorInfo: '#2196f3',

    // 文本颜色
    colorText: '#333333',
    colorTextSecondary: '#666666',
    colorTextDisabled: '#999999',

    // 背景色
    colorBackground: '#f5f5f5',
    colorSurface: '#ffffff',

    // 边框色
    colorBorder: '#e0e0e0',
    colorBorderLight: '#f0f0f0',

    // 圆角
    radiusSmall: '4rpx',
    radiusMedium: '8rpx',
    radiusLarge: '12rpx',

    // 间距
    spacingSmall: '8rpx',
    spacingMedium: '16rpx',
    spacingLarge: '24rpx',

    // 字体大小
    fontSizeSmall: '24rpx',
    fontSizeMedium: '28rpx',
    fontSizeLarge: '32rpx',
  },

  // 组件默认配置
  components: {
    // Button 按钮
    button: {
      height: '88rpx',
      borderRadius: '8rpx',
      fontSize: '28rpx',
    },

    // Cell 单元格
    cell: {
      padding: '24rpx 32rpx',
      fontSize: '28rpx',
    },

    // Input 输入框
    input: {
      height: '88rpx',
      padding: '0 24rpx',
      fontSize: '28rpx',
      borderRadius: '8rpx',
    },

    // Toast 提示
    toast: {
      fontSize: '28rpx',
      borderRadius: '8rpx',
    },

    // Dialog 对话框
    dialog: {
      borderRadius: '16rpx',
      titleFontSize: '32rpx',
      contentFontSize: '28rpx',
    },

    // Popup 弹出层
    popup: {
      borderRadius: '16rpx 16rpx 0 0',
    },
  },
}

// 导出配置
export default wotConfig
