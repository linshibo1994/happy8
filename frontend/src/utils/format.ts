// 格式化日期
export const formatDate = (date: string | number | Date, format: string = 'YYYY-MM-DD'): string => {
  const d = new Date(date)
  const year = d.getFullYear()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  const hours = String(d.getHours()).padStart(2, '0')
  const minutes = String(d.getMinutes()).padStart(2, '0')
  const seconds = String(d.getSeconds()).padStart(2, '0')

  return format
    .replace('YYYY', String(year))
    .replace('MM', month)
    .replace('DD', day)
    .replace('HH', hours)
    .replace('mm', minutes)
    .replace('ss', seconds)
}

// 格式化数字 (添加千位分隔符)
export const formatNumber = (num: number): string => {
  return num.toLocaleString('zh-CN')
}

// 格式化金额 (单位: 元/万/亿)
export const formatMoney = (money: number): string => {
  if (money >= 100000000) {
    return (money / 100000000).toFixed(2) + '亿'
  } else if (money >= 10000) {
    return (money / 10000).toFixed(2) + '万'
  }
  return formatNumber(money)
}

// 格式化期号
export const formatIssueCode = (code: string | number): string => {
  const p = String(code)
  if (p.length === 7) {
    return `${p.slice(0, 4)}年第${p.slice(4)}期`
  }
  return p
}

// 格式化球号码 (不足两位补0)
export const formatBallNumber = (num: number): string => {
  return num.toString().padStart(2, '0')
}

// 格式化百分比
export const formatPercent = (value: number, decimals = 1): string => {
  return `${(value * 100).toFixed(decimals)}%`
}

// 格式化置信度
export const formatConfidence = (confidence: number): string => {
  if (confidence >= 0.8) return '高'
  if (confidence >= 0.6) return '中'
  return '低'
}

// 获取置信度颜色
export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return 'var(--color-success)'
  if (confidence >= 0.6) return 'var(--color-warning)'
  return 'var(--color-error)'
}
