<template>
  <div class="history-page">
    <DataVBorder :type="1">
      <div class="page-content">
        <div class="page-header">
          <h2 class="page-title">历史数据</h2>
          <div class="header-actions">
            <span class="data-count" v-if="totalItems > 0">共 {{ totalItems }} 条</span>
            <div class="update-actions">
              <button
                class="btn-update"
                @click="appendLatest"
                :disabled="isUpdating || lotteryStore.loading"
              >
                追加最新一期
              </button>
              <div class="append-group">
                <input
                  type="number"
                  v-model.number="appendCount"
                  min="1"
                  max="100"
                  placeholder="5"
                />
                <button
                  class="btn-update btn-update-secondary"
                  @click="appendRecent"
                  :disabled="isUpdating || lotteryStore.loading"
                >
                  追加最近N期
                </button>
              </div>
            </div>
            <button class="btn-export" @click="exportData" :disabled="totalItems === 0">
              导出 CSV
            </button>
          </div>
        </div>

        <!-- 筛选 -->
        <div class="filters">
          <div class="filter-item">
            <label>期数范围</label>
            <input type="number" v-model.number="filters.periodStart" placeholder="起始期号" />
            <span class="filter-separator">-</span>
            <input type="number" v-model.number="filters.periodEnd" placeholder="结束期号" />
          </div>
          <div class="filter-item">
            <label>号码搜索</label>
            <input type="text" v-model="filters.numberSearch" placeholder="输入号码，如: 01,15,33" />
          </div>
          <button class="btn-search" @click="searchHistory">搜索</button>
        </div>

        <!-- 加载状态 -->
        <div v-if="lotteryStore.loading" class="loading-state">
          <span class="loading-text">数据加载中...</span>
        </div>

        <!-- 空数据状态 -->
        <div v-else-if="filteredData.length === 0" class="empty-state">
          <span class="empty-text">暂无匹配数据</span>
        </div>

        <!-- 数据表格 -->
        <template v-else>
          <div class="table-container">
            <table class="data-table">
              <thead>
                <tr>
                  <th>期号</th>
                  <th>开奖日期</th>
                  <th>红球</th>
                  <th>蓝球</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in displayData" :key="item.period">
                  <td class="period-cell">{{ item.period }}</td>
                  <td class="date-cell">{{ item.date }}</td>
                  <td class="balls-cell">
                    <div class="balls-row">
                      <LotteryBall
                        v-for="(num, i) in item.red_balls"
                        :key="'r-' + i"
                        :number="num"
                        type="red"
                        size="sm"
                      />
                    </div>
                  </td>
                  <td class="balls-cell">
                    <LotteryBall :number="item.blue_ball" type="blue" size="sm" />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- 分页 -->
          <div class="pagination">
            <button class="page-btn" :disabled="currentPage <= 1" @click="goToPage(currentPage - 1)">
              上一页
            </button>
            <span class="page-info">
              第 {{ currentPage }} / {{ totalPages }} 页
            </span>
            <button class="page-btn" :disabled="currentPage >= totalPages" @click="goToPage(currentPage + 1)">
              下一页
            </button>
          </div>
        </template>
      </div>
    </DataVBorder>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useLotteryStore } from '@/stores/lottery'
import { useAppStore } from '@/stores/app'
import DataVBorder from '@/components/common/DataVBorder.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'

const lotteryStore = useLotteryStore()
const appStore = useAppStore()

const filters = ref<{
  periodStart: number | ''
  periodEnd: number | ''
  numberSearch: string
}>({
  periodStart: '',
  periodEnd: '',
  numberSearch: ''
})

const currentPage = ref(1)
const pageSize = ref(20)
const appendCount = ref(5)
const isUpdating = ref(false)

// 直接使用 store 中的数据
const historyData = computed(() => lotteryStore.historyData)

const filteredData = computed(() => {
  let result = [...historyData.value]

  const startIssue = filters.value.periodStart === '' ? null : Number(filters.value.periodStart)
  const endIssue = filters.value.periodEnd === '' ? null : Number(filters.value.periodEnd)
  const numberSearch = filters.value.numberSearch.trim()

  if (startIssue !== null || endIssue !== null) {
    result = result.filter((item) => {
      const issue = Number(item.period)
      if (Number.isNaN(issue)) {
        return false
      }
      if (startIssue !== null && issue < startIssue) {
        return false
      }
      if (endIssue !== null && issue > endIssue) {
        return false
      }
      return true
    })
  }

  if (numberSearch) {
    const queryNumbers = numberSearch
      .split(/[\s,，]+/)
      .map((text) => Number(text))
      .filter((value) => Number.isInteger(value) && value >= 1 && value <= 33)

    if (queryNumbers.length > 0) {
      result = result.filter((item) => {
        const allBalls = [...item.red_balls, item.blue_ball]
        return queryNumbers.every((value) => allBalls.includes(value))
      })
    }
  }

  return result
})

const totalItems = computed(() => filteredData.value.length)
const totalPages = computed(() => Math.ceil(totalItems.value / pageSize.value))

const displayData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredData.value.slice(start, end)
})

const goToPage = (page: number) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
  }
}

const searchHistory = () => {
  currentPage.value = 1
}

const exportData = () => {
  if (filteredData.value.length === 0) {
    appStore.notify.warning('当前没有可导出的数据')
    return
  }

  const headers = ['期号', '开奖日期', '红球', '蓝球']
  const rows = filteredData.value.map((item) => [
    item.period,
    item.date,
    item.red_balls.map((num) => String(num).padStart(2, '0')).join(' '),
    String(item.blue_ball).padStart(2, '0')
  ])

  const csvText = [headers, ...rows]
    .map((line) => line.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    .join('\n')

  const blob = new Blob(['\ufeff' + csvText], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ssq_history_${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
  appStore.notify.success(`已导出 ${filteredData.value.length} 条数据`)
}

const appendLatest = async () => {
  if (isUpdating.value) {
    return
  }
  isUpdating.value = true
  try {
    const res = await lotteryStore.refresh()
    if (res.success) {
      appStore.notify.success(res.message || '已追加最新一期数据')
      currentPage.value = 1
    } else {
      appStore.notify.error(res.message || '追加最新一期失败')
    }
  } finally {
    isUpdating.value = false
  }
}

const appendRecent = async () => {
  if (isUpdating.value) {
    return
  }

  const normalizedCount = Math.max(1, Math.min(100, Math.floor(appendCount.value || 5)))
  appendCount.value = normalizedCount

  isUpdating.value = true
  try {
    const res = await lotteryStore.appendRecent(normalizedCount)
    if (res.success) {
      appStore.notify.success(res.message || `已追加最近${normalizedCount}期数据`)
      currentPage.value = 1
    } else {
      appStore.notify.error(res.message || '追加最近数据失败')
    }
  } finally {
    isUpdating.value = false
  }
}

onMounted(async () => {
  await Promise.all([
    lotteryStore.fetchLatest(),
    lotteryStore.fetchHistory(1000)
  ])
})
</script>

<style scoped>
.history-page {
  padding: 24px;
}

.page-content {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  font-size: 20px;
  color: var(--text-primary);
}

.btn-export {
  padding: 10px 20px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.3s;
}

.btn-export:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.btn-export:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}

.data-count {
  font-size: 14px;
  color: var(--text-secondary);
}

.update-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.append-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.append-group input {
  width: 72px;
  padding: 8px 10px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
}

.btn-update {
  padding: 10px 16px;
  background: var(--gradient-primary);
  border: none;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
  transition: all 0.3s;
}

.btn-update:hover:not(:disabled) {
  filter: brightness(1.08);
}

.btn-update:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-update-secondary {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
}

.btn-update-secondary:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.filters {
  display: flex;
  gap: 24px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}

.filter-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-item label {
  font-size: 14px;
  color: var(--text-secondary);
}

.filter-item input {
  padding: 8px 12px;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  width: 120px;
}

.filter-separator {
  color: var(--text-muted);
}

.btn-search {
  padding: 8px 24px;
  background: var(--gradient-primary);
  border: none;
  border-radius: 6px;
  color: white;
  cursor: pointer;
}

.table-container {
  overflow-x: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
}

.data-table th,
.data-table td {
  padding: 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
}

.data-table th {
  background: var(--bg-secondary);
  color: var(--text-secondary);
  font-weight: 500;
}

.data-table tr:hover {
  background: var(--bg-hover);
}

.period-cell {
  color: var(--color-primary);
  font-weight: 500;
}

.date-cell {
  color: var(--text-secondary);
}

.balls-cell {
  padding: 8px 16px;
}

.balls-row {
  display: flex;
  gap: 8px;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-top: 24px;
}

.page-btn {
  padding: 8px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.3s;
}

.page-btn:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  color: var(--text-secondary);
}

.loading-state,
.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

.loading-text {
  color: var(--text-secondary);
  font-size: 16px;
}

.empty-text {
  color: var(--text-muted);
  font-size: 14px;
}
</style>
