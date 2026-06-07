<script setup lang="ts">
import { computed } from 'vue'
import { Activity, AlertTriangle, CheckCircle2, Database, RefreshCcw, ShieldCheck, Wifi } from 'lucide-vue-next'

import { useAlgorithmStore } from '@/stores/algorithm'
import { useLotteryStore } from '@/stores/lottery'
import { useUserStore } from '@/stores/user'

type DiagnosticStatus = 'healthy' | 'warning' | 'down'

interface DiagnosticItem {
  id: string
  name: string
  status: DiagnosticStatus
  value: string
  detail: string
  action: string
}

const userStore = useUserStore()
const lotteryStore = useLotteryStore()
const algorithmStore = useAlgorithmStore()

const statusText: Record<DiagnosticStatus, string> = {
  healthy: '正常',
  warning: '需关注',
  down: '异常',
}

const statusIcon = {
  healthy: CheckCircle2,
  warning: AlertTriangle,
  down: AlertTriangle,
}

const dataSyncItems = computed<DiagnosticItem[]>(() => [
  {
    id: 'latest-draw',
    name: '最新开奖同步',
    status: 'healthy',
    value: lotteryStore.latestResult.issue,
    detail: `数据更新时间 ${new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(lotteryStore.lastUpdatedAt))}`,
    action: '保持自动同步',
  },
  {
    id: 'history-window',
    name: '历史数据窗口',
    status: lotteryStore.history.length >= 100 ? 'healthy' : 'warning',
    value: `${lotteryStore.history.length} 期`,
    detail: '深度学习和高级综合算法建议保留 100 期以上数据。',
    action: '同步更多开奖历史',
  },
])

const algorithmItems = computed<DiagnosticItem[]>(() => [
  {
    id: 'algorithm-enabled',
    name: '算法可用数',
    status: 'healthy',
    value: `${algorithmStore.enabledAlgorithms.length} / ${algorithmStore.algorithms.length}`,
    detail: '当前核心算法均可展示，执行权限由会员等级控制。',
    action: '查看算法中心',
  },
  {
    id: 'premium-latency',
    name: '高复杂度算法耗时',
    status: 'warning',
    value: '3.7s 平均',
    detail: 'Transformer、GNN、超级预测器耗时较高，移动端需保持阶段进度反馈。',
    action: '观察批量预测耗时',
  },
  {
    id: 'cache-hit',
    name: '预测缓存命中',
    status: 'healthy',
    value: '24%',
    detail: '缓存用于降低重复预测等待，不改变历史复盘结果。',
    action: '复盘缓存记录',
  },
])

const apiItems: DiagnosticItem[] = [
  {
    id: 'lottery-api',
    name: '开奖接口',
    status: 'healthy',
    value: '128ms',
    detail: 'GET /api/v1/lottery/latest 响应稳定。',
    action: '继续监控',
  },
  {
    id: 'prediction-api',
    name: '预测接口',
    status: 'healthy',
    value: '940ms',
    detail: 'POST /api/v1/predictions/predict 处于正常响应区间。',
    action: '保留超时重试',
  },
  {
    id: 'membership-api',
    name: '会员接口',
    status: 'warning',
    value: '状态展示',
    detail: '前端仅展示订单状态，不处理真实支付敏感参数。',
    action: '上线前做安全 Review',
  },
]

const allItems = computed(() => [...dataSyncItems.value, ...algorithmItems.value, ...apiItems])
const warningCount = computed(() => allItems.value.filter((item) => item.status === 'warning').length)
const downCount = computed(() => allItems.value.filter((item) => item.status === 'down').length)
</script>

<template>
  <section class="system-page" aria-labelledby="system-title">
    <header class="system-page__hero">
      <div>
        <span class="section-kicker">System Diagnostics</span>
        <h2 id="system-title">系统诊断</h2>
        <p>管理员入口风格的状态看板，集中查看数据同步、算法诊断和接口状态。这里仅展示状态，不执行生产同步操作。</p>
      </div>
      <div class="system-page__admin-badge" :class="{ 'system-page__admin-badge--muted': !userStore.isAdmin }">
        <ShieldCheck :size="18" aria-hidden="true" />
        <span>{{ userStore.isAdmin ? '管理员视图' : '体验用户预览' }}</span>
      </div>
    </header>

    <section class="system-page__overview" aria-label="系统概览">
      <div>
        <Activity :size="22" aria-hidden="true" />
        <span>总体状态</span>
        <strong>{{ downCount ? '异常' : warningCount ? '需关注' : '正常' }}</strong>
      </div>
      <div>
        <AlertTriangle :size="22" aria-hidden="true" />
        <span>关注项</span>
        <strong>{{ warningCount }}</strong>
      </div>
      <div>
        <Database :size="22" aria-hidden="true" />
        <span>当前期号</span>
        <strong>{{ lotteryStore.latestResult.issue }}</strong>
      </div>
      <div>
        <Wifi :size="22" aria-hidden="true" />
        <span>接口异常</span>
        <strong>{{ downCount }}</strong>
      </div>
    </section>

    <div class="system-page__grid">
      <section class="system-page__panel">
        <div class="system-page__panel-title">
          <Database :size="20" aria-hidden="true" />
          <h3>数据同步</h3>
        </div>
        <ul>
          <li v-for="item in dataSyncItems" :key="item.id" :class="`system-page__item--${item.status}`">
            <component :is="statusIcon[item.status]" :size="18" aria-hidden="true" />
            <span>
              <strong>{{ item.name }}</strong>
              <small>{{ item.detail }}</small>
            </span>
            <em>{{ item.value }}</em>
            <button type="button">{{ item.action }}</button>
          </li>
        </ul>
      </section>

      <section class="system-page__panel">
        <div class="system-page__panel-title">
          <Activity :size="20" aria-hidden="true" />
          <h3>算法诊断</h3>
        </div>
        <ul>
          <li v-for="item in algorithmItems" :key="item.id" :class="`system-page__item--${item.status}`">
            <component :is="statusIcon[item.status]" :size="18" aria-hidden="true" />
            <span>
              <strong>{{ item.name }}</strong>
              <small>{{ item.detail }}</small>
            </span>
            <em>{{ item.value }}</em>
            <RouterLink :to="item.id === 'algorithm-enabled' ? '/algorithms' : '/history'">
              {{ item.action }}
            </RouterLink>
          </li>
        </ul>
      </section>
    </div>

    <section class="system-page__panel">
      <div class="system-page__panel-title">
        <Wifi :size="20" aria-hidden="true" />
        <h3>接口状态</h3>
        <button type="button">
          <RefreshCcw :size="16" aria-hidden="true" />
          刷新状态
        </button>
      </div>

      <div class="system-page__api-table">
        <div class="system-page__api-row system-page__api-row--head">
          <span>接口</span>
          <span>状态</span>
          <span>指标</span>
          <span>说明</span>
          <span>下一步</span>
        </div>
        <div v-for="item in apiItems" :key="item.id" class="system-page__api-row">
          <span>{{ item.name }}</span>
          <strong :class="`system-page__status--${item.status}`">{{ statusText[item.status] }}</strong>
          <em>{{ item.value }}</em>
          <span>{{ item.detail }}</span>
          <span>{{ item.action }}</span>
        </div>
      </div>
    </section>

    <section v-if="warningCount || downCount" class="system-page__empty-action">
      <h3>有诊断项需要跟进</h3>
      <p>优先处理历史数据窗口和会员接口安全 Review，再执行上线前全链路验证。</p>
      <RouterLink to="/history">查看复盘记录</RouterLink>
    </section>
  </section>
</template>

<style scoped lang="scss">
.system-page {
  display: grid;
  gap: 22px;
}

.system-page__hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}

.system-page__hero h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 31px;
  line-height: 1.2;
}

.system-page__hero p {
  max-width: 760px;
  margin: 9px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.system-page__admin-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
  border: 1px solid color-mix(in srgb, var(--h8-color-cinnabar) 40%, var(--h8-color-line));
  border-radius: 999px;
  background: var(--h8-color-cinnabar);
  color: #fff;
  padding: 8px 12px;
  font-weight: 800;
  white-space: nowrap;
}

.system-page__admin-badge--muted {
  border-color: var(--h8-color-line);
  background: var(--h8-color-surface-strong);
  color: var(--h8-color-text-muted);
}

.system-page__overview {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.system-page__overview div,
.system-page__panel,
.system-page__empty-action {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
}

.system-page__overview div {
  display: grid;
  gap: 8px;
  padding: 16px;
}

.system-page__overview svg {
  color: var(--h8-color-cinnabar);
}

.system-page__overview span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.system-page__overview strong {
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 25px;
}

.system-page__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.system-page__panel {
  min-width: 0;
  padding: 18px;
}

.system-page__panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.system-page__panel-title h3 {
  flex: 1;
  margin: 0;
  font-family: var(--h8-font-title);
  font-size: 21px;
}

.system-page__panel-title svg {
  color: var(--h8-color-cinnabar);
}

.system-page__panel-title button,
.system-page li button,
.system-page li a,
.system-page__empty-action a {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  justify-content: center;
  gap: 7px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 0 10px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}

.system-page ul {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.system-page li {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto auto;
  gap: 12px;
  align-items: center;
  border-bottom: 1px solid var(--h8-color-line);
  padding: 14px 0;
}

.system-page li:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.system-page li > svg {
  color: var(--h8-color-turquoise);
}

.system-page__item--warning > svg,
.system-page__status--warning {
  color: var(--h8-color-risk-orange);
}

.system-page__item--down > svg,
.system-page__status--down {
  color: var(--h8-color-cinnabar);
}

.system-page__status--healthy {
  color: var(--h8-color-turquoise);
}

.system-page li strong {
  display: block;
}

.system-page li small {
  display: block;
  margin-top: 4px;
  color: var(--h8-color-text-muted);
  line-height: 1.45;
}

.system-page li em,
.system-page__api-row em {
  color: var(--h8-color-data-blue);
  font-family: var(--h8-font-number);
  font-style: normal;
  font-weight: 800;
  white-space: nowrap;
}

.system-page__api-table {
  overflow: auto;
}

.system-page__api-row {
  display: grid;
  grid-template-columns: 140px 90px 110px minmax(260px, 1fr) 170px;
  gap: 12px;
  min-width: 880px;
  border-bottom: 1px solid var(--h8-color-line);
  padding: 13px 0;
}

.system-page__api-row:last-child {
  border-bottom: 0;
}

.system-page__api-row--head {
  color: var(--h8-color-text-muted);
  font-size: 12px;
  font-weight: 800;
}

.system-page__empty-action {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  border-style: dashed;
  padding: 18px;
}

.system-page__empty-action h3,
.system-page__empty-action p {
  width: 100%;
  margin: 0;
}

.system-page__empty-action p {
  color: var(--h8-color-text-muted);
}

.system-page__empty-action a {
  border-color: var(--h8-color-cinnabar);
  background: var(--h8-color-cinnabar);
  color: #fff;
}

@media (max-width: 1080px) {
  .system-page__overview,
  .system-page__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .system-page__hero {
    display: grid;
  }

  .system-page__overview,
  .system-page__grid {
    grid-template-columns: 1fr;
  }

  .system-page li {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .system-page li em,
  .system-page li button,
  .system-page li a {
    grid-column: 2;
    justify-self: start;
  }
}
</style>
