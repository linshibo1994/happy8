<script setup lang="ts">
import { computed } from 'vue'
import { Crown, Gauge, ShieldCheck, Sparkles } from 'lucide-vue-next'

import MembershipPlanCard, { type MembershipPlanView } from '@/components/membership/MembershipPlanCard.vue'
import OrderStatusList, { type MembershipOrderView } from '@/components/membership/OrderStatusList.vue'
import { useMembershipStore } from '@/stores/membership'

const membershipStore = useMembershipStore()

const quotaPercent = computed(() =>
  Math.round((membershipStore.status.remainingPredictions / Math.max(membershipStore.status.dailyLimit, 1)) * 100),
)

const plans: MembershipPlanView[] = [
  {
    id: 'free',
    level: 'free',
    name: '免费体验',
    priceText: '¥0',
    quotaText: '每日 3 次',
    description: '适合先查看基础预测和历史复盘体验。',
    benefits: ['基础统计算法', '预测历史复盘', '开奖数据查看'],
  },
  {
    id: 'vip',
    level: 'vip',
    name: 'VIP 会员',
    priceText: '¥29 / 月',
    quotaText: '每日 12 次',
    description: '开放马尔可夫、集成学习和概率类算法。',
    benefits: ['批量预测', 'VIP 算法执行', '命中表现追踪', '更多历史窗口'],
    recommended: membershipStore.status.level === 'free',
  },
  {
    id: 'premium',
    level: 'premium',
    name: 'Premium 会员',
    priceText: '¥68 / 月',
    quotaText: '每日 30 次',
    description: '面向重度复盘用户，开放深度学习和高置信度算法。',
    benefits: ['全部算法执行', '高级综合模型', '订单状态追踪', '优先诊断提示'],
    recommended: membershipStore.status.level !== 'premium',
  },
]

const orders: MembershipOrderView[] = [
  {
    id: 'H8-20260607-001',
    planName: 'Premium 会员',
    amountText: '¥68.00',
    status: 'paid',
    createdAt: '2026-06-07 09:30',
    validUntil: '2026-07-07',
  },
  {
    id: 'H8-20260507-014',
    planName: 'VIP 会员',
    amountText: '¥29.00',
    status: 'closed',
    createdAt: '2026-05-07 21:14',
    validUntil: '2026-06-07',
  },
]
</script>

<template>
  <section class="membership-page" aria-labelledby="membership-title">
    <header class="membership-page__hero">
      <div>
        <span class="section-kicker">Membership</span>
        <h2 id="membership-title">会员中心</h2>
        <p>查看当前等级、预测次数、权益和订单状态。升级路径以状态展示为主，不打断算法和历史浏览。</p>
      </div>
      <RouterLink to="/algorithms">查看可用算法</RouterLink>
    </header>

    <section class="membership-page__status" aria-label="当前会员状态">
      <div class="membership-page__status-main">
        <Crown :size="26" aria-hidden="true" />
        <span>当前等级</span>
        <h3>{{ membershipStore.status.levelName }}</h3>
        <p>权益包含：{{ membershipStore.status.benefits.join('、') }}</p>
      </div>

      <div class="membership-page__quota">
        <div>
          <Gauge :size="22" aria-hidden="true" />
          <span>今日剩余次数</span>
        </div>
        <strong>{{ membershipStore.status.remainingPredictions }} / {{ membershipStore.status.dailyLimit }}</strong>
        <div class="membership-page__progress" aria-label="剩余预测次数占比">
          <span :style="{ width: `${quotaPercent}%` }" />
        </div>
      </div>

      <div class="membership-page__next">
        <ShieldCheck :size="22" aria-hidden="true" />
        <span>升级路径</span>
        <p>先确认算法需求和每日次数，再选择套餐；订单状态会在下方独立展示。</p>
      </div>
    </section>

    <section class="membership-page__benefits" aria-label="权益概览">
      <div>
        <Sparkles :size="20" aria-hidden="true" />
        <strong>权益概览</strong>
      </div>
      <ul>
        <li v-for="benefit in membershipStore.status.benefits" :key="benefit">{{ benefit }}</li>
      </ul>
    </section>

    <section class="membership-page__plans" aria-label="套餐列表">
      <MembershipPlanCard
        v-for="plan in plans"
        :key="plan.id"
        :plan="plan"
        :current="plan.level === membershipStore.status.level"
      />
    </section>

    <OrderStatusList :orders="orders" />
  </section>
</template>

<style scoped lang="scss">
.membership-page {
  display: grid;
  gap: 22px;
}

.membership-page__hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}

.membership-page__hero h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 31px;
  line-height: 1.2;
}

.membership-page__hero p {
  max-width: 760px;
  margin: 9px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.membership-page__hero a {
  display: inline-flex;
  min-height: 40px;
  align-items: center;
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  padding: 0 14px;
  font-weight: 800;
  white-space: nowrap;
}

.membership-page__status {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.9fr) minmax(240px, 0.8fr);
  gap: 14px;
}

.membership-page__status > div,
.membership-page__benefits {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 18px;
}

.membership-page__status-main {
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--h8-color-bronze) 14%, transparent), transparent 48%),
    var(--h8-color-surface-strong) !important;
}

.membership-page__status-main svg,
.membership-page__quota svg,
.membership-page__next svg,
.membership-page__benefits svg {
  color: var(--h8-color-cinnabar);
}

.membership-page__status-main span,
.membership-page__quota span,
.membership-page__next span {
  display: block;
  margin-top: 8px;
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.membership-page__status-main h3 {
  margin: 5px 0 8px;
  color: var(--h8-color-bronze);
  font-family: var(--h8-font-title);
  font-size: 28px;
}

.membership-page__status-main p,
.membership-page__next p {
  margin: 0;
  color: var(--h8-color-text-muted);
  line-height: 1.6;
}

.membership-page__quota > div {
  display: flex;
  align-items: center;
  gap: 8px;
}

.membership-page__quota > div span {
  margin: 0;
}

.membership-page__quota strong {
  display: block;
  margin-top: 16px;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 34px;
  line-height: 1;
}

.membership-page__progress {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--h8-color-data-blue) 12%, var(--h8-color-line));
  margin-top: 18px;
}

.membership-page__progress span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--h8-color-data-blue);
}

.membership-page__benefits {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.membership-page__benefits > div {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--h8-color-cinnabar);
}

.membership-page__benefits ul {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.membership-page__benefits li {
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  color: var(--h8-color-text);
  padding: 5px 9px;
  font-size: 13px;
}

.membership-page__plans {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

@media (max-width: 1080px) {
  .membership-page__status,
  .membership-page__plans {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 680px) {
  .membership-page__hero,
  .membership-page__benefits {
    display: grid;
  }

  .membership-page__hero a {
    width: fit-content;
  }
}
</style>
