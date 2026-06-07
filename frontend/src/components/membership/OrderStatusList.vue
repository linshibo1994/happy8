<script setup lang="ts">
export interface MembershipOrderView {
  id: string
  planName: string
  amountText: string
  status: 'paid' | 'pending' | 'closed'
  createdAt: string
  validUntil: string
}

defineProps<{
  orders: MembershipOrderView[]
}>()

const statusText: Record<MembershipOrderView['status'], string> = {
  paid: '已生效',
  pending: '待确认',
  closed: '已关闭',
}
</script>

<template>
  <section class="order-list" aria-label="订单状态">
    <div class="order-list__header">
      <h3>订单状态</h3>
      <p>仅展示订单结果、金额和有效期，不处理真实支付参数。</p>
    </div>

    <ul v-if="orders.length">
      <li v-for="order in orders" :key="order.id">
        <span>
          <strong>{{ order.planName }}</strong>
          <small>{{ order.createdAt }} / {{ order.id }}</small>
        </span>
        <span class="order-list__amount">{{ order.amountText }}</span>
        <span class="order-list__valid">有效期至 {{ order.validUntil }}</span>
        <span class="order-list__status" :class="`order-list__status--${order.status}`">
          {{ statusText[order.status] }}
        </span>
      </li>
    </ul>

    <div v-else class="order-list__empty">
      <strong>暂无订单</strong>
      <p>选择一个套餐后，这里会显示订单状态和有效期。</p>
    </div>
  </section>
</template>

<style scoped lang="scss">
.order-list {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 18px;
}

.order-list__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid var(--h8-color-line);
  padding-bottom: 14px;
}

.order-list h3 {
  margin: 0;
  font-family: var(--h8-font-title);
  font-size: 21px;
}

.order-list p {
  margin: 0;
  color: var(--h8-color-text-muted);
  line-height: 1.55;
}

.order-list ul {
  display: grid;
  gap: 0;
  margin: 0;
  padding: 0;
  list-style: none;
}

.order-list li {
  display: grid;
  grid-template-columns: minmax(180px, 1fr) auto auto auto;
  gap: 12px;
  align-items: center;
  border-bottom: 1px solid var(--h8-color-line);
  padding: 14px 0;
}

.order-list li:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.order-list strong {
  display: block;
}

.order-list small {
  display: block;
  margin-top: 4px;
  color: var(--h8-color-text-muted);
  font-size: 12px;
}

.order-list__amount,
.order-list__valid {
  font-family: var(--h8-font-number);
  white-space: nowrap;
}

.order-list__valid {
  color: var(--h8-color-text-muted);
}

.order-list__status {
  justify-self: end;
  border: 1px solid var(--h8-color-line);
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}

.order-list__status--paid {
  border-color: color-mix(in srgb, var(--h8-color-turquoise) 48%, var(--h8-color-line));
  color: var(--h8-color-turquoise);
}

.order-list__status--pending {
  border-color: color-mix(in srgb, var(--h8-color-risk-orange) 50%, var(--h8-color-line));
  color: var(--h8-color-risk-orange);
}

.order-list__status--closed {
  color: var(--h8-color-text-muted);
}

.order-list__empty {
  display: grid;
  gap: 6px;
  padding-top: 14px;
}

@media (max-width: 760px) {
  .order-list__header,
  .order-list li {
    display: grid;
    grid-template-columns: 1fr;
  }

  .order-list__status {
    justify-self: start;
  }
}
</style>
