<script setup lang="ts">
import { computed, ref } from 'vue'
import { Bell, Clock, Save, Settings, UserRound } from 'lucide-vue-next'

import { useMembershipStore } from '@/stores/membership'
import { usePredictionStore } from '@/stores/prediction'
import { useUiStore } from '@/stores/ui'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()
const membershipStore = useMembershipStore()
const predictionStore = usePredictionStore()
const uiStore = useUiStore()

const nickname = ref(userStore.profile.nickname)
const roleLabel = computed(() => (userStore.isAdmin ? '管理员' : '普通用户'))

const preferences = ref({
  defaultAlgorithm: predictionStore.selectedAlgorithm,
  analysisPeriods: predictionStore.analysisPeriods,
  predictCount: predictionStore.predictCount,
  density: uiStore.density,
  rationalNotice: true,
})

const notifications = ref({
  drawOpened: true,
  predictionReviewed: true,
  quotaWarning: true,
  systemDiagnostics: userStore.isAdmin,
})

const saveMessage = ref('')

const saveProfile = () => {
  userStore.updateProfile({ nickname: nickname.value.trim() || '体验用户' })
  saveMessage.value = '资料已保存到当前会话'
}

const predictionCount = computed(() => predictionStore.history.length)
const averageConfidence = computed(() => {
  if (!predictionStore.history.length) {
    return '0%'
  }

  const total = predictionStore.history.reduce((sum, record) => sum + record.confidence, 0)
  return `${Math.round((total / predictionStore.history.length) * 100)}%`
})
</script>

<template>
  <section class="profile-page" aria-labelledby="profile-title">
    <header class="profile-page__hero">
      <div>
        <span class="section-kicker">Profile</span>
        <h2 id="profile-title">个人中心</h2>
        <p>维护个人资料、预测偏好和通知设置。偏好配置以减少重复输入为目标，不改变预测结果的概率属性。</p>
      </div>
      <button type="button" @click="saveProfile">
        <Save :size="17" aria-hidden="true" />
        保存资料
      </button>
    </header>

    <div v-if="saveMessage" class="profile-page__message" role="status">{{ saveMessage }}</div>

    <section class="profile-page__layout">
      <form class="profile-page__profile" @submit.prevent="saveProfile">
        <div class="profile-page__section-title">
          <UserRound :size="20" aria-hidden="true" />
          <h3>资料</h3>
        </div>

        <div class="profile-page__avatar" aria-hidden="true">
          {{ userStore.profile.nickname.slice(0, 1) }}
        </div>

        <label>
          <span>昵称</span>
          <input v-model="nickname" type="text" autocomplete="nickname" />
        </label>

        <label>
          <span>角色</span>
          <input :value="roleLabel" type="text" disabled />
        </label>

        <label>
          <span>会员等级</span>
          <input :value="membershipStore.status.levelName" type="text" disabled />
        </label>
      </form>

      <section class="profile-page__stats" aria-label="账户统计">
        <div class="profile-page__section-title">
          <Clock :size="20" aria-hidden="true" />
          <h3>统计</h3>
        </div>

        <dl>
          <div>
            <dt>预测记录</dt>
            <dd>{{ predictionCount }}</dd>
          </div>
          <div>
            <dt>平均置信度</dt>
            <dd>{{ averageConfidence }}</dd>
          </div>
          <div>
            <dt>今日剩余</dt>
            <dd>{{ membershipStore.status.remainingPredictions }}</dd>
          </div>
          <div>
            <dt>默认分析期数</dt>
            <dd>{{ preferences.analysisPeriods }}</dd>
          </div>
        </dl>
      </section>
    </section>

    <section class="profile-page__settings">
      <div class="profile-page__section-title">
        <Settings :size="20" aria-hidden="true" />
        <h3>偏好设置</h3>
      </div>

      <div class="profile-page__form-grid">
        <label>
          <span>默认算法</span>
          <select v-model="preferences.defaultAlgorithm">
            <option value="frequency">频率分析</option>
            <option value="hot_cold">冷热分析</option>
            <option value="markov">马尔可夫链</option>
            <option value="ensemble">集成学习</option>
            <option value="transformer">Transformer</option>
          </select>
        </label>

        <label>
          <span>分析期数</span>
          <input v-model.number="preferences.analysisPeriods" type="number" min="10" max="200" />
        </label>

        <label>
          <span>预测个数</span>
          <input v-model.number="preferences.predictCount" type="number" min="1" max="20" />
        </label>

        <label>
          <span>表格密度</span>
          <select v-model="preferences.density">
            <option value="comfortable">舒适</option>
            <option value="compact">紧凑</option>
          </select>
        </label>
      </div>

      <label class="profile-page__switch">
        <input v-model="preferences.rationalNotice" type="checkbox" />
        <span>预测结果区固定展示理性使用提示</span>
      </label>
    </section>

    <section class="profile-page__settings">
      <div class="profile-page__section-title">
        <Bell :size="20" aria-hidden="true" />
        <h3>通知设置</h3>
      </div>

      <div class="profile-page__switch-list">
        <label>
          <input v-model="notifications.drawOpened" type="checkbox" />
          <span>最新开奖同步后提醒</span>
        </label>
        <label>
          <input v-model="notifications.predictionReviewed" type="checkbox" />
          <span>预测记录完成复盘后提醒</span>
        </label>
        <label>
          <input v-model="notifications.quotaWarning" type="checkbox" />
          <span>剩余预测次数低于 3 次时提醒</span>
        </label>
        <label>
          <input v-model="notifications.systemDiagnostics" type="checkbox" />
          <span>系统诊断出现异常时提醒</span>
        </label>
      </div>
    </section>
  </section>
</template>

<style scoped lang="scss">
.profile-page {
  display: grid;
  gap: 22px;
}

.profile-page__hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
}

.profile-page__hero h2 {
  margin: 4px 0 0;
  font-family: var(--h8-font-title);
  font-size: 31px;
  line-height: 1.2;
}

.profile-page__hero p {
  max-width: 760px;
  margin: 9px 0 0;
  color: var(--h8-color-text-muted);
  line-height: 1.7;
}

.profile-page__hero button {
  display: inline-flex;
  min-height: 40px;
  align-items: center;
  gap: 8px;
  border: 0;
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-cinnabar);
  color: #fff;
  padding: 0 14px;
  font-weight: 800;
  cursor: pointer;
  white-space: nowrap;
}

.profile-page__message {
  border: 1px solid color-mix(in srgb, var(--h8-color-turquoise) 42%, var(--h8-color-line));
  border-radius: var(--h8-radius-control);
  background: color-mix(in srgb, var(--h8-color-turquoise) 8%, var(--h8-color-surface));
  color: var(--h8-color-turquoise);
  padding: 10px 12px;
}

.profile-page__layout {
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
  gap: 14px;
}

.profile-page__profile,
.profile-page__stats,
.profile-page__settings {
  display: grid;
  gap: 16px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-panel);
  background: var(--h8-color-surface-strong);
  box-shadow: 0 12px 36px var(--h8-color-shadow);
  padding: 18px;
}

.profile-page__section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--h8-color-cinnabar);
}

.profile-page__section-title h3 {
  margin: 0;
  color: var(--h8-color-text);
  font-family: var(--h8-font-title);
  font-size: 21px;
}

.profile-page__avatar {
  display: grid;
  width: 72px;
  height: 72px;
  place-items: center;
  border-radius: 50%;
  background: var(--h8-color-cinnabar);
  color: #fff;
  font-family: var(--h8-font-title);
  font-size: 30px;
  font-weight: 800;
}

.profile-page label {
  display: grid;
  gap: 7px;
}

.profile-page label > span {
  color: var(--h8-color-text-muted);
  font-size: 13px;
  font-weight: 700;
}

.profile-page input,
.profile-page select {
  min-height: 40px;
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  background: var(--h8-color-surface);
  color: var(--h8-color-text);
  padding: 0 10px;
}

.profile-page input:disabled {
  color: var(--h8-color-text-muted);
}

.profile-page__stats dl {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}

.profile-page__stats dl div {
  border: 1px solid var(--h8-color-line);
  border-radius: var(--h8-radius-control);
  padding: 12px;
}

.profile-page__stats dt {
  color: var(--h8-color-text-muted);
  font-size: 13px;
}

.profile-page__stats dd {
  margin: 7px 0 0;
  color: var(--h8-color-cinnabar);
  font-family: var(--h8-font-number);
  font-size: 26px;
  font-weight: 800;
}

.profile-page__form-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.profile-page__switch,
.profile-page__switch-list label {
  display: flex;
  grid-template-columns: none;
  align-items: center;
  gap: 9px;
}

.profile-page__switch input,
.profile-page__switch-list input {
  width: 18px;
  height: 18px;
  min-height: 0;
  accent-color: var(--h8-color-cinnabar);
}

.profile-page__switch-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

@media (max-width: 980px) {
  .profile-page__layout,
  .profile-page__form-grid,
  .profile-page__switch-list {
    grid-template-columns: 1fr;
  }

  .profile-page__stats dl {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 620px) {
  .profile-page__hero {
    display: grid;
  }

  .profile-page__hero button {
    width: fit-content;
  }

  .profile-page__stats dl {
    grid-template-columns: 1fr;
  }
}
</style>
