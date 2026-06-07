import { expect, test } from '@playwright/test'

test('加载工作台骨架', async ({ page }) => {
  await page.goto('/')

  await expect(page).toHaveTitle(/Happy8/)
  await expect(page.getByRole('heading', { name: '工作台', exact: true })).toBeVisible()
  await expect(page.getByLabel('桌面主导航')).toContainText('预测')
})

test('移动端布局保留主导航和核心入口', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')

  await expect(page.getByLabel('移动主导航')).toBeVisible()
  await expect(page.getByLabel('移动主导航')).toContainText('工作台')
  await expect(page.getByLabel('移动主导航')).toContainText('预测')
  await expect(page.getByRole('heading', { name: '工作台', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: '最新开奖', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /一键预测|预测中|再次预测/ })).toBeVisible()
})
