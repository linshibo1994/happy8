const fs = require('fs');
const path = require('path');

console.log('🔧 开始修复 uni-app bugs...\n');

try {
  // 1. 修复 easycom.js
  console.log('1️⃣ 修复 easycom.js debounce 调用...');
  const easycomPath = path.join(__dirname, '../node_modules/@dcloudio/vite-plugin-uni/dist/configureServer/easycom.js');
  if (fs.existsSync(easycomPath)) {
    let easycomContent = fs.readFileSync(easycomPath, 'utf8');
    if (easycomContent.includes('debounce(refresh, 100)') && !easycomContent.includes('clearTimeout: global.clearTimeout')) {
      easycomContent = easycomContent.replace(
        'uni_shared_1.debounce(refresh, 100)',
        'uni_shared_1.debounce(refresh, 100, { clearTimeout: global.clearTimeout || clearTimeout, setTimeout: global.setTimeout || setTimeout })'
      );
      fs.writeFileSync(easycomPath, easycomContent);
      console.log('   ✅ easycom.js 已修复\n');
    } else {
      console.log('   ✅ easycom.js 已经修复\n');
    }
  } else {
    console.log('   ⚠️  找不到 easycom.js\n');
  }

  // 2. 创建缺失的 SSR 文件
  console.log('2️⃣ 创建缺失的 SSR 文件...');
  const ssrDir = path.join(__dirname, '../node_modules/@dcloudio/vite-plugin-uni/lib/ssr');
  if (!fs.existsSync(ssrDir)) {
    fs.mkdirSync(ssrDir, { recursive: true });
  }

  const entryServerPath = path.join(ssrDir, 'entry-server.js');
  fs.writeFileSync(entryServerPath, `import { createSSRApp } from 'vue'
import App from './App.vue'

export function createApp() {
  const app = createSSRApp(App)
  return { app }
}

export async function render() {
  const { app } = createApp()
  return app }
`);

  const definePath = path.join(ssrDir, 'define.js');
  fs.writeFileSync(definePath, `const defines = __DEFINES__
const unit = __UNIT__
const unitRatio = __UNIT_RATIO__
const unitPrecision = __UNIT_PRECISION__

export { defines, unit, unitRatio, unitPrecision }
`);

  console.log('   ✅ SSR 文件已创建\n');

  // 3. 修复 vue external 问题
  console.log('3️⃣ 修复 "vue cannot be marked as external" 问题...');
  const configPath = path.join(__dirname, '../node_modules/@dcloudio/vite-plugin-uni/dist/configResolved/config.js');
  if (fs.existsSync(configPath)) {
    let configContent = fs.readFileSync(configPath, 'utf8');
    if (configContent.includes('//   let ssr = (config as any).ssr')) {
      configContent = configContent.replace(
        `    //   let ssr = (config as any).ssr as SSROptions
    //   if (!ssr) {
    //     ssr = {}
    //   }
    //   if (ssr.external) {
    //     const index = ssr.external.findIndex((name) => name === 'vue')
    //     if (index !== -1) {
    //       ssr.external.splice(index, 1)
    //     }
    //   }
    //   if (!ssr.noExternal) {
    //     ssr.noExternal = ['vue']
    //   } else if (!ssr.noExternal.includes('vue')) {
    //     ssr.noExternal.push('vue')
    //   }`,
        `    let ssr = config.ssr
    if (!ssr) {
        ssr = config.ssr = {}
    }
    if (ssr.external) {
        const index = ssr.external.findIndex((name) => name === 'vue')
        if (index !== -1) {
            ssr.external.splice(index, 1)
        }
    }
    if (!ssr.noExternal) {
        ssr.noExternal = ['vue']
    } else if (!ssr.noExternal.includes('vue')) {
        ssr.noExternal.push('vue')
    }`
      );
      fs.writeFileSync(configPath, configContent);
      console.log('   ✅ SSR config 已修复\n');
    } else {
      console.log('   ✅ SSR config 已经修复\n');
    }
  } else {
    console.log('   ⚠️  找不到 config.js\n');
  }

  console.log('✨ 所有修复完成！\n');

} catch (error) {
  console.error('❌ 修复过程中出现错误：', error.message);
  process.exit(1);
}