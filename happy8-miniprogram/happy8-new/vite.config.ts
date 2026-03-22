import { defineConfig } from "vite";
import uni from "@dcloudio/vite-plugin-uni";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [uni()],
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: '@import "@/uni.scss";', // Globally import Sass variables and mixins
        logger: {
          warn: () => {} // Suppress Sass deprecation warnings from wot-design-uni
        }
      }
    }
  }
});
