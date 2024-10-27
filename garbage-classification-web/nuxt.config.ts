import wasmPack from "vite-plugin-wasm-pack";

export default defineNuxtConfig({
  compatibilityDate: "2024-04-03",
  modules: [
    "@vueuse/nuxt",
    "@nuxt/ui",
    "@morev/vue-transitions/nuxt",
    "nuxt-maplibre",
  ],
  devtools: { enabled: true },
  vite: {
    plugins: [wasmPack("./classification-model")],
    optimizeDeps: {
      exclude: ["classification-model"],
    },
  },
  nitro: {
    experimental: {
      wasm: true,
    },
  },
  app: {
    baseURL: "/TurboTrash/",
  },
});
