import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vitest/config';

// Built assets land inside the Python package so the wheel ships them and
// FastAPI mounts them at /dashboard; `base: './'` keeps every asset URL
// relative so the mount point does not matter.
export default defineConfig({
  base: './',
  plugins: [tailwindcss(), svelte()],
  build: {
    outDir: '../src/acceptrate/serve/static',
    emptyOutDir: true,
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
});
