import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'esnext'
  },
   rollupOptions: {
      input: {
        main: './loader.js',  // First entry point
      }
    }
});
