import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'esnext'
  },
   rollupOptions: {
      input: {
        main: './main.js',  // First entry point
      }
    }
});
