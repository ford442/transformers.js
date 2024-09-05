import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'esnext'
  },
    rollupOptions: {
    external: ['@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow']
  }
});
