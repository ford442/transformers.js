import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'esnext',
    external: ['@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow']
  }
});
