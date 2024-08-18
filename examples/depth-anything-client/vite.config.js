import { defineConfig } from 'vite';
export default defineConfig({
  build: {
    target: 'esnext'
  },
    resolve: {
    alias: {
      'three/addons/loaders/GLTFLoader.js': 'three/examples/jsm/loaders/GLTFLoader.js'
    }}
});
