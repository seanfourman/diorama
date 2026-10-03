import { defineConfig, type Plugin } from 'vite';
// @ts-expect-error: a plain JavaScript module, shared with scripts/check.mjs
import { handleSceneApi } from './scripts/lib/sceneApi.mjs';

// The dev server also runs the local scene API (M5): uploads, COLMAP, and saving
// trained scenes into data/. See scripts/lib/sceneApi.mjs.
const sceneApi: Plugin = {
  name: 'diorama-scene-api',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      handleSceneApi(req, res).then((handled: boolean) => handled || next(), next);
    });
  },
};

export default defineConfig({ plugins: [sceneApi] });
