import { mount } from 'svelte';

import App from './App.svelte';
import './app.css';

const THEMES = new Set(['light', 'dark']);

/** `?theme=dark|light` pins the scheme (tokens.css honours [data-theme]); otherwise the OS decides. */
function applyThemeOverride(): void {
  const theme = new URLSearchParams(window.location.search).get('theme');
  if (theme !== null && THEMES.has(theme)) {
    document.documentElement.dataset.theme = theme;
  }
}

const target = document.getElementById('app');
if (target === null) {
  throw new Error('dashboard: #app mount point is missing from index.html');
}

applyThemeOverride();

export default mount(App, { target });
