/*
 * App shell bootstrap (loaded by psystat-app/public/index.html, the SPA
 * template). Registers the service worker. Kept as an external file because
 * the CSP does not allow inline scripts.
 */
(function () {
  if (!('serviceWorker' in navigator)) return;
  window.addEventListener('load', function () {
    navigator.serviceWorker.register('/sw.js').catch(function () { /* offline support is optional */ });
  });
})();
