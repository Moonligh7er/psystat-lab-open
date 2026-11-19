/*
 * Cookie / analytics consent banner (shared by the app shell and the static
 * site). Stores the choice in localStorage `psystat_cookie_consent`
 * ('accepted' | 'rejected') and fires a `psystat:consent` window event so the
 * app can start analytics only after the user accepts.
 *
 * No inline scripts or inline event handlers: the site CSP forbids them.
 */
(function () {
  var KEY = 'psystat_cookie_consent';

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return 'unavailable'; }
  }

  function decide(value) {
    try { window.localStorage.setItem(KEY, value); } catch (e) { /* ignore */ }
    var el = document.getElementById('cookie-consent');
    if (el && el.parentNode) el.parentNode.removeChild(el);
    try {
      window.dispatchEvent(new CustomEvent('psystat:consent', { detail: value }));
    } catch (e) { /* old browsers */ }
  }

  function button(id, label, value) {
    var b = document.createElement('button');
    b.type = 'button';
    b.id = id;
    b.textContent = label;
    b.addEventListener('click', function () { decide(value); });
    return b;
  }

  function show() {
    if (stored() || document.getElementById('cookie-consent')) return;

    if (!document.getElementById('cookie-consent-style')) {
      var s = document.createElement('style');
      s.id = 'cookie-consent-style';
      s.textContent =
        '#cookie-consent{position:fixed;bottom:0;left:0;right:0;background:#1E293B;color:#CBD5E1;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px;z-index:99999;font-family:system-ui,sans-serif;font-size:14px;border-top:1px solid #334155;flex-wrap:wrap}' +
        '#cookie-consent a{color:#818CF8;text-decoration:underline}' +
        '#cookie-consent .cookie-consent-buttons{display:flex;gap:8px}' +
        '#cookie-consent button{padding:8px 20px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:14px}' +
        '#cookie-accept{background:#6366F1;color:#fff}' +
        '#cookie-reject{background:transparent;color:#94A3B8;border:1px solid #475569 !important}';
      document.head.appendChild(s);
    }

    var d = document.createElement('div');
    d.id = 'cookie-consent';
    d.setAttribute('role', 'region');
    d.setAttribute('aria-label', 'Cookie consent');

    var text = document.createElement('span');
    text.appendChild(document.createTextNode('We use cookies for analytics. See our '));
    var link = document.createElement('a');
    link.href = '/privacy.html#cookies';
    link.textContent = 'Privacy Policy';
    text.appendChild(link);
    text.appendChild(document.createTextNode('.'));

    var buttons = document.createElement('div');
    buttons.className = 'cookie-consent-buttons';
    buttons.appendChild(button('cookie-reject', 'Reject', 'rejected'));
    buttons.appendChild(button('cookie-accept', 'Accept', 'accepted'));

    d.appendChild(text);
    d.appendChild(buttons);
    document.body.appendChild(d);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', show);
  } else {
    show();
  }
})();
