/* Page script for educators.html (replaces inline onclick handlers; CSP has no 'unsafe-inline'). */
document.querySelectorAll('.faq-item > .faq-q').forEach(function (q) {
  q.setAttribute('role', 'button');
  q.setAttribute('tabindex', '0');
  function toggle() { q.parentElement.classList.toggle('open'); }
  q.addEventListener('click', toggle);
  q.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
  });
});
