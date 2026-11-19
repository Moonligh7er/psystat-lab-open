/* Page script for encyclopedia.html (moved out of inline <script> so the CSP needs no 'unsafe-inline'). */

(function() {
  var searchBox = document.getElementById('searchBox');
  var noResults = document.getElementById('noResults');
  var entries = Array.prototype.slice.call(document.querySelectorAll('.entry[data-title]'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('.cat-section'));

  function normalize(s) { return (s || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' '); }

  searchBox.addEventListener('input', function() {
    var q = normalize(searchBox.value).trim();
    var anyVisible = false;

    entries.forEach(function(e) {
      var hay = normalize(e.getAttribute('data-title') + ' ' + e.getAttribute('data-content') + ' ' + e.textContent);
      var match = q === '' || hay.indexOf(q) !== -1;
      e.style.display = match ? '' : 'none';
      if (match) anyVisible = true;
    });

    // Hide section headers if no entries in that section are visible
    sections.forEach(function(sec) {
      if (sec.id === 'references') return;
      var visible = sec.querySelectorAll('.entry[data-title]:not([style*="display: none"])');
      sec.style.display = visible.length > 0 ? '' : 'none';
    });

    noResults.style.display = anyVisible ? 'none' : 'block';
  });
})();
