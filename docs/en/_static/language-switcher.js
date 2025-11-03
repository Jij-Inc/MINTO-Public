(function(){
  function findSidebar(){
    return document.querySelector('.bd-sidebar-primary') || document.querySelector('.sidebar') || document.querySelector('aside');
  }
  function currentLang(){
    // Infer from path: /docs/en/... or /docs/ja/...
    var path = location.pathname;
    if (path.includes('/docs/ja/')) return 'ja';
    return 'en';
  }
  function counterpartHref(){
    // Swap /docs/en/ <-> /docs/ja/ in the current URL
    var href = location.href;
    if (href.includes('/docs/en/')) return href.replace('/docs/en/', '/docs/ja/');
    if (href.includes('/docs/ja/')) return href.replace('/docs/ja/', '/docs/en/');
    // Fallback: go to top of the other language
    if (currentLang()==='en') return location.origin + '/docs/ja/';
    return location.origin + '/docs/en/';
  }
  function render(){
    var sb = findSidebar();
    if(!sb) return;
    if (sb.querySelector('#minto-lang-switch')) return; // avoid duplicates
    var btn = document.createElement('button');
    btn.id = 'minto-lang-switch';
    btn.type = 'button';
    var lang = currentLang();
    btn.textContent = (lang==='en') ? '日本語へ' : 'English';
    btn.style.margin = '8px 12px';
    btn.style.padding = '6px 10px';
    btn.style.borderRadius = '6px';
    btn.style.border = '1px solid var(--pst-color-border,#ccc)';
    btn.style.cursor = 'pointer';
    btn.addEventListener('click', function(){
      window.location.href = counterpartHref();
    });
    // Insert at top of sidebar
    sb.insertBefore(btn, sb.firstChild);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
