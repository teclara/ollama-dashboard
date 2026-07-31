(function(){
  const WORKSPACES = {
    overview: ['Overview', 'Live operating state and work requiring attention'],
    activity: ['Activity', 'Request volume, latency, clients, and journal evidence'],
    models: ['Models', 'Load, acquire, inspect, and remove local models'],
    performance: ['Performance', 'Controlled generation benchmarks across local models'],
    system: ['System', 'Compute, host, storage, service, and network telemetry'],
    settings: ['Settings', 'Read-only Ollama service configuration'],
  };

  function init(defaultWorkspace){
    const panels = Array.from(document.querySelectorAll('[data-workspace]'));
    const links = Array.from(document.querySelectorAll('[data-workspace-link]'));
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('nav-backdrop');
    const toggle = document.getElementById('menu-toggle');
    const content = document.getElementById('workspace-main');

    function closeNav(restoreFocus=false){
      if(!sidebar || !toggle || !backdrop) return;
      sidebar.dataset.open = 'false';
      backdrop.dataset.open = 'false';
      toggle.setAttribute('aria-expanded','false');
      if(restoreFocus) toggle.focus();
    }
    function openNav(){
      sidebar.dataset.open = 'true';
      backdrop.dataset.open = 'true';
      toggle.setAttribute('aria-expanded','true');
      const current = sidebar.querySelector('[aria-current="page"]') || sidebar.querySelector('a');
      if(current) current.focus();
    }
    function activate(){
      const requested = location.hash.slice(1);
      const key = panels.some(panel => panel.dataset.workspace === requested)
        ? requested : defaultWorkspace;
      for(const panel of panels) panel.hidden = panel.dataset.workspace !== key;
      for(const link of links){
        const url = new URL(link.href, location.href);
        const active = url.pathname === location.pathname && url.hash === '#'+key;
        if(active) link.setAttribute('aria-current','page');
        else link.removeAttribute('aria-current');
      }
      const meta = WORKSPACES[key] || WORKSPACES[defaultWorkspace];
      const title = document.getElementById('workspace-title');
      const subtitle = document.getElementById('workspace-subtitle');
      if(title) title.textContent = meta[0];
      if(subtitle) subtitle.textContent = meta[1];
      document.title = `${meta[0]} · Ollama Operations`;
      if(content) content.scrollTop = 0;
      closeNav(false);
    }
    if(toggle) toggle.addEventListener('click', () =>
      sidebar.dataset.open === 'true' ? closeNav(true) : openNav());
    if(backdrop) backdrop.addEventListener('click', () => closeNav(true));
    const skip = document.querySelector('.skip-link');
    if(skip) skip.addEventListener('click', event => {
      event.preventDefault();
      if(content) content.focus();
    });
    document.addEventListener('keydown', event => {
      if(event.key === 'Escape' && sidebar && sidebar.dataset.open === 'true') closeNav(true);
    });
    for(const link of links) link.addEventListener('click', () => closeNav(false));
    window.addEventListener('hashchange', activate);
    activate();
  }

  window.AppShell = {init};
})();
