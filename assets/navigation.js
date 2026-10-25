(() => {
  const views = [...document.querySelectorAll('[data-view]')];
  let current = 'recognition';
  let resume = false;
  function route(focus = true) {
    const target = location.hash.slice(1);
    const next = views.some(v => v.dataset.view === target) ? target : 'recognition';
    if (next !== 'recognition' && current === 'recognition') {
      resume = state.cameraActive || Boolean(state.projectStream);
      stopProjectCamera();
      setStatus('Recognition paused');
    }
    if (current === 'alphabet' && next !== 'alphabet') {
      document.querySelectorAll('#aslGrid video').forEach(video => video.pause());
    }
    current = next;
    views.forEach(v => { v.hidden = v.dataset.view !== current; });
    document.querySelectorAll('[data-view-link]').forEach(a => {
      if (a.dataset.viewLink === current) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    document.querySelector('.app-nav').classList.remove('menu-open');
    document.getElementById('menuToggle').setAttribute('aria-expanded', 'false');
    document.getElementById('testingCameraNotice').textContent = resume
      ? 'Live recognition is paused while you use another workspace. It resumes when you return to Recognition after testing.'
      : 'Choose a file, review its preview, then click Predict. Files are processed on your device.';
    const heading = document.querySelector(`[data-view="${current}"] h1`);
    document.title = `${heading.textContent} · SignSync`;
    if (focus) heading.focus({preventScroll:true});
    if (current === 'recognition' && resume && state.recognitionReady && !window.signSyncMediaTestBusy) {
      resume = false; initProjectCamera();
    }
  }
  window.SignSyncNavigation = { canRun: () => current === 'recognition', afterTest: () => {
    if (current === 'recognition' && resume) { resume = false; initProjectCamera(); }
  }};
  document.getElementById('menuToggle').addEventListener('click', () => {
    const open = document.querySelector('.app-nav').classList.toggle('menu-open');
    document.getElementById('menuToggle').setAttribute('aria-expanded', String(open));
  });
  document.addEventListener('keydown', e => { if(e.key === 'Escape') {
    document.querySelector('.app-nav').classList.remove('menu-open');
    document.getElementById('menuToggle').setAttribute('aria-expanded','false');
  }});
  window.addEventListener('hashchange', () => route());
  window.addEventListener('pagehide', () => stopProjectCamera());
  route(false);
})();
