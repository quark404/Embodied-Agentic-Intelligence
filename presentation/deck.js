(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const dots = document.getElementById('slideDots');
  const pageCount = document.getElementById('pageCount');
  const previous = document.getElementById('prevButton');
  const next = document.getElementById('nextButton');
  const print = document.getElementById('printButton');
  let current = 0;
  let touchStartX = null;

  const buttons = slides.map((slide, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `跳转到第 ${index + 1} 页：${slide.querySelector('h1,h2')?.textContent?.trim() || ''}`);
    button.addEventListener('click', () => show(index));
    dots.appendChild(button);
    return button;
  });

  function show(index, updateHash = true) {
    const bounded = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, position) => {
      const active = position === bounded;
      slide.classList.toggle('is-active', active);
      slide.setAttribute('aria-hidden', String(!active));
      slide.inert = !active;
      if (active) slide.scrollTop = 0;
      buttons[position].setAttribute('aria-current', String(active));
    });
    current = bounded;
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    pageCount.textContent = `${String(current + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
    document.title = `${slides[current].querySelector('h1,h2')?.textContent?.trim().replace(/\s+/g, ' ') || '研究汇报'} · Embodied Agentic Intelligence`;
    if (updateHash && window.location.hash !== `#slide-${current + 1}`) {
      try { history.replaceState(null, '', `#slide-${current + 1}`); }
      catch { window.location.hash = `slide-${current + 1}`; }
    }
  }

  previous.addEventListener('click', () => show(current - 1));
  next.addEventListener('click', () => show(current + 1));
  print.addEventListener('click', () => window.print());
  window.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'].includes(document.activeElement?.tagName)) return;
    if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(current + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(current - 1); }
    else if (event.key === 'Home') { event.preventDefault(); show(0); }
    else if (event.key === 'End') { event.preventDefault(); show(slides.length - 1); }
    else if (event.key.toLowerCase() === 'f' && document.fullscreenEnabled) {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    }
  });
  window.addEventListener('hashchange', () => {
    const match = /^#slide-(\d+)$/.exec(window.location.hash);
    if (match) show(Number(match[1]) - 1, false);
  });
  document.addEventListener('touchstart', event => {
    touchStartX = event.touches.length === 1 && !event.target.closest('.matrix-wrap')
      ? event.touches[0].clientX : null;
  }, {passive: true});
  document.addEventListener('touchend', event => {
    if (touchStartX === null || event.changedTouches.length !== 1) return;
    const delta = event.changedTouches[0].clientX - touchStartX;
    if (Math.abs(delta) > 75) show(current + (delta < 0 ? 1 : -1));
    touchStartX = null;
  }, {passive: true});
  const initial = /^#slide-(\d+)$/.exec(window.location.hash);
  show(initial ? Number(initial[1]) - 1 : 0, false);
})();
