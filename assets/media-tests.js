(() => {
  const image = document.getElementById('staticTestUpload');
  const video = document.getElementById('dynamicTestUpload');
  const box = document.getElementById('mediaTestResult');
  const status = document.getElementById('mediaTestStatus');
  const predictButton = document.createElement('button');
  predictButton.type = 'button';
  predictButton.className = 'btn btn-primary';
  predictButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7z"/></svg><span>Predict</span>';
  const predictActions = document.createElement('div');
  predictActions.className = 'text-controls';
  predictActions.style.marginTop = '18px';
  predictActions.append(predictButton);
  let predictSelected = null;
  predictButton.addEventListener('click', () => {
    if (window.signSyncMediaTestBusy) return;
    if (!predictSelected) {
      box.dataset.error = 'true';
      status.textContent = 'No file selected. Please insert an image or video before clicking Predict.';
      status.scrollIntoView({block:'nearest'});
      return;
    }
    predictSelected();
  });
  const preview = document.createElement('div');
  preview.className = 'selected-file-preview';
  preview.hidden = true;
  document.getElementById('mediaTestFilename').after(preview);
  preview.after(predictActions);
  let previewUrl = null;
  function clearPreview() {
    preview.querySelector('video')?.pause();
    preview.replaceChildren();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.hidden = true;
  }
  window.addEventListener('pagehide', clearPreview);
  function selectFile(input, kind) {
    const file = input.files[0];
    if (!file || window.signSyncMediaTestBusy) return;
    predictSelected = null;
    clearPreview();
    document.getElementById('mediaTestLetter').textContent = '—';
    document.getElementById('mediaTestConfidence').textContent = '';
    document.getElementById('mediaTestFilename').textContent = file.name;
    const supported = kind === 'image' ? /\.(jpe?g|png|webp)$/i : /\.(mp4|mov|avi|mkv|webm|ogv)$/i;
    const limit = kind === 'image' ? 10 : 50;
    box.dataset.error = 'false';
    if (!supported.test(file.name) || !file.size || file.size > limit * 1048576) {
      box.dataset.error = 'true'; status.textContent = `Choose a supported, non-empty ${kind} up to ${limit} MB.`; input.value = ''; return;
    }
    // Retain the File in this selection; uploads only begin on Predict.
    image.value = video.value = '';
    previewUrl = URL.createObjectURL(file);
    const media = document.createElement(kind === 'image' ? 'img' : 'video');
    if (kind === 'image') media.alt = `Selected image: ${file.name}`;
    else { media.controls = true; media.preload = 'metadata'; }
    media.src = previewUrl;
    const caption = document.createElement('p');
    caption.textContent = `Selected ${kind}: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn btn-secondary';
    removeButton.textContent = 'Remove file';
    removeButton.setAttribute('aria-label', `Remove selected ${kind}`);
    const actions = document.createElement('div');
    actions.className = 'text-controls';
    actions.append(removeButton);
    preview.append(media, caption, actions);
    preview.hidden = false;
    status.textContent = 'File ready. Click Predict to start.';
    let controller = null;
    let removed = false;
    removeButton.addEventListener('click', () => {
      removed = true;
      predictSelected = null;
      controller?.abort();
      clearPreview();
      image.value = video.value = '';
      document.getElementById('mediaTestFilename').textContent = 'No file selected';
      document.getElementById('mediaTestLetter').textContent = '—';
      document.getElementById('mediaTestConfidence').textContent = '';
      box.dataset.error = 'false';
      box.removeAttribute('aria-busy');
      status.textContent = 'File removed. Choose an image or video to begin.';
      if (!window.signSyncMediaTestBusy) input.focus();
    });
    predictSelected = async () => {
    if (removed || window.signSyncMediaTestBusy) return;
    window.signSyncMediaTestBusy = true;
    predictButton.disabled = true;
    image.disabled = video.disabled = document.getElementById('liveToggleBtn').disabled = true;
    document.getElementById('mediaTestLetter').textContent = '—';
    document.getElementById('mediaTestConfidence').textContent = '';
    box.dataset.error = 'false';
    stopProjectCamera();
    box.setAttribute('aria-busy','true');
    status.textContent = `Processing ${kind}…`;
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(),120000);
    try {
      let response;
      for (let attempt=0; attempt<4; attempt++) {
        response = await fetch(`./api/test-${kind}?${new URLSearchParams({name:file.name})}`, {method:'POST',body:file,headers:{'Content-Type':file.type || 'application/octet-stream'},signal:controller.signal});
        if (response.status !== 429 || attempt === 3) break;
        await response.text(); await new Promise(resolve => setTimeout(resolve,500));
      }
      const result = await response.json();
      if (removed) return;
      if (!response.ok) throw new Error(result.error || 'Unable to process this file.');
      document.getElementById('mediaTestLetter').textContent = result.predicted || '—';
      if (result.predicted && Number.isFinite(result.confidence)) document.getElementById('mediaTestConfidence').textContent = `Model confidence: ${(result.confidence*100).toFixed(1)}%`;
      status.textContent = result.predicted ? `Prediction complete: ${result.predicted}.` : `No confident ${kind === 'video' ? 'J or Z movement' : 'letter'} detected. Try a clearer ${kind}.`;
    } catch(error) {
      if (removed) return;
      box.dataset.error = 'true'; status.textContent = error.name === 'AbortError' ? 'Processing timed out. Try a smaller file.' : error.message;
    } finally {
      clearTimeout(timeout); box.removeAttribute('aria-busy'); window.signSyncMediaTestBusy = false;
      image.disabled = video.disabled = document.getElementById('liveToggleBtn').disabled = false;
      predictButton.disabled = false;
      input.value = ''; setStatus('Recognition paused'); window.SignSyncNavigation?.afterTest();
      if (removed) input.focus();
    }
    };
  }
  image.addEventListener('change', () => selectFile(image,'image'));
  video.addEventListener('change', () => selectFile(video,'video'));
})();
