(() => {
  const imageInput = document.getElementById('staticTestUpload');
  const videoInput = document.getElementById('dynamicTestUpload');
  const box = document.getElementById('mediaTestResult');
  const status = document.getElementById('mediaTestStatus');
  const filename = document.getElementById('mediaTestFilename');
  const letter = document.getElementById('mediaTestLetter');
  const confidence = document.getElementById('mediaTestConfidence');
  const queue = document.createElement('ol');
  queue.className = 'media-test-queue';
  queue.style.cssText = 'display:grid;gap:12px;padding:0;list-style:none';
  filename.after(queue);

  const predictButton = document.createElement('button');
  predictButton.type = 'button';
  predictButton.className = 'btn btn-primary';
  predictButton.textContent = 'Predict all';
  const predictActions = document.createElement('div');
  predictActions.className = 'text-controls';
  predictActions.style.marginTop = '18px';
  predictActions.append(predictButton);
  queue.after(predictActions);

  const entries = [];
  const MAX_FILE_BYTES = 4 * 1024 * 1024;
  let busy = false;

  function resetSummary() {
    letter.textContent = '—';
    confidence.textContent = '';
    box.dataset.error = 'false';
  }

  function updateCount() {
    const activeCount = entries.filter(entry => !entry.removed).length;
    filename.textContent = activeCount
      ? `${activeCount} file${activeCount === 1 ? '' : 's'} selected`
      : 'No files selected';
    predictButton.textContent = activeCount > 1 ? 'Predict all' : 'Predict';
    predictButton.disabled = busy || activeCount === 0;
  }

  function removeEntry(entry) {
    if (entry.removed) return;
    entry.removed = true;
    entry.controller?.abort();
    if (entry.previewUrl) URL.revokeObjectURL(entry.previewUrl);
    entry.row.remove();
    updateCount();
    if (!busy && entries.every(item => item.removed)) {
      status.textContent = 'Choose an image or video to begin.';
      resetSummary();
    }
  }

  function addFiles(input, kind) {
    if (busy) return;
    const files = Array.from(input.files || []);
    input.value = '';
    if (!files.length) return;

    const supported = kind === 'image'
      ? /\.(jpe?g|png|webp)$/i
      : /\.(mp4|mov|avi|mkv|webm|ogv)$/i;
    let added = 0;
    const skipped = [];
    for (const file of files) {
      if (!file.size || file.size > MAX_FILE_BYTES || !supported.test(file.name)) {
        const reason = file.size > MAX_FILE_BYTES ? 'larger than 4 MB' : 'an unsupported or empty file';
        skipped.push(`${file.name} (${reason})`);
        box.dataset.error = 'true';
        continue;
      }

      const row = document.createElement('li');
      row.className = 'surface media-test-queue-item';
      row.style.cssText = 'display:grid;grid-template-columns:minmax(90px,160px) minmax(0,1fr);gap:12px;align-items:center;padding:12px';
      const media = document.createElement(kind === 'image' ? 'img' : 'video');
      media.style.cssText = 'width:100%;max-height:130px;object-fit:contain;background:#f3f6f8;border-radius:5px';
      const previewUrl = URL.createObjectURL(file);
      media.src = previewUrl;
      if (kind === 'image') media.alt = `Selected image: ${file.name}`;
      else { media.controls = true; media.preload = 'metadata'; }

      const details = document.createElement('div');
      details.style.cssText = 'min-width:0;display:grid;gap:6px';
      const name = document.createElement('strong');
      name.textContent = file.name;
      name.style.cssText = 'overflow-wrap:anywhere;font-size:13px';
      const size = document.createElement('span');
      size.textContent = `${kind === 'image' ? 'Image' : 'Video'} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
      size.className = 'field-help';
      const result = document.createElement('span');
      result.className = 'field-help';
      result.setAttribute('role', 'status');
      result.textContent = 'Ready';
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'btn btn-secondary';
      remove.textContent = 'Remove';
      remove.disabled = busy;
      remove.addEventListener('click', () => removeEntry(entry));
      details.append(name, size, result, remove);
      row.append(media, details);
      queue.append(row);

      const entry = {file, kind, row, result, remove, previewUrl, controller: null, removed: false};
      entries.push(entry);
      added++;
    }

    if (added) {
      resetSummary();
      const count = entries.filter(entry => !entry.removed).length;
      status.textContent = `${count} file(s) ready. Click Predict to process them one at a time.${skipped.length ? ` Skipped: ${skipped.join(', ')}.` : ''}`;
      if (skipped.length) box.dataset.error = 'true';
    } else if (skipped.length) {
      status.textContent = `No files added. Skipped: ${skipped.join(', ')}.`;
    }
    updateCount();
  }

  async function predict(entry) {
    if (entry.removed) return null;
    entry.result.textContent = 'Processing…';
    entry.result.style.color = '';
    entry.remove.disabled = true;
    entry.controller = new AbortController();
    const timeout = setTimeout(() => entry.controller?.abort(), 120000);
    try {
      let response;
      for (let attempt = 0; attempt < 4; attempt++) {
        response = await signSyncRequest(`test-${entry.kind}`, {
          method: 'POST',
          body: entry.file,
          headers: {'Content-Type': entry.file.type || 'application/octet-stream'},
          signal: entry.controller.signal
        });
        if (response.status !== 429 || attempt === 3) break;
        await response.text();
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      const result = await response.json();
      if (entry.removed) return null;
      if (!response.ok) throw new Error(result.error || 'Unable to process this file.');
      entry.result.textContent = result.predicted
        ? `Prediction: ${result.predicted}${Number.isFinite(result.confidence) ? ` · ${(result.confidence * 100).toFixed(1)}% confidence` : ''}`
        : `No confident ${entry.kind === 'video' ? 'J or Z movement' : 'letter'} detected.`;
      return {ok: true, result};
    } catch (error) {
      if (entry.removed) return null;
      entry.result.textContent = error.name === 'AbortError' ? 'Timed out. Try a shorter or smaller file.' : error.message;
      entry.result.style.color = 'var(--danger)';
      return {ok: false, error};
    } finally {
      clearTimeout(timeout);
      entry.controller = null;
      entry.remove.disabled = false;
    }
  }

  async function predictAll() {
    const pending = entries.filter(entry => !entry.removed);
    if (!pending.length || busy) {
      if (!pending.length) {
        box.dataset.error = 'true';
        status.textContent = 'No files selected. Choose one or more images or videos first.';
      }
      return;
    }

    busy = true;
    window.signSyncMediaTestBusy = true;
    predictButton.disabled = true;
    imageInput.disabled = videoInput.disabled = document.getElementById('liveToggleBtn').disabled = true;
    pending.forEach(entry => { entry.remove.disabled = true; });
    stopProjectCamera();
    box.setAttribute('aria-busy', 'true');
    status.textContent = `Processing ${pending.length} file${pending.length === 1 ? '' : 's'} one at a time…`;
    let successes = 0;
    let failures = 0;
    let lastPrediction = null;
    try {
      for (const entry of pending) {
        if (entry.removed) continue;
        const outcome = await predict(entry);
        if (!outcome) continue;
        if (outcome.ok) {
          successes++;
          lastPrediction = outcome.result;
        } else failures++;
      }
      const completed = successes + failures;
      status.textContent = `Finished ${completed} of ${pending.length}: ${successes} succeeded${failures ? `, ${failures} failed` : ''}.`;
      if (lastPrediction?.predicted) {
        letter.textContent = lastPrediction.predicted;
        if (Number.isFinite(lastPrediction.confidence)) confidence.textContent = `Last prediction confidence: ${(lastPrediction.confidence * 100).toFixed(1)}%`;
      }
      box.dataset.error = failures ? 'true' : 'false';
    } finally {
      busy = false;
      window.signSyncMediaTestBusy = false;
      box.removeAttribute('aria-busy');
      imageInput.disabled = videoInput.disabled = document.getElementById('liveToggleBtn').disabled = false;
      entries.forEach(entry => { if (!entry.removed) entry.remove.disabled = false; });
      setStatus('Recognition paused');
      window.SignSyncNavigation?.afterTest();
      updateCount();
    }
  }

  imageInput.addEventListener('change', () => addFiles(imageInput, 'image'));
  videoInput.addEventListener('change', () => addFiles(videoInput, 'video'));
  predictButton.addEventListener('click', predictAll);
  window.addEventListener('pagehide', () => entries.forEach(entry => {
    if (entry.previewUrl) URL.revokeObjectURL(entry.previewUrl);
  }));
  updateCount();
})();
