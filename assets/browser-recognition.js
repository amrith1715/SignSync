(() => {
  const workerUrl = new URL('./browser-worker.js', document.currentScript.src);
  let worker, nextId = 1;
  const pending = new Map();
  const abortError = () => new DOMException('Recognition cancelled.', 'AbortError');
  function call(type, payload, signal, transfer = []) {
    if (signal?.aborted) return Promise.reject(abortError());
    if (!worker) {
      worker = new Worker(workerUrl);
      worker.onmessage = ({data}) => {
        const request = pending.get(data.id);
        if (!request) return;
        pending.delete(data.id); request.cleanup();
        if (data.error) request.reject(new Error(data.error));
        else request.resolve(data.result);
      };
      worker.onerror = () => {
        for (const request of pending.values()) {
          request.cleanup(); request.reject(new Error('The browser recognition engine could not start. Refresh and try a current Chrome, Edge, Firefox or Safari browser.'));
        }
        pending.clear(); worker.terminate(); worker = null;
      };
    }
    return new Promise((resolve, reject) => {
      const id = nextId++;
      const abort = () => { pending.delete(id); reject(abortError()); };
      const cleanup = () => signal?.removeEventListener('abort', abort);
      pending.set(id, {resolve, reject, cleanup});
      signal?.addEventListener('abort', abort, {once: true});
      try { worker.postMessage({id, type, payload}, transfer); }
      catch (error) { pending.delete(id); cleanup(); reject(error); }
    });
  }

  function waitFor(video, event, signal, action) {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        clearTimeout(timer); video.removeEventListener(event, done);
        video.removeEventListener('error', failed); signal?.removeEventListener('abort', abort);
      };
      const done = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error('This browser cannot decode the video. Try an MP4 with H.264 video or a WebM file.')); };
      const abort = () => { cleanup(); reject(abortError()); };
      const timer = setTimeout(failed, 20000);
      video.addEventListener(event, done, {once: true});
      video.addEventListener('error', failed, {once: true});
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) { abort(); return; }
      try { action?.(); } catch (error) { cleanup(); reject(error); }
    });
  }

  async function testVideo(file, signal) {
    const video = document.createElement('video');
    video.preload = 'auto'; video.muted = true; video.playsInline = true;
    const url = URL.createObjectURL(file);
    try {
      await waitFor(video, 'loadeddata', signal, () => { video.src = url; video.load(); });
      if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error('The video has no readable duration. Try a standard MP4 or WebM file.');
      if (video.videoWidth * video.videoHeight > 20_000_000) throw new Error('Use a video with frames smaller than 20 megapixels.');
      const info = await call('status', null, signal);
      const count = info.sequence_length;
      let start = 0, end = Math.max(0, video.duration - Math.min(0.05, video.duration / 100));
      for (let pass = 0; pass < 2; pass++) {
        await call('video-start', null, signal);
        const frames = [], times = [];
        for (let index = 0; index < count; index++) {
          if (signal?.aborted) throw abortError();
          const time = start + (end - start) * index / (count - 1);
          if (Math.abs(video.currentTime - time) > 0.00001) {
            await waitFor(video, 'seeked', signal, () => { video.currentTime = time; });
          }
          const bitmap = await createImageBitmap(video);
          if (signal?.aborted) { bitmap.close(); throw abortError(); }
          frames.push(await call('video-frame', bitmap, signal, [bitmap]));
          times.push(time);
        }
        const first = frames.findIndex(Boolean);
        const last = frames.findLastIndex(Boolean);
        if (first < 0) throw new Error('No hand was detected in this video. Use a clear recording with the whole hand visible.');
        // Ignore empty lead-in/ending frames, then resample the complete visible span.
        // Never fill missing landmarks or skip gaps inside a gesture.
        if (pass === 0 && (first > 0 || last < count - 1) && last > first) {
          start = times[first]; end = times[last]; continue;
        }
        if (frames.some(frame => !frame)) throw new Error('Hand tracking was lost during the gesture. Keep your whole hand visible and try again.');
        return await call('video-finish', frames, signal);
      }
      throw new Error('Could not find a complete visible gesture in the video.');
    } finally {
      video.pause(); video.removeAttribute('src'); video.load(); URL.revokeObjectURL(url);
      call('reset').catch(() => {});
    }
  }

  async function request(type, {body, signal} = {}) {
    if (type.startsWith('test-') && (!body?.size || body.size > 4 * 1024 * 1024)) {
      return Response.json({error: 'Choose a nonempty file no larger than 4 MB.'}, {status: 413});
    }
    try {
      const result = type === 'test-video' ? await testVideo(body, signal) : await call(type, body, signal);
      return Response.json(result);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      return Response.json({error: error.message || String(error)}, {status: 400});
    }
  }
  window.SignSyncBrowser = {request, reset: () => call('reset').catch(() => {})};
  window.addEventListener('pagehide', () => {
    worker?.terminate(); worker = null;
    for (const request of pending.values()) { request.cleanup(); request.reject(abortError()); }
    pending.clear();
  });
})();
