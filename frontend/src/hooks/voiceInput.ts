export type VoiceInput = { start(): Promise<void>; stop(): Promise<Blob | null>; abort(): void };
type Events = { text(value: string): void; ready(): void; error(message: string): void };

// SpeechRecognition is not yet included in TypeScript's standard browser types.
type Recognition = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
  onend: (() => void) | null; onerror: ((event: { error: string }) => void) | null;
  start(): void; stop(): void; abort(): void;
};
type SpeechWindow = Window & { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition };

export function hasLiveCaptions() {
  const browser = window as SpeechWindow;
  return Boolean(browser.SpeechRecognition || browser.webkitSpeechRecognition);
}

export function browserInput(events: Events): VoiceInput {
  const browser = window as SpeechWindow;
  const Constructor = browser.SpeechRecognition || browser.webkitSpeechRecognition;
  if (!Constructor) throw new Error('Live captions are unavailable. Choose recorded audio.');
  const recognition = new Constructor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-NG';
  let active = false, prefix = '', current = '';
  let timer: ReturnType<typeof setTimeout> | undefined;
  let finish: (() => void) | undefined;
  recognition.onresult = (event) => {
    current = Array.from(event.results).map(result => result[0].transcript).join(' ');
    events.text((prefix + ' ' + current).trim());
  };
  recognition.onend = () => {
    if (finish) { const resolve = finish; finish = undefined; resolve(); return; }
    if (active) {
      prefix = (prefix + ' ' + current).trim(); current = '';
      timer = setTimeout(() => {
        if (!active) return;
        try { recognition.start(); } catch { active = false; events.error('Listening stopped. Restart voice mode.'); }
      }, 400);
    }
  };
  recognition.onerror = ({ error }) => {
    if (error === 'no-speech' || error === 'aborted') return;
    active = false;
    events.error(error === 'not-allowed' ? 'Microphone access was denied. Allow it in your browser settings.'
      : 'Live captions could not connect. End voice mode and choose recorded audio.');
  };
  return {
    async start() { active = true; recognition.start(); },
    async stop() {
      active = false; clearTimeout(timer);
      await new Promise<void>(resolve => {
        const timeout = setTimeout(() => { finish = undefined; recognition.abort(); resolve(); }, 1800);
        finish = () => { clearTimeout(timeout); resolve(); };
        recognition.stop();
      });
      recognition.onresult = null;
      return null;
    },
    abort() {
      active = false; clearTimeout(timer); recognition.onresult = null;
      recognition.onend = null; recognition.onerror = null; recognition.abort(); finish?.();
    },
  };
}

export function recordedInput(events: Events): VoiceInput {
  let stream: MediaStream | undefined, recorder: MediaRecorder | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let cancelled = false;
  const chunks: BlobPart[] = [];
  let resolveStop: (blob: Blob) => void = () => {};
  const stopped = new Promise<Blob>(resolve => { resolveStop = resolve; });
  return {
    async start() {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
        throw new Error('Microphone recording needs Chrome or Edge on localhost or HTTPS.');
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true } });
      if (cancelled) { stream.getTracks().forEach(track => track.stop()); return; }
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        clearTimeout(timer); stream?.getTracks().forEach(track => track.stop());
        resolveStop(new Blob(chunks, { type: recorder?.mimeType || 'audio/webm' }));
        if (!cancelled) events.ready();
      };
      recorder.onerror = () => { events.error('Recording failed. End voice mode and try again.'); stream?.getTracks().forEach(track => track.stop()); };
      recorder.start();
      timer = setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 30000);
    },
    async stop() {
      cancelled = true;
      if (!recorder) throw new Error('The microphone is not ready yet.');
      if (recorder.state === 'recording') recorder.stop();
      return stopped;
    },
    abort() {
      cancelled = true; clearTimeout(timer);
      if (recorder?.state === 'recording') recorder.stop();
      stream?.getTracks().forEach(track => track.stop());
    },
  };
}
