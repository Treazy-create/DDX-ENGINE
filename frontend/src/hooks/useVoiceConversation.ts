import { useEffect, useRef, useState } from 'react';
import { browserInput, hasLiveCaptions, recordedInput, type VoiceInput } from './voiceInput';

type Phase = 'off' | 'starting' | 'listening' | 'ready' | 'thinking' | 'speaking' | 'paused';
type Options = {
  transcript(text: string): void;
  error(message: string): void;
  transcribe(blob: Blob): Promise<string>;
};

export function useVoiceConversation(options: Options) {
  const [enabled, setEnabled] = useState(false);
  const [phase, setPhase] = useState<Phase>('off');
  const [engine, setEngine] = useState<'browser' | 'recorded'>(() => hasLiveCaptions() ? 'browser' : 'recorded');
  const [muted, setMuted] = useState(false);
  const active = useRef(false), mute = useRef(false), generation = useRef(0);
  const input = useRef<VoiceInput | null>(null), text = useRef('');
  const callbacks = useRef(options), engineRef = useRef(engine), phaseRef = useRef<Phase>('off');
  const restart = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const utterance = useRef<SpeechSynthesisUtterance | null>(null);
  callbacks.current = options; engineRef.current = engine;
  function transition(next: Phase) { phaseRef.current = next; setPhase(next); }
  function release() { clearTimeout(restart.current); input.current?.abort(); input.current = null; }
  function stopSpeech() {
    if (utterance.current) { utterance.current.onend = null; utterance.current.onerror = null; }
    utterance.current = null;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  }
  function disable() {
    active.current = false; generation.current++; release(); stopSpeech();
    setEnabled(false); transition('off');
  }
  async function listen() {
    if (!active.current) return;
    release(); text.current = ''; callbacks.current.transcript(''); transition('starting');
    const token = ++generation.current;
    const current = () => token === generation.current && active.current;
    const events = {
      text(value: string) { if (current()) { text.current = value; callbacks.current.transcript(value); } },
      ready() { if (current() && phaseRef.current === 'listening') transition('ready'); },
      error(message: string) { if (current()) { release(); transition('paused'); callbacks.current.error(message); } },
    };
    try {
      input.current = engineRef.current === 'browser' ? browserInput(events) : recordedInput(events);
      await input.current.start();
      if (current()) transition('listening');
    } catch (error) {
      if (current()) { release(); transition('paused'); callbacks.current.error(error instanceof Error ? error.message : 'Microphone failed.'); }
    }
  }
  function enable() { active.current = true; setEnabled(true); void listen(); }
  async function capture() {
    if (!active.current) return text.current;
    if (phaseRef.current === 'paused' && text.current.trim()) { transition('thinking'); return text.current; }
    if (!input.current) throw new Error('Restart listening before sending.');
    transition('thinking');
    const token = generation.current;
    const blob = await input.current.stop(); input.current = null;
    if (!active.current || token !== generation.current) throw new Error('Voice mode ended. Nothing was sent.');
    if (blob) {
      if (!blob.size || blob.size > 4 * 1024 * 1024) throw new Error('Recording is empty or too large. Record up to 30 seconds.');
      text.current = await callbacks.current.transcribe(blob);
    }
    if (!active.current || token !== generation.current) throw new Error('Voice mode ended. Nothing was sent.');
    callbacks.current.transcript(text.current);
    if (!text.current.trim()) throw new Error('No speech was captured. Restart listening or switch to text.');
    return text.current.trim();
  }
  function speak(message: string, resume = true) {
    release(); stopSpeech();
    const token = ++generation.current;
    const complete = () => {
      if (token !== generation.current) return;
      utterance.current = null;
      if (active.current && resume) {
        transition('starting'); restart.current = setTimeout(() => void listen(), 450);
      } else if (active.current) { disable(); } else transition('off');
    };
    if (mute.current && active.current) { complete(); return; }
    if (!('speechSynthesis' in window)) {
      transition(active.current ? 'paused' : 'off');
      callbacks.current.error('Automatic speech is unavailable in this browser. The reply is visible in the chat.'); return;
    }
    const speech = new SpeechSynthesisUtterance(message);
    utterance.current = speech; speech.lang = 'en-GB'; speech.rate = .98;
    const voices = window.speechSynthesis.getVoices();
    speech.voice = voices.find(voice => voice.lang === 'en-GB') || voices.find(voice => voice.lang.startsWith('en')) || null;
    speech.onend = complete;
    speech.onerror = () => {
      if (token !== generation.current) return;
      transition(active.current ? 'paused' : 'off');
      callbacks.current.error('Your browser could not play the reply. Press Read aloud to retry or Restart listening.');
    };
    transition('speaking'); window.speechSynthesis.speak(speech);
  }
  function toggleMute() {
    mute.current = !mute.current; setMuted(mute.current);
    if (mute.current && phaseRef.current === 'speaking') { stopSpeech(); if (active.current) void listen(); else transition('off'); }
  }
  function pause() { release(); stopSpeech(); if (active.current) transition('paused'); }
  useEffect(() => {
    const hidden = () => { if (document.hidden) disable(); };
    document.addEventListener('visibilitychange', hidden);
    return () => { document.removeEventListener('visibilitychange', hidden); active.current = false; generation.current++; release(); stopSpeech(); };
  }, []);
  return { enabled, phase, engine, setEngine, muted, toggleMute, enable, disable, capture, speak, pause,
    restart: listen, liveAvailable: hasLiveCaptions(), speaking: phase === 'speaking',
    canSend: ['listening', 'ready', 'paused'].includes(phase), isEnabled: () => active.current };
}
