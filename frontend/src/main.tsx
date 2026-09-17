import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUp,
  CalendarDays,
  Check,
  CircleHelp,
  LoaderCircle,
  Menu,
  MessageCircle,
  Mic,
  Plus,
  ShieldCheck,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import "./styles.css";
import { useVoiceConversation } from './hooks/useVoiceConversation';
import { AppointmentBooking } from './components/AppointmentBooking';

const API = "http://127.0.0.1:8000";

type Reply = {
  session_id: string;
  reply: string;
  status: string;
  offer_booking?: boolean;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  time: string;
  status?: string;
  offerBooking?: boolean;
};

class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function api<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 110000);

  try {
    const response = await fetch(API + path, {
      ...options,
      signal: controller.signal,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new ApiError(
        typeof data.detail === "string"
          ? data.detail
          : "Please check the details and try again.",
        response.status
      );
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;

    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(
        "The reply is taking longer than expected. Your message is still here; you can retry."
      );
    }

    throw new Error(
      "Cannot reach the local service. Check that the backend terminal is still running."
    );
  } finally {
    clearTimeout(timeout);
  }
}

function post(body: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

function timestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function App() {
  const [session, setSession] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [reviewTranscript, setReviewTranscript] = useState(false);
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [page, setPage] = useState<"chat" | "appointments">("chat");
  const [expired, setExpired] = useState(false);

  const locked = useRef(false);
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const about = useRef<HTMLDialogElement>(null);

  const pending = useRef<{
    text: string;
    request_id: string;
  } | null>(null);

  const lastReply = messages
    .filter((message) => message.role === "assistant")
    .at(-1);

  const finished = Boolean(
    lastReply &&
      lastReply.status === "urgent"
  );

  function setWorking(value: boolean) {
    locked.current = value;
    setBusy(value);
  }

  const voice = useVoiceConversation({
    transcript: (text) => { setDraft(text); if (!text) setReviewTranscript(false); },
    error: setError,
    transcribe: async (blob) => {
      if (!session) throw new Error('Start a conversation first.');
      const form = new FormData();
      const extension = blob.type.includes('mp4') ? 'mp4' : 'webm';
      form.append('audio', blob, 'recording.' + extension);
      const result = await api<{ text: string }>('/sessions/' + session + '/transcribe', { method: 'POST', body: form });
      return result.text;
    },
  });
  const recording = voice.phase === 'listening';
  const speaking = voice.speaking;
  function openAppointments() {
    voice.disable(); setPage('appointments'); setMenuOpen(false);
  }

  useEffect(() => {
    bottom.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
      block: "end",
    });
  }, [messages, busy]);

  function handleError(error: unknown) {
    setError(
      error instanceof Error ? error.message : "Something went wrong."
    );

    if (error instanceof ApiError && error.status === 404) {
      voice.disable();
      setExpired(true);
      setNotice("This session has expired. Start a new conversation.");
    }
  }

  function addReply(reply: Reply) {
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: reply.reply,
        time: timestamp(),
        status: reply.status,
        offerBooking: reply.offer_booking,
      },
    ]);
    if (voice.isEnabled()) voice.speak(reply.reply, reply.status !== 'urgent');
  }

  async function startConversation(event: React.FormEvent) {
    event.preventDefault();
    if (locked.current) return;

    setWorking(true);
    setError("");

    try {
      const reply = await api<Reply>(
        "/sessions",
        post({ age: Number(age), sex })
      );

      setSession(reply.session_id);
      setExpired(false);
      setNotice("");
      addReply(reply);
      setTimeout(() => input.current?.focus(), 0);
    } catch (error) {
      handleError(error);
    } finally {
      setWorking(false);
    }
  }

  async function newConversation() {
    if (locked.current) return;

    if (
      messages.length &&
      !window.confirm("Start a new conversation? This conversation will be cleared.")
    ) {
      return;
    }

    setWorking(true);

    voice.disable();
    setPage("chat");

    if (session) {
      try {
        await api("/sessions/" + session, { method: "DELETE" });
      } catch {
        // A stopped server should not prevent clearing the local screen.
      }
    }

    setSession(null);
    setMessages([]);
    setDraft("");
    setError("");
    setNotice("");
    setExpired(false);
    setMenuOpen(false);
    pending.current = null;
    setWorking(false);
  }

  async function sendMessage() {
    if (!session || locked.current || finished || expired) return;
    setWorking(true); setError(''); setNotice('');
    try {
      const text = voice.enabled && !reviewTranscript ? await voice.capture() : draft.trim();
      if (voice.enabled && voice.engine === 'recorded' && !reviewTranscript) {
        setDraft(text); setReviewTranscript(true); voice.pause();
        setNotice('Check or correct your transcript, then press Send to submit it.');
        return;
      }
      if (!text) throw new Error('Enter or speak a message first.');
      if (text.length > 1800) throw new Error('Please keep each message under 1,800 characters. End voice mode to edit the transcript.');
      if (!pending.current || pending.current.text !== text) pending.current = { text, request_id: crypto.randomUUID() };
      const reply = await api<Reply>('/sessions/' + session + '/messages', post(pending.current));
      setMessages(current => [...current, { id: crypto.randomUUID(), role: 'user', text, time: timestamp() }]);
      setDraft(''); setReviewTranscript(false); pending.current = null; addReply(reply);
    } catch (error) {
      voice.pause(); handleError(error);
    } finally { setWorking(false); }
  }

  function readAloud() {
    if (!lastReply) return;
    if (speaking) { voice.pause(); return; }
    voice.speak(lastReply.text, !finished);
  }

  function toggleVoice() {
    if (voice.enabled) { voice.disable(); return; }
    if (draft.trim()) { setError('Send or clear your typed draft before starting voice mode.'); return; }
    setError(''); voice.enable();
  }

  return (
    <div className="app">
      {menuOpen && (
        <button
          className="scrim"
          aria-label="Close navigation"
          onClick={() => setMenuOpen(false)}
        />
      )}

      <aside className={"sidebar " + (menuOpen ? "open" : "")}>
        <div className="brand">
          <span className="logo"><img src="/logo.jpg" alt="" /></span>
          <div>DDx Engine<small>A little more clarity.</small></div>
        </div>

        <button
          className="new-chat"
          onClick={newConversation}
          disabled={busy}
        >
          <Plus size={19} /> New conversation
        </button>

        <button
          className="current-chat"
          aria-current={page === "chat" ? "page" : undefined}
          onClick={() => {
            setPage("chat"); setMenuOpen(false);
            input.current?.focus();
          }}
        >
          <MessageCircle size={19} /> Current conversation
        </button>

        <button className="about-link booking-nav" disabled={busy || finished} onClick={openAppointments}
          aria-current={page === 'appointments' ? 'page' : undefined}>
          <CalendarDays size={19} /> Demo appointments
        </button>
        <div className="sidebar-bottom">
          <div className="quiet-note">
            <ShieldCheck size={20} />
            <p>Your space to talk.<span>One question at a time.</span></p>
          </div>

          <button className="about-link" onClick={() => about.current?.showModal()}>
            <CircleHelp size={18} /> About this service
          </button>
        </div>
      </aside>

      <main>
        <header>
          <button
            className="icon-button mobile-menu"
            aria-label="Open navigation"
            onClick={() => setMenuOpen(true)}
          >
            <Menu />
          </button>

          <div className="heading">
            <h1>{page === "chat" ? "Let’s talk about how you feel." : "Your next conversation."}</h1>
            <p>Describe what’s bothering you. We’ll take it from there.</p>
          </div>

          <button
            className="read-button"
            onClick={readAloud}
            disabled={!lastReply || recording || page !== "chat"}
            aria-label={speaking ? "Stop reading" : "Read the last reply aloud"}
          >
            {speaking ? <VolumeX size={19} /> : <Volume2 size={19} />}
            <span>{speaking ? "Stop reading" : "Read aloud"}</span>
          </button>
        </header>

        {page === 'appointments' ? <AppointmentBooking onBack={() => setPage('chat')} /> : <>
        <section className="conversation">
          <div className="wallpaper" aria-hidden="true" />

          <div className="conversation-inner">
            {!session ? (
              <div className="welcome">
                <span className="welcome-logo">
                  <img src="/logo.jpg" alt="DDx medical emblem" />
                </span>

                <h2>A good place to start<br />is how you’re feeling.</h2>
                <p>
                  You don’t need the right medical words.
                  Tell us in your own way, by typing or speaking.
                </p>

                <form className="start-form" onSubmit={startConversation}>
                  <h3>Before we begin</h3>
                  <p>A couple of details for more relevant questions.</p>

                  <div className="fields">
                    <label>
                      Your age
                      <input
                        type="number"
                        min="0"
                        max="120"
                        required
                        placeholder="Age"
                        value={age}
                        onChange={(event) => setAge(event.target.value)}
                      />
                    </label>

                    <label>
                      Recorded sex
                      <select
                        required
                        value={sex}
                        onChange={(event) => setSex(event.target.value)}
                      >
                        <option value="" disabled>Select</option>
                        <option value="F">Female</option>
                        <option value="M">Male</option>
                      </select>
                    </label>
                  </div>

                  <button className="primary start-button" disabled={busy}>
                    {busy
                      ? <LoaderCircle size={18} className="spin" />
                      : <MessageCircle size={18} />}
                    Start conversation
                  </button>

                  <small>
                    Messages and recordings you submit are processed by OpenAI.
                    This session is not saved as chat history.
                  </small>
                </form>
              </div>
            ) : (
              <>
                <div className="today"><span>Today</span></div>

                <div className="messages" role="log" aria-live="polite">
                  {messages.map((message) => (
                    <article
                      className={"message " + message.role}
                      key={message.id}
                    >
                      <div className={
                        "bubble " + (message.status === "urgent" ? "urgent" : "")
                      }>
                        {message.role === "assistant" && (
                          <span className="author">DDx</span>
                        )}

                        <p>{message.text}</p>

                        {message.role === "assistant" &&
                          (message.offerBooking || ["assessment", "uncertain"].includes(message.status || "")) && (
                            <div className="handoff">
                              <button className="primary" onClick={openAppointments} disabled={busy || finished}>
                                <CalendarDays size={18} /> Book a demo appointment
                              </button>
                              <small>Fictional practitioners · No real booking</small>
                            </div>
                          )}
                      </div>

                      <div className="message-time">
                        {message.time}
                        {message.role === "user" && <Check size={12} />}
                      </div>
                    </article>
                  ))}
                </div>

                {busy && (
                  <div className="thinking" role="status">
                    <LoaderCircle size={16} className="spin" />
                    {notice || "Thinking through what you shared…"}
                  </div>
                )}

                <div ref={bottom} />
              </>
            )}
          </div>
        </section>

        <footer>
          {error && (
            <div className="error" role="alert">
              <span>{error}</span>
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={17} />
              </button>
            </div>
          )}

          {session && !finished && !expired && <div className="voice-toolbar">
            <button type="button" className="back-link" onClick={toggleVoice} disabled={busy && !voice.enabled}>
              <Mic size={17} /> {voice.enabled ? 'End voice mode' : 'Start voice mode'}
            </button>
            {!voice.enabled && <label>Voice input
              <select value={voice.engine} onChange={event => voice.setEngine(event.target.value as 'browser' | 'recorded')} disabled={busy}>
                {voice.liveAvailable && <option value="browser">Live captions · browser</option>}
                <option value="recorded">Recorded audio · OpenAI</option>
              </select>
            </label>}
            {voice.enabled && <>
              <span role="status">{{off:'Off', starting:'Starting microphone…', listening:'Listening — press Send when finished', ready:'Recording ready — press Send', thinking:'Preparing reply…', speaking:'Speaking…', paused:'Voice paused'}[voice.phase]}</span>
              <button type="button" className="back-link" onClick={voice.toggleMute}>{voice.muted ? <VolumeX size={17} /> : <Volume2 size={17} />}{voice.muted ? 'Unmute replies' : 'Mute replies'}</button>
              {voice.phase === 'paused' && !busy && <button type="button" className="back-link" onClick={() => void voice.restart()}>Restart listening</button>}
            </>}
            <small>{voice.engine === 'browser' ? 'Live captions may send audio to your browser’s speech service.' : 'Press Send to transcribe, review the text, then Send again to submit. Recordings are limited to 30 seconds.'} Replies use a synthetic browser voice.</small>
          </div>}
          {notice && !busy && !recording && (
            <p className="notice" role="status">{notice}</p>
          )}

          {finished || expired ? (
            <div className="conversation-end">
              <span>
                {lastReply?.status === "urgent"
                  ? "Please follow the urgent guidance above."
                  : "Would you like to discuss another concern?"}
              </span>

              <button onClick={newConversation} disabled={busy}>
                <Plus size={16} /> New conversation
              </button>
            </div>
          ) : (
            <form
              className={"composer " + (recording ? "is-recording" : "")}
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage();
              }}
            >
              <label className="sr-only" htmlFor="message">Your message</label>

              <textarea
                id="message"
                ref={input}
                rows={2}
                maxLength={1800}
                value={draft}
                disabled={!session || busy}
                readOnly={voice.enabled && !reviewTranscript}
                placeholder={
                  !session
                    ? "Start a conversation above to begin…"
                    : voice.enabled
                    ? "Voice mode — your transcript appears here…"
                    : "Describe what you’re feeling…"
                }
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
              />

              <div className="composer-buttons">
                <button
                  type="button"
                  className={"mic-button " + (recording ? "recording" : "")}
                  aria-label={voice.enabled ? "End voice mode" : "Start voice mode"}
                  onClick={toggleVoice}
                  disabled={!session || (busy && !voice.enabled)}
                >
                  <Mic size={21} />
                </button>

                <button
                  type="submit"
                  className="send-button"
                  aria-label={voice.enabled && voice.engine === "recorded" && !reviewTranscript ? "Transcribe recording" : "Send message"}
                  disabled={!session || busy || (voice.enabled ? !voice.canSend : !draft.trim())}
                >
                  {busy
                    ? <LoaderCircle size={20} className="spin" />
                    : <ArrowUp size={22} />}
                </button>
              </div>
            </form>
          )}

          <p className="footer-note">
            AI-assisted guidance, not a confirmed diagnosis.
            <span>For emergencies, seek urgent in-person help.</span>
          </p>
        </footer>
        </>}
      </main>

      <dialog
        ref={about}
        onClick={(event) => {
          if (event.target === about.current) about.current.close();
        }}
      >
        <div className="dialog-heading">
          <h2>About DDx Engine</h2>
          <button
            className="icon-button"
            aria-label="Close about"
            onClick={() => about.current?.close()}
          >
            <X />
          </button>
        </div>

        <p>
          A conversational symptom-assessment project. It collects information
          and may suggest a possibility; it cannot confirm a diagnosis.
        </p>
        <p>
          Messages and submitted recordings are processed by OpenAI.
          Conversations remain temporarily in server memory and clear on restart.
          Read-aloud uses a synthetic browser voice.
        </p>
        <p>
          Appointment profiles are fictional. Bookings are saved only in this browser,
          and no real practitioner is contacted. You can cancel them on the appointments page.
        </p>

        <button className="primary" onClick={() => about.current?.close()}>
          Back to conversation
        </button>
      </dialog>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);