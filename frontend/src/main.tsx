import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUp,
  ArrowUpRight,
  Check,
  CircleHelp,
  LoaderCircle,
  Menu,
  MessageCircle,
  Mic,
  Plus,
  ShieldCheck,
  Square,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import "./styles.css";

const API = "http://127.0.0.1:8000";
const WHATSAPP = "https://wa.link/u2krux";

type Reply = {
  session_id: string;
  reply: string;
  status: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  time: string;
  status?: string;
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
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [speaking, setSpeaking] = useState(false);
  const [expired, setExpired] = useState(false);

  const locked = useRef(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const microphoneStream = useRef<MediaStream | null>(null);
  const recordingClock = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordingStop = useRef<ReturnType<typeof setTimeout> | null>(null);
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

  function stopRecordingTimers() {
    if (recordingClock.current) clearInterval(recordingClock.current);
    if (recordingStop.current) clearTimeout(recordingStop.current);
  }

  useEffect(() => {
    bottom.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
      block: "end",
    });
  }, [messages, busy]);

  useEffect(() => {
    return () => {
      stopRecordingTimers();
      if (recorder.current) {
        recorder.current.onstop = null;
        if (recorder.current.state === "recording") {
          recorder.current.stop();
        }
      }
      microphoneStream.current?.getTracks().forEach((track) => track.stop());
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    };
  }, []);

  function handleError(error: unknown) {
    setError(
      error instanceof Error ? error.message : "Something went wrong."
    );

    if (error instanceof ApiError && error.status === 404) {
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
      },
    ]);
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
    if (locked.current || recording) return;

    if (
      messages.length &&
      !window.confirm("Start a new conversation? This conversation will be cleared.")
    ) {
      return;
    }

    setWorking(true);

    if ("speechSynthesis" in window) window.speechSynthesis.cancel();

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
    setSpeaking(false);
    setMenuOpen(false);
    pending.current = null;
    setWorking(false);
  }

  async function sendMessage() {
    const text = draft.trim();

    if (
      !text ||
      !session ||
      locked.current ||
      recording ||
      finished ||
      expired
    ) {
      return;
    }

    // Reuse the request ID when retrying the same message.
    if (!pending.current || pending.current.text !== text) {
      pending.current = { text, request_id: crypto.randomUUID() };
    }

    setWorking(true);
    setError("");
    setNotice("");

    try {
      const reply = await api<Reply>(
        "/sessions/" + session + "/messages",
        post(pending.current)
      );

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "user",
          text,
          time: timestamp(),
        },
      ]);

      setDraft("");
      pending.current = null;
      addReply(reply);
    } catch (error) {
      handleError(error);
    } finally {
      setWorking(false);
      setTimeout(() => input.current?.focus(), 0);
    }
  }

  function readAloud() {
    if (!lastReply) return;

    if (!("speechSynthesis" in window)) {
      setError("Read-aloud is unavailable. Please try Chrome or Edge.");
      return;
    }

    window.speechSynthesis.cancel();

    if (speaking) {
      setSpeaking(false);
      return;
    }

    const speech = new SpeechSynthesisUtterance(lastReply.text);
    speech.lang = "en";
    speech.rate = 0.95;
    speech.onend = () => setSpeaking(false);
    speech.onerror = () => setSpeaking(false);

    setSpeaking(true);
    window.speechSynthesis.speak(speech);
  }

  async function toggleMicrophone() {
    if (recorder.current?.state === "recording") {
      recorder.current.stop();
      return;
    }

    if (!session || locked.current || finished || expired) return;

    if (draft.trim()) {
      setError("Send or clear your draft before recording a new message.");
      return;
    }

    setWorking(true);
    setError("");
    setNotice("");

    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error("Use Chrome or Edge on localhost for microphone recording.");
      }

      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      setSpeaking(false);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      microphoneStream.current = stream;

      const mime = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ].find((type) => MediaRecorder.isTypeSupported(type));

      const recordingDevice = new MediaRecorder(
        stream,
        mime ? { mimeType: mime } : {}
      );

      recorder.current = recordingDevice;
      const chunks: BlobPart[] = [];
      let failed = false;

      recordingDevice.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };

      recordingDevice.onerror = () => {
        failed = true;
        stopRecordingTimers();
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        setWorking(false);
        setError("Recording failed. Please try again or type your message.");
      };

      recordingDevice.onstop = async () => {
        stopRecordingTimers();
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);

        if (failed) return;

        setWorking(true);
        setNotice("Turning your recording into text…");

        try {
          const type = recordingDevice.mimeType;
          const extension = type.includes("mp4")
            ? "mp4"
            : type.includes("ogg")
            ? "ogg"
            : "webm";

          const audio = new Blob(chunks, { type });

          if (!audio.size) throw new Error("The recording was empty. Try again.");

          const form = new FormData();
          form.append("audio", audio, "recording." + extension);

          const result = await api<{ text: string }>(
            "/sessions/" + session + "/transcribe",
            { method: "POST", body: form }
          );

          setDraft(result.text);
          setNotice("Check your transcript, then press Send.");
        } catch (error) {
          setNotice("");
          handleError(error);
        } finally {
          setWorking(false);
          setTimeout(() => input.current?.focus(), 0);
        }
      };

      recordingDevice.start();
      setSeconds(0);
      setRecording(true);
      setWorking(false);

      recordingClock.current = setInterval(
        () => setSeconds((value) => value + 1),
        1000
      );

      recordingStop.current = setTimeout(() => {
        if (recordingDevice.state === "recording") recordingDevice.stop();
      }, 30000);
    } catch (error) {
      microphoneStream.current?.getTracks().forEach((track) => track.stop());
      handleError(error);
      setWorking(false);
    }
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
          disabled={busy || recording}
        >
          <Plus size={19} /> New conversation
        </button>

        <button
          className="current-chat"
          aria-current="page"
          onClick={() => {
            setMenuOpen(false);
            input.current?.focus();
          }}
        >
          <MessageCircle size={19} /> Current conversation
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
            <h1>Let’s talk about how you feel.</h1>
            <p>Describe what’s bothering you. We’ll take it from there.</p>
          </div>

          <button
            className="read-button"
            onClick={readAloud}
            disabled={!lastReply || recording}
            aria-label={speaking ? "Stop reading" : "Read the last reply aloud"}
          >
            {speaking ? <VolumeX size={19} /> : <Volume2 size={19} />}
            <span>{speaking ? "Stop reading" : "Read aloud"}</span>
          </button>
        </header>

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
                          ["assessment", "uncertain"].includes(message.status || "") && (
                            <div className="handoff">
                              <a
                                href={WHATSAPP}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                <MessageCircle size={18} />
                                Continue on WhatsApp
                                <ArrowUpRight size={17} />
                              </a>
                              <small>Demo contact · Opens WhatsApp</small>
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

          {recording && (
            <div className="record-status" role="status">
              <span className="record-dot" />
              Recording {seconds}s / 30s — press stop when finished.
            </div>
          )}

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
                disabled={!session || busy || recording}
                placeholder={
                  !session
                    ? "Start a conversation above to begin…"
                    : recording
                    ? "Listening…"
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
                  aria-label={recording ? "Stop recording" : "Record a voice message"}
                  onClick={toggleMicrophone}
                  disabled={!session || busy}
                >
                  {recording
                    ? <Square size={18} fill="currentColor" />
                    : <Mic size={21} />}
                </button>

                <button
                  type="submit"
                  className="send-button"
                  aria-label="Send message"
                  disabled={!session || busy || recording || !draft.trim()}
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
          The WhatsApp link connects to the project’s demonstration contact,
          not a verified medical service.
        </p>

        <button className="primary" onClick={() => about.current?.close()}>
          Back to conversation
        </button>
      </dialog>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);