"""Local-only class demonstration. Start with one uvicorn worker."""
import asyncio
import copy
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .core import evidence_answer, model_evidence, model_text, short_answer
from .conversation import (Observation, Extracted, EXTRACTION_INSTRUCTIONS,
                           apply_evidence, decide)
from .dialogue import phrase_question

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(HERE / '.env')
SESSIONS = {}
INFERENCE_LOCK = asyncio.Lock()
BUDGET_LOCK = asyncio.Lock()
COUNTS_PATH = HERE / 'usage_counts.json'
COUNTS = json.loads(COUNTS_PATH.read_text()) if COUNTS_PATH.exists() else {}
LIMITS = {'text': int(os.getenv('MAX_TEXT_CALLS', '60')),
          'transcription': int(os.getenv('MAX_TRANSCRIPTIONS', '20')),
          'speech': int(os.getenv('MAX_SPEECH_CALLS', '20'))}
ORIGINS = ['http://127.0.0.1:8000', 'http://localhost:8000',
           'http://127.0.0.1:5173', 'http://localhost:5173']


@asynccontextmanager
async def lifespan(app):
    global definitions, conditions, tokenizer, classifier, client, catalog
    definitions = json.loads(
        (ROOT / 'data/processed/evidence_definitions.json').read_text(encoding='utf-8'))
    raw = json.loads(
        (ROOT / 'data/processed/condition_definitions.json').read_text(encoding='utf-8'))
    conditions = {value['condition_name']: value for value in raw.values()}
    path = Path(
        os.getenv('MODEL_PATH', 'models/distilbert_20260916_013654/best_model'))
    if not path.is_absolute():
        path = ROOT / path
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, pad_token='[PAD]')
    classifier = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True).eval()
    key = os.getenv('OPENAI_API_KEY', '').strip()
    client = AsyncOpenAI(api_key=key, max_retries=0,
                         timeout=45) if key else None
    catalog = json.dumps({code: {
        'question': meta['question_en'], 'type': meta['data_type'],
        'values': ({str(v): evidence_answer(meta, [str(v)]) for v in meta.get('possible-values', [])}
                   if meta['data_type'] != 'B' else ['Yes', 'No'])
    } for code, meta in definitions.items()}, ensure_ascii=False)
    print('DDx model loaded locally. OpenAI key configured:',
          client is not None, flush=True)
    yield
    if client:
        await client.close()


app = FastAPI(title='DDx demonstration backend', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS,
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[
                   'localhost', '127.0.0.1', 'testserver'])


@app.middleware('http')
async def local_origin(request, call_next):
    if request.headers.get('origin') and request.headers['origin'] not in ORIGINS:
        return JSONResponse({'detail': 'Only the local demo interface may access this API.'}, 403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.exception_handler(APIError)
async def api_error(request, exc):
    status = getattr(exc, 'status_code', None)
    if status == 401:
        detail = 'OpenAI rejected the API key. Check backend/.env and restart.'
    elif status == 429:
        detail = 'OpenAI quota or rate limit reached. Check your API billing; no automatic retry was made.'
    elif status == 404:
        detail = 'Configured OpenAI model is unavailable for this key. Check backend/.env.'
    else:
        detail = 'OpenAI request failed or timed out. Your conversation was kept; try again when ready.'
    return JSONResponse({'detail': detail}, 502)


async def reserve(kind):
    if client is None:
        raise HTTPException(
            503, 'Set OPENAI_API_KEY in backend/.env, then restart the server.')
    async with BUDGET_LOCK:
        if COUNTS.get(kind, 0) >= LIMITS[kind]:
            raise HTTPException(
                429, f'Local {kind} request cap reached. Review usage before increasing it.')
        COUNTS[kind] = COUNTS.get(kind, 0) + 1
        # Count attempts before requests, including failures; never store symptoms.
        temporary = COUNTS_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(COUNTS), encoding='utf-8')
        temporary.replace(COUNTS_PATH)


class Start(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: Literal['M', 'F']


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=1800)
    request_id: UUID


def session_for(sid):
    session = SESSIONS.get(sid)
    if session is None or time.time() - session['touched'] > 7200:
        SESSIONS.pop(sid, None)
        raise HTTPException(
            404, 'Session expired or server restarted. Start a new conversation.')
    session['touched'] = time.time()
    return session


def response_for(sid, session, reply, status, prediction=None):
    session['last_reply'] = reply
    session['status'] = status
    session['audio'] = None
    return {'session_id': sid, 'reply': reply, 'status': status,
            'prediction': prediction, 'prediction_source': classifier.config.model_type if prediction else None,
            'offer_booking': status in ('assessment', 'uncertain', 'booking'),
            'recognized_findings': [{'question': definitions[k]['question_en'], 'answer': evidence_answer(definitions[k], v)} for k, v in session['answers'].items()],
            'voice_disclosure': 'Spoken replies use an AI-generated voice.'}


@torch.inference_mode()
def predict(session):
    text = model_text(session['age'], session['sex'],
                      session['answers'], definitions, session['initial'])
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256,
                       return_token_type_ids=False)
    scores = classifier(**inputs).logits.softmax(-1)[0]
    values, indices = scores.topk(min(5, len(scores)))
    return [(classifier.config.id2label[i], value) for i, value in zip(indices.tolist(), values.tolist())]


@app.get('/')
async def test_page():
    return FileResponse(HERE / 'test.html')


@app.get('/health')
async def health():
    return {'model_loaded': True, 'openai_configured': client is not None,
            'conditions': classifier.config.num_labels, 'calls': COUNTS, 'limits': LIMITS}


@app.post('/sessions')
async def start(body: Start):
    now = time.time()
    for sid in list(SESSIONS):
        if now - SESSIONS[sid]['touched'] > 7200:
            del SESSIONS[sid]
    if len(SESSIONS) >= 20:
        raise HTTPException(
            429, 'Too many active demo sessions. Delete an old session first.')
    sid = str(uuid4())
    session = {'age': body.age, 'sex': body.sex, 'answers': {}, 'asked': [], 'initial': None,
               'history': [], 'turns': 0, 'done': False, 'touched': now, 'lock': asyncio.Lock(),
               'responses': {}, 'last_question': '', 'last_reply': '', 'audio': None, 'pending_question': None}
    SESSIONS[sid] = session
    return response_for(sid, session, 'Hello. What has been bothering you, and when did it start?', 'question')


@app.delete('/sessions/{sid}')
async def delete(sid: str):
    SESSIONS.pop(sid, None)
    return {'deleted': True}


@app.post('/sessions/{sid}/messages')
async def chat(sid: str, body: Message):
    session = session_for(sid)
    async with session['lock']:
        request_id = str(body.request_id)
        if request_id in session['responses']:
            return session['responses'][request_id]
        text = body.text.strip()
        if not text:
            raise HTTPException(422, 'Please enter a message.')
        work = {k: copy.deepcopy(v) for k, v in session.items() if k != 'lock'}
        last_code = work.get('pending_question')
        direct = short_answer(text, last_code, definitions)
        if direct is not None:
            data = Extracted(observations=[Observation(code=k, values=v, quote=text)
                                          for k, v in direct.items()],
                             urgent=False, urgent_quote='', outside_scope=False)
            if not direct and last_code:
                data.unknown_observations = [Observation(code=last_code, values=[], quote=text)]
        else:
            await reserve('text')
            extraction = await client.responses.parse(
                model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), store=False,
                instructions=EXTRACTION_INSTRUCTIONS + '\nCATALOG:\n' + catalog,
                input=json.dumps({'pending_question': work['last_question'] if last_code else None,
                                  'known_findings': work['answers'],
                                  'recent_conversation': work['history'][-8:],
                                  'current_message': text}),
                text_format=Extracted, max_output_tokens=1600,
            )
            data = extraction.output_parsed
            if data is None:
                raise HTTPException(502, 'I could not interpret that message. Please retry.')
        work['turns'] += 1
        work['history'].append({'role': 'user', 'content': text})
        updates = apply_evidence(work, data, text, definitions)
        ranking = work.get('model_ranking', [])
        # Do not run diagnostic inference on an unclarified report of confusion.
        if (work['evidence_changed'] and model_evidence(work['answers'], definitions)
                and updates.get('E_39') != ['Yes']):
            async with INFERENCE_LOCK:
                ranking = await run_in_threadpool(predict, work)
        decision = decide(work, data, text, definitions, conditions, ranking, updates)
        reply = await phrase_question(client, reserve, decision, work)
        if decision.question_code:
            if decision.question_code not in work['asked']:
                work['asked'].append(decision.question_code)
            work['pending_question'] = decision.question_code
            work['last_question'] = reply
        work['done'] = decision.status == 'urgent'
        if decision.status in ('assessment', 'uncertain', 'urgent') or work['evidence_changed']:
            work['prediction'] = decision.prediction
        work['history'].append({'role': 'assistant', 'content': reply})
        work['history'] = work['history'][-24:]
        answer = response_for(sid, work, reply, decision.status, work.get('prediction'))
        work['responses'][request_id] = answer
        if len(work['responses']) > 100:
            del work['responses'][next(iter(work['responses']))]
        session.update(work)
        return answer


@app.get('/sessions/{sid}/debug')
async def debug(sid: str):
    session = session_for(sid)
    return {'purpose': 'Developer inspection only; scores are not clinical probabilities',
            'recognized_findings': {k: {'question': definitions[k]['question_en'], 'values': v} for k, v in session['answers'].items()},
            'model_input': model_text(session['age'], session['sex'], session['answers'], definitions, session['initial']),
            'model_ranking': session.get('model_ranking', []),
            'negative_findings_used_by_classifier': False,
            'extraction_audit': session.get('extraction_audit', {}),
            'unknown_findings': session.get('unknown', [])}


@app.post('/sessions/{sid}/transcribe')
async def transcribe(sid: str, audio: UploadFile = File(...)):
    session = session_for(sid)
    async with session['lock']:
        suffix = Path(audio.filename or '').suffix.lower()
        if suffix not in ('.webm', '.wav', '.mp3', '.m4a', '.mp4', '.ogg'):
            raise HTTPException(
                415, 'Use webm, wav, mp3, m4a, mp4, or ogg audio.')
        payload = await audio.read(4 * 1024 * 1024 + 1)
        await audio.close()
        if not payload or len(payload) > 4 * 1024 * 1024:
            raise HTTPException(
                413, 'Recording must be nonempty and under 4 MB. Keep recordings under 30 seconds.')
        await reserve('transcription')
        result = await client.audio.transcriptions.create(
            model=os.getenv('OPENAI_TRANSCRIBE_MODEL',
                            'gpt-4o-mini-transcribe'),
            file=('recording' + suffix, payload,
                  audio.content_type or 'application/octet-stream'),
            language='en', response_format='json')
        # Return editable text; do not silently submit potentially wrong transcription.
        return {'text': result.text}


@app.post('/sessions/{sid}/speech')
async def speech(sid: str):
    session = session_for(sid)
    async with session['lock']:
        if session['audio'] is None:
            await reserve('speech')
            result = await client.audio.speech.create(
                model=os.getenv('OPENAI_TTS_MODEL', 'gpt-4o-mini-tts'), voice='coral',
                input=session['last_reply'][:1600], response_format='mp3')
            session['audio'] = result.content
        return Response(session['audio'], media_type='audio/mpeg')
