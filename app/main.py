import base64
import io
import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI  # used only as a client for Ollama's OpenAI-compatible API

from app.config import settings
from app.agent import graph

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title='Medical Agent Educational API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(','),
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['Authorization', 'Content-Type', 'X-Conversation-ID'],
)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)
    image_data: str | None = None


def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(
        authorization or '', f'Bearer {settings.api_token}'
    ):
        raise HTTPException(status_code=401, detail='Invalid API token')


# ---------------------------------------------------------------------------
# Local model helpers
# ---------------------------------------------------------------------------

def ollama_client(timeout: int = 120) -> OpenAI:
    """OpenAI SDK pointed at the local Ollama server. The api_key is ignored by Ollama."""
    return OpenAI(base_url=settings.ollama_base_url, api_key='ollama', timeout=timeout)


_whisper_model = None


def get_whisper():
    """Load faster-whisper lazily so the server starts fast and only loads it when needed."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            getattr(settings, 'whisper_model', 'base'),
            device='cpu',
            compute_type='int8',
        )
    return _whisper_model


def inspect_image(data):
    if not data:
        return ''
    try:
        header, payload = data.split(',', 1)
        mime = header.split(';')[0].split(':')[1]
        raw = base64.b64decode(payload, validate=True)
        if mime not in ('image/jpeg', 'image/png', 'image/webp') or len(raw) > 5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f'Invalid image: {exc}')

    result = ollama_client().chat.completions.create(
        model=settings.ollama_vision_model,
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            'Describe only objectively visible medical-relevant features. '
                            'Do not diagnose, estimate severity, or assert absence of illness. '
                            'If image unclear, say so.'
                        ),
                    },
                    {'type': 'image_url', 'image_url': {'url': data}},
                ],
            }
        ],
        max_tokens=300,
    )
    return result.choices[0].message.content or ''


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get('/', response_class=HTMLResponse)
def home():
    return (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'configured': bool(settings.pinecone_api_key),
        'tavily': bool(settings.tavily_api_key),
        'llm': settings.ollama_chat_model,
        'vision': settings.ollama_vision_model,
    }


@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID'),
):
    auth(authorization)

    if not settings.pinecone_api_key:
        raise HTTPException(503, 'Set PINECONE_API_KEY in .env')

    history = [
        x
        for x in payload.history
        if x.get('role') in ('user', 'assistant') and isinstance(x.get('content'), str)
    ][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {'configurable': {'thread_id': thread_id}}

    try:
        observation = inspect_image(payload.image_data)

        state = graph.invoke(
            {
                'question': payload.message,
                'history': history,
                'image_description': observation,
                'kb': [],
                'web': [],
                'answer': '',
                'route': '',
            },
            config=config,
        )

        return {
            'answer': state['answer'],
            'sources': (
                [{'type': 'pdf', 'page': x['page'], 'score': x['score']} for x in state['kb']]
                + [{'type': 'web', 'url': x['url'], 'title': x['title']} for x in state['web']]
            ),
            'tools': (
                ['pinecone']
                + (['tavily'] if state['route'] == 'web' and settings.tavily_api_key else [])
                + (['vision'] if observation else [])
            ),
            'conversation_id': thread_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:', repr(exc), flush=True)
        raise HTTPException(
            502,
            'AI service failed. Check that Ollama is running (ollama serve), the models are '
            'pulled, and the Pinecone index/credentials are correct. See server logs.',
        )


@app.post('/api/transcribe')
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    auth(authorization)
    data = await file.read(10_000_001)
    if len(data) > 10_000_000:
        raise HTTPException(413, 'Audio exceeds 10 MB')

    try:
        segments, _ = get_whisper().transcribe(io.BytesIO(data))
        text = ' '.join(s.text.strip() for s in segments).strip()
        return {'text': text}
    except Exception as exc:
        print('Transcribe error:', repr(exc), flush=True)
        raise HTTPException(502, 'Speech transcription unavailable')


@app.post('/api/speak')
def speak(payload: dict, authorization: str | None = Header(default=None)):
    """Ollama has no text-to-speech. Use the browser's speechSynthesis in the frontend instead."""
    auth(authorization)
    raise HTTPException(
        501,
        'Server-side speech synthesis is not available with local models. '
        'Use the browser speechSynthesis API instead.',
    )