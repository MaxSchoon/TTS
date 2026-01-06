# Voice AI Customer Service

An experimental project exploring how voice models combined with fast, advanced LLMs can replace traditional customer service. The goal is to create AI agents that understand context better and answer customer questions more accurately than human representatives.

## Features

- **LLM Understanding**: Google Gemini 3 Flash (`gemini-3-flash-preview`) processes customer queries with context awareness
- **Text-to-Speech**: OpenAI `gpt-4o-mini-tts` or ElevenLabs converts responses to natural speech
- **Customer Service Mode**: `--gemini-query` flag pipes input through LLM before voice synthesis
- **Multi-format Input**: Supports Markdown, TXT, PDF, and DOCX files

## Setup

1. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

   > On Linux you may need `python3-tk` for the GUI picker.

2. Copy and configure environment:

   ```bash
   cp .env.local.example .env.local
   ```

3. Edit `.env.local` with your API keys:
   - `GEMINI_API_KEY` - For LLM understanding ([Get key](https://aistudio.google.com/apikey))
   - `OPENAI_API_KEY` - For OpenAI TTS
   - `ELEVENLABS_API` - For ElevenLabs TTS

## Usage

### Customer Service Mode (LLM + Voice)

Process a customer query through Gemini, then convert the response to speech:

```bash
python3 tts.py --gemini-query --text "How do I reset my password?"
```

### Standard Text-to-Speech

```bash
python3 tts.py --text "Your order has been shipped" --output notification.mp3
```

### From File

```bash
python3 tts.py --input-file response.txt --output ./dist/response.wav --format wav
```

### Test All Connections

```bash
python3 tts.py --test
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--text` | Text string to convert to speech |
| `--input-file` | Path to a text file (MD/TXT/PDF/DOCX) |
| `--output` | Output file path (defaults to `~/Downloads/tts-output.mp3`) |
| `--format` | Audio format: `mp3` or `wav` (default: `mp3`) |
| `--provider` | TTS provider: `openai` or `elevenlabs` |
| `--gemini-query` | Process input through Gemini LLM first (customer service mode) |
| `--gemini-api-key` | Provide Gemini API key directly |
| `--voice` | Override the default voice |
| `--model` | Override the model name |
| `--api-key` | Provide TTS API key directly |
| `--project` | Project ID for OpenAI or ElevenLabs |
| `--choose-file` | Open GUI file picker |
| `--chunk-size` | Override chunking threshold (default: 3400 chars) |
| `--test` | Test API connections |

## Environment Variables

### Gemini (LLM Understanding)

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google AI API key |
| `GEMINI_MODEL` | Model ID (default: `gemini-3-flash-preview`) |

### OpenAI (TTS)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for OpenAI TTS) |
| `OPENAI_MODEL` | TTS model (default: `gpt-4o-mini-tts`) |
| `OPENAI_VOICE` | Voice ID (default: `alloy`) |
| `OPENAI_PROJECT` | Project ID for project-scoped keys |

### ElevenLabs (TTS)

| Variable | Description |
|----------|-------------|
| `ELEVENLABS_API` | ElevenLabs API key |
| `ELEVENLABS_MODEL` | Model (default: `eleven_multilingual_v2`) |
| `ELEVENLABS_VOICE` | Voice ID |
| `ELEVENLABS_STABILITY` | Voice stability (0-1) |
| `ELEVENLABS_SIMILARITY` | Similarity boost (0-1) |
| `ELEVENLABS_STYLE` | Style (0-1) |
| `ELEVENLABS_SPEAKER_BOOST` | Speaker boost (`true`/`false`) |

## How It Works

1. **Input**: Customer query via text, file, or GUI picker
2. **LLM Processing** (optional): Gemini 3 Flash analyzes the query and generates a helpful response
3. **Voice Synthesis**: OpenAI or ElevenLabs converts the response to natural speech
4. **Output**: MP3/WAV audio file ready for playback

The system uses a customer service-optimized prompt that instructs the LLM to be helpful, accurate, and empathetic. The TTS voice is configured for warm, clear delivery suitable for customer interactions.

## Supported Input Formats

- `*.md` / `*.txt` - Read as UTF-8 (Markdown headings stripped for clean narration)
- `*.pdf` - Requires `PyPDF2`
- `*.docx` - Requires `python-docx`

## References

- [Gemini API Docs](https://ai.google.dev/gemini-api/docs/gemini-3)
- [OpenAI TTS Docs](https://platform.openai.com/docs/models/gpt-4o-mini-tts)
- [ElevenLabs Docs](https://elevenlabs.io/docs/capabilities/text-to-speech)
