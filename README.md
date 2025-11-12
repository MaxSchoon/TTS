# GPT-4o Mini / ElevenLabs TTS CLI

Small CLI that sends text to either OpenAI's `gpt-4o-mini-tts` endpoint or ElevenLabs' text-to-speech API and writes the MP3/WAV output to disk. Output defaults to `~/Downloads/tts-output.mp3`, and once an audio file is generated the script opens the containing folder so you can immediately grab it. The OpenAI flow mirrors the [official docs](https://platform.openai.com/docs/models/gpt-4o-mini-tts) by posting JSON to `https://api.openai.com/v1/audio/speech`, and the ElevenLabs flow uses `https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`.

## Setup

1. Install the dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

   > On Linux you may need `python3-tk` for the GUI picker.

2. Copy the example environment and fill in the API key:

   ```bash
   cp .env.local.example .env.local
   ```

3. Edit `.env.local` with your `OPENAI_API_KEY` (add `OPENAI_PROJECT` if you're using project-scoped keys, and adjust `OPENAI_MODEL` / `OPENAI_VOICE` if desired).

## Usage

```bash
python3 tts.py --text "Hello world" --output greeting.mp3
```

OR read from a file:

```bash
python3 tts.py --input-file speech.txt --output ./dist/speech.wav --format wav
```

When the CLI starts it asks which provider you want to use (`1` for OpenAI, `2` for ElevenLabs). Pass `--provider openai` (or `elevenlabs`) if you prefer a non-interactive workflow. `--api-key` and `--project` always apply to the provider you selected.

Command-line options include:

- `--text` / `--input-file`: mutually exclusive source for the spoken text.
- `--output`: where the audio file will be written (extension optional; defaults to `.mp3`); if no path is provided the script writes to `~/Downloads/tts-output.mp3`.
- `--format`: encoding format (`mp3` or `wav`), defaults to `mp3`. (Automatic chunking currently targets MP3; choose MP3 for long texts.)
- `--voice`: override the voice from `.env.local`.
- `--model`: override the model name (defaults to `gpt-4o-mini-tts`).
- `--provider`: skip the interactive provider prompt.
- `--api-key`: provide your provider-specific API key inline instead of relying on `.env.local`.
- `--project`: set the `OpenAI-Project` (OpenAI) or `xi-project-id` (ElevenLabs) header when needed.
- `--choose-markdown` / `--choose-file`: open a GUI file picker and select any Markdown/TXT/PDF/DOCX file on your system.
- `--chunk-size`: override the automatic chunking threshold (default 3,400 characters). Set to `0` to disable chunking and enforce the model's raw limit.

If you run `python tts.py` without any options, the CLI opens a file picker so you can choose a supported document from anywhere on your machine. Long texts are automatically split into model-safe chunks (roughly 4,000 characters per request). The script always asks the TTS model to narrate the content as a relaxed, conversational podcast at natural 1× speed, and the default output format is MP3.

## Environment (or CLI) variables

### OpenAI

- `OPENAI_API_KEY` (required unless you specify `--api-key` while using `--provider openai`)
- `OPENAI_MODEL` defaults to `gpt-4o-mini-tts`
- `OPENAI_VOICE` defaults to `alloy`
- `OPENAI_PROJECT` optional; fills the `OpenAI-Project` header for project-scoped keys

### ElevenLabs

- `ELEVENLABS_API` (required for `--provider elevenlabs`)
- `ELEVENLABS_MODEL` defaults to `eleven_multilingual_v2`
- `ELEVENLABS_VOICE` defaults to the public "Rachel" voice ID (`21m00Tcm4TlvDq8ikWAM`). Replace this with any voice ID from your account.
- `ELEVENLABS_PROJECT` optional; when set it becomes the `xi-project-id` header. The sample `proj_xxxxx` placeholder is ignored—delete the line or supply a real ID if your workspace requires it.
- `ELEVENLABS_STABILITY`, `ELEVENLABS_SIMILARITY`, `ELEVENLABS_STYLE`, `ELEVENLABS_SPEAKER_BOOST` control the request's `voice_settings` payload. Defaults match ElevenLabs' recommended Rachel settings (0.55 stability / 0.75 similarity / 0 style / speaker boost on).

Run `python tts.py --provider elevenlabs --api-key <your-key> --project <proj>` (or `--provider openai`) to bypass the `.env.local` file if you prefer not to store secrets on disk.  
**Note:** OpenAI keys that start with `sk-proj-` always require an `OpenAI-Project` header (`OPENAI_PROJECT` or `--project`). The script refuses to run if it detects a project-scoped key without a matching project ID, preventing the opaque 401 errors OpenAI returns otherwise.

## How it works

The script loads `.env.local` (if present) via `python-dotenv`, validates that the API key for your selected provider exists, and then streams the appropriate POST request: OpenAI calls `/v1/audio/speech`, while ElevenLabs calls `/v1/text-to-speech/{voice_id}` with the requested `model_id`, `output_format`, optional `xi-project-id`, and `voice_settings` (stability, similarity, style, speaker boost). When OpenAI is selected, the script prepends an instruction telling the model to read the supplied content like a professional podcast host at natural 1× speed. Long inputs are split into 3.4k-character chunks (tweak with `--chunk-size`) so they stay below the GPT-4o Mini TTS limits, and each chunk is stitched into a single MP3 while a CLI progress bar tracks completion. By default the file is written to `~/Downloads/tts-output.mp3`, and once that completes the folder containing the file is opened so you can pick it right away. Adjusting `--format` toggles either the `format` payload field (OpenAI) or the ElevenLabs output format, while `--voice` lets you try different voices/voice IDs. Supported input formats include Markdown (`.md`), plain text (`.txt`), PDF (`.pdf`), and Word documents (`.docx`).

## Supported inputs

- `*.md` and `*.txt`: read directly as UTF-8 text (Markdown headings automatically strip the leading `#` so the narrator doesn't say “hashtag”.)
- `*.pdf`: requires `PyPDF2` to parse the PDF pages.
- `*.docx`: requires `python-docx` to read the document paragraphs.

`choose-file`/`choose-markdown` uses `tkinter`’s native file picker. Make sure your platform has the relevant GUI runtime (e.g., `python3-tk` on Debian-based Linux) if the picker fails to open.

## References

- OpenAI docs: https://platform.openai.com/docs/models/gpt-4o-mini-tts
- ElevenLabs docs: https://elevenlabs.io/docs/capabilities/text-to-speech
