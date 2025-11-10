# GPT-4o Mini TTS CLI

Small CLI that sends text to OpenAI's `gpt-4o-mini-tts` speech endpoint and writes the MP3/WAV output to disk. Output defaults to `~/Downloads/tts-output.mp3`, and once an audio file is generated the script opens the containing folder so you can immediately grab it. The flow mirrors the [official docs](https://platform.openai.com/docs/models/gpt-4o-mini-tts) by posting JSON to `https://api.openai.com/v1/audio/speech`.

## Setup

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
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

Command-line options include:

- `--text` / `--input-file`: mutually exclusive source for the spoken text.
- `--output`: where the audio file will be written (extension optional; defaults to `.mp3`); if no path is provided the script writes to `~/Downloads/tts-output.mp3`.
- `--format`: encoding format (`mp3` or `wav`), defaults to `mp3`. (Automatic chunking currently targets MP3; choose MP3 for long texts.)
- `--voice`: override the voice from `.env.local`.
- `--model`: override the model name (defaults to `gpt-4o-mini-tts`).
- `--api-key`: provide your OpenAI API key inline instead of relying on an environment file.
- `--project`: set the `OpenAI-Project` header when you use project-scoped keys (or set `OPENAI_PROJECT`).
- `--choose-markdown` / `--choose-file`: open a GUI file picker and select any Markdown/TXT/PDF/DOCX file on your system.
- `--chunk-size`: override the automatic chunking threshold (default 3,400 characters). Set to `0` to disable chunking and enforce the model's raw limit.

If you run `python tts.py` without any options, the CLI opens a file picker so you can choose a supported document from anywhere on your machine. Long texts are automatically split into model-safe chunks (roughly 4,000 characters per request). The script always asks the TTS model to narrate the content as a relaxed, conversational podcast at natural 1× speed, and the default output format is MP3.

## Environment (or CLI) variables

- `OPENAI_API_KEY` (required unless you specify `--api-key`)
- `OPENAI_MODEL` defaults to `gpt-4o-mini-tts`
- `OPENAI_VOICE` defaults to `alloy`
- `OPENAI_PROJECT` optional; fills the `OpenAI-Project` header for project-scoped keys

Run `python tts.py --api-key <your-key> [--project proj_xxx] …` to bypass the `.env.local` file if you prefer not to store secrets on disk.  
**Note:** Keys that start with `sk-proj-` always require an `OpenAI-Project` header (`OPENAI_PROJECT` or `--project`). The script now refuses to run if it detects a project-scoped key without a matching project ID, preventing the opaque 401 errors OpenAI returns otherwise.

## How it works

The script loads `.env.local` (if present) via `python-dotenv`, validates that the API key exists, and then streams a POST request to `/v1/audio/speech`. Before sending, it prepends an instruction telling the model to read the supplied content like a professional podcast host at natural 1× speed. Long inputs are split into 3.4k-character chunks (tweak with `--chunk-size`) so they stay below OpenAI's current limit for `gpt-4o-mini-tts`, and each chunk is stitched into a single MP3 while a CLI progress bar tracks completion. By default the file is written to `~/Downloads/tts-output.mp3`, and once that completes the folder containing the file is opened so you can pick it right away. Adjusting `--format` toggles the `format` payload field while `--voice` lets you try different voices. Supported input formats include Markdown (`.md`), plain text (`.txt`), PDF (`.pdf`), and Word documents (`.docx`).

## Supported inputs

- `*.md` and `*.txt`: read directly as UTF-8 text
- `*.pdf`: requires `PyPDF2` to parse the PDF pages.
- `*.docx`: requires `python-docx` to read the document paragraphs.

`choose-file`/`choose-markdown` uses `tkinter`’s native file picker. Make sure your platform has the relevant GUI runtime (e.g., `python3-tk` on Debian-based Linux) if the picker fails to open.

## References

- OpenAI docs: https://platform.openai.com/docs/models/gpt-4o-mini-tts
