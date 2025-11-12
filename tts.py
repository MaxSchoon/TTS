#!/usr/bin/env python3
"""
CLI helper that sends text to OpenAI's gpt-4o-mini-tts endpoint and saves
the resulting audio locally.

Docs: https://platform.openai.com/docs/models/gpt-4o-mini-tts
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None
try:
    from docx import Document
except ImportError:  # pragma: no cover - optional dependency
    Document = None

API_URL = "https://api.openai.com/v1/audio/speech"
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DOC_URL = "https://platform.openai.com/docs/models/gpt-4o-mini-tts"
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env.local"
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID
DEFAULT_ELEVENLABS_STABILITY = 0.55
DEFAULT_ELEVENLABS_SIMILARITY = 0.75
DEFAULT_ELEVENLABS_STYLE = 0.0
DEFAULT_ELEVENLABS_SPEAKER_BOOST = True
DEFAULT_FORMAT = "mp3"
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_OUTPUT = DOWNLOADS_DIR / "tts-output.mp3"
MAX_MODEL_CHARS = 4000
DEFAULT_CHUNK_SIZE = 3400
PODCAST_INSTRUCTION = (
    "Instruction: Read the following text like a polished, conversational podcast host. "
    "Keep the delivery relaxed, friendly, and at a natural 1x speed. "
    "Do not read this instruction aloud—only the provided content."
)
SUPPORTED_PROVIDERS = ("openai", "elevenlabs")
ELEVENLABS_OUTPUT_FORMATS = {
    "mp3": ("mp3_44100_128", "audio/mpeg"),
    "wav": ("wav_44100_16_bit_mono", "audio/wav"),
}
ELEVENLABS_PROJECT_PLACEHOLDERS = {"proj_xxxxx"}


def mask_api_key(key: str) -> str:
    """Return a masked representation of the API key for logging."""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def load_env_file() -> None:
    """Load .env.local (if present) exactly once."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)


def determine_provider(cli_choice: str | None) -> str:
    if cli_choice:
        return cli_choice
    if not sys.stdin.isatty():
        print(
            "No provider specified and stdin is non-interactive; defaulting to OpenAI.",
            file=sys.stderr,
        )
        return "openai"

    prompt = (
        "\nSelect a TTS provider:\n"
        "  1. OpenAI (gpt-4o-mini-tts)\n"
        "  2. ElevenLabs\n"
        "Enter 1 or 2: "
    )
    while True:
        try:
            choice = input(prompt).strip().lower()
        except KeyboardInterrupt:
            print("\nProvider selection canceled.", file=sys.stderr)
            sys.exit(1)
        mapping = {
            "1": "openai",
            "openai": "openai",
            "2": "elevenlabs",
            "elevenlabs": "elevenlabs",
        }
        if choice in mapping:
            return mapping[choice]
        print("Invalid selection. Please enter 1 for OpenAI or 2 for ElevenLabs.")


def load_openai_settings(
    cli_api_key: str | None = None, cli_project: str | None = None
):
    api_key = (cli_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. See .env.local.example.")

    model = (os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    voice = (os.getenv("OPENAI_VOICE") or DEFAULT_VOICE).strip()
    project = (cli_project or os.getenv("OPENAI_PROJECT") or "").strip() or None

    if api_key.startswith("sk-proj-") and not project:
        raise RuntimeError(
            "Detected a project-scoped key (sk-proj-...), but no OPENAI_PROJECT/--project was provided. "
            "Project-scoped keys always require the matching project ID."
        )

    if cli_api_key:
        source = "CLI argument"
    elif os.getenv("OPENAI_API_KEY"):
        source = "environment/.env.local"
    else:
        source = "unknown source"
    return api_key, model, voice, project, source


def load_elevenlabs_settings(
    cli_api_key: str | None = None, cli_project: str | None = None
):
    api_key = (cli_api_key or os.getenv("ELEVENLABS_API") or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API is missing. See .env.local.example.")

    model = (os.getenv("ELEVENLABS_MODEL") or DEFAULT_ELEVENLABS_MODEL).strip()
    voice = (os.getenv("ELEVENLABS_VOICE") or DEFAULT_ELEVENLABS_VOICE).strip()
    project = (cli_project or os.getenv("ELEVENLABS_PROJECT") or "").strip() or None
    if project and project in ELEVENLABS_PROJECT_PLACEHOLDERS:
        print(
            "Ignoring ELEVENLABS_PROJECT placeholder value. Remove it from your .env.local "
            "or set a real project ID if your workspace requires it.",
            file=sys.stderr,
        )
        project = None

    if not voice:
        raise RuntimeError(
            "ELEVENLABS_VOICE is missing. Provide a voice ID from your ElevenLabs account."
        )

    if cli_api_key:
        source = "CLI argument"
    elif os.getenv("ELEVENLABS_API"):
        source = "environment/.env.local"
    else:
        source = "unknown source"
    return api_key, model, voice, project, source


def load_elevenlabs_voice_settings() -> dict:
    def parse_float(name: str, default: float, min_value: float = 0.0, max_value: float = 1.0):
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            raise RuntimeError(f"{name} must be a number.")
        return max(min_value, min(max_value, value))

    def parse_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"{name} must be a boolean (true/false).")

    return {
        "stability": parse_float("ELEVENLABS_STABILITY", DEFAULT_ELEVENLABS_STABILITY),
        "similarity_boost": parse_float(
            "ELEVENLABS_SIMILARITY", DEFAULT_ELEVENLABS_SIMILARITY
        ),
        "style": parse_float("ELEVENLABS_STYLE", DEFAULT_ELEVENLABS_STYLE, 0.0, 1.0),
        "use_speaker_boost": parse_bool(
            "ELEVENLABS_SPEAKER_BOOST", DEFAULT_ELEVENLABS_SPEAKER_BOOST
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert text to speech with gpt-4o-mini-tts.",
        epilog=f"Docs: {DOC_URL}",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-t",
        "--text",
        help="Text string to convert into audio.",
    )
    input_group.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Path to a UTF-8 text file whose contents will be spoken.",
    )
    input_group.add_argument(
        "--choose-markdown",
        "--choose-file",
        action="store_true",
        dest="choose_file",
        help="Use a GUI picker to select any Markdown/TXT/PDF/DOCX file on your computer.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where the generated audio should be written. Defaults to the input file name (new extension) or ~/Downloads/tts-output.mp3.",
    )
    parser.add_argument(
        "--format",
        choices=("mp3", "wav"),
        default=DEFAULT_FORMAT,
        help="Audio encoding format. Default is mp3.",
    )
    parser.add_argument(
        "--voice",
        help="Override the default voice. Falls back to OPENAI_VOICE or alloy.",
    )
    parser.add_argument(
        "--model",
        help="Override the model name from the environment.",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="Specify the TTS provider (OpenAI or ElevenLabs). If omitted, the CLI will prompt you.",
    )
    parser.add_argument(
        "--api-key",
        help="Provide the provider-specific API key directly instead of relying on .env.local.",
    )
    parser.add_argument(
        "--project",
        help="Project ID header for the selected provider.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum characters per API request. Large texts are automatically split "
            "into chunks so they stay under gpt-4o-mini-tts limits. Set to 0 to disable chunking."
        ),
    )
    return parser.parse_args()


def choose_file_via_dialog() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("tkinter is required to use the file picker.")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filetypes = [
        ("Markdown", "*.md"),
        ("Text", "*.txt"),
        ("PDF", "*.pdf"),
        ("Word document", "*.docx"),
        ("All supported", "*.md *.txt *.pdf *.docx"),
        ("All files", "*.*"),
    ]

    path_str = filedialog.askopenfilename(
        title="Select a file to convert to speech",
        initialdir=str(Path.cwd()),
        filetypes=filetypes,
    )
    root.destroy()

    if not path_str:
        raise FileNotFoundError("File selection canceled.")
    return Path(path_str)


def read_file_contents(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        contents = path.read_text(encoding="utf-8", errors="ignore").strip()
        if suffix == ".md":
            contents = strip_markdown_headings(contents)
        return contents

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("Install PyPDF2 to read PDF files.")
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
        return "\n".join(texts).strip()

    if suffix == ".docx":
        if Document is None:
            raise RuntimeError("Install python-docx to read DOCX files.")
        doc = Document(path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs).strip()

    raise ValueError(f"Unsupported file extension: {suffix}")


def strip_markdown_headings(content: str) -> str:
    """Remove leading '#' markers so headings are spoken without hashtags."""
    cleaned_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            cleaned_lines.append(stripped)
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def chunk_text_for_model(text: str, limit: int) -> list[str]:
    """Split text into chunks that fit within the model's character limit."""
    sanitized_limit = max(1, limit)
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        extra_len = len(para) + (2 if current else 0)
        if current_len + extra_len <= sanitized_limit:
            current.append(para)
            current_len += extra_len
            continue

        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        if len(para) <= sanitized_limit:
            current = [para]
            current_len = len(para)
        else:
            for i in range(0, len(para), sanitized_limit):
                slice_text = para[i : i + sanitized_limit]
                chunks.append(slice_text)

    if current:
        chunks.append("\n\n".join(current))

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def print_progress(current: int, total: int, width: int = 30) -> None:
    if total <= 0:
        return
    fraction = current / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\rProcessing chunks [{bar}] {current}/{total}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def read_input_text(args: argparse.Namespace) -> tuple[str, Path | None]:
    if args.text:
        return args.text.strip(), None

    if args.choose_file:
        chosen_path = choose_file_via_dialog()
        return read_file_contents(chosen_path), chosen_path

    if args.input_file:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
        return read_file_contents(args.input_file), args.input_file

    print("No explicit input provided; opening a file picker so you can choose your source document.")
    chosen_path = choose_file_via_dialog()
    return read_file_contents(chosen_path), chosen_path


def ensure_output_path(base_path: Path, fmt: str) -> Path:
    output_path = base_path
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{fmt}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def reveal_output_folder(folder_path: Path) -> None:
    """Open the folder containing the generated audio so the user can pick the file."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", folder_path], check=False)
        elif sys.platform == "win32":
            subprocess.run(["explorer", folder_path], check=False)
        else:
            subprocess.run(["xdg-open", folder_path], check=False)
    except Exception:
        print(
            "Unable to open the folder automatically; you can find the file at "
            f"{folder_path}",
            file=sys.stderr,
        )



def synthesize_openai(
    api_key: str,
    model: str,
    voice: str,
    payload_text: str,
    fmt: str,
    project: str | None = None,
) -> bytes:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if project:
        headers["OpenAI-Project"] = project
    body = {
        "model": model,
        "voice": voice,
        "input": payload_text,
        "format": fmt,
    }

    audio_chunks: list[bytes] = []
    with requests.post(API_URL, json=body, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=8_192):
            if chunk:
                audio_chunks.append(chunk)
    return b"".join(audio_chunks)


def synthesize_elevenlabs(
    api_key: str,
    model: str,
    voice_id: str,
    payload_text: str,
    fmt: str,
    project: str | None = None,
    voice_settings: dict | None = None,
) -> bytes:
    format_name, accept_header = ELEVENLABS_OUTPUT_FORMATS[fmt]
    url = f"{ELEVENLABS_API_URL}/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": accept_header,
    }
    if project:
        headers["xi-project-id"] = project
    body = {
        "model_id": model,
        "text": payload_text,
    }
    if voice_settings:
        body["voice_settings"] = voice_settings
    params = {"output_format": format_name}
    audio_chunks: list[bytes] = []
    with requests.post(
        url, json=body, headers=headers, params=params, stream=True, timeout=60
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=8_192):
            if chunk:
                audio_chunks.append(chunk)
    return b"".join(audio_chunks)


def main():
    args = parse_arguments()
    load_env_file()
    provider = determine_provider(args.provider)

    loader = load_openai_settings if provider == "openai" else load_elevenlabs_settings
    try:
        (
            api_key,
            default_model,
            default_voice,
            default_project,
            api_key_source,
        ) = loader(args.api_key, args.project)
    except RuntimeError as exc:
        print(f"Environment error: {exc}", file=sys.stderr)
        sys.exit(1)

    model = args.model or default_model
    voice = args.voice or default_voice
    project = args.project or default_project
    fmt = args.format
    elevenlabs_voice_settings = None
    if provider == "elevenlabs":
        try:
            elevenlabs_voice_settings = load_elevenlabs_voice_settings()
        except RuntimeError as exc:
            print(f"Environment error: {exc}", file=sys.stderr)
            sys.exit(1)

    provider_label = "OpenAI" if provider == "openai" else "ElevenLabs"
    print(
        f"Using {provider_label} API key from {api_key_source}: {mask_api_key(api_key)}",
        file=sys.stderr,
    )
    if project:
        print(f"Using {provider_label} project: {project}", file=sys.stderr)

    text_to_speak, source_path = read_input_text(args)
    if not text_to_speak:
        print("Cannot synthesize empty text.", file=sys.stderr)
        sys.exit(1)
    if args.output is not None:
        desired_output = args.output
    elif source_path is not None:
        desired_output = source_path.with_suffix(f".{fmt}")
    else:
        desired_output = DEFAULT_OUTPUT
    output_path = ensure_output_path(desired_output, fmt)

    instruction_budget = len(PODCAST_INSTRUCTION) + 2 if provider == "openai" else 0
    model_limit = max(1, MAX_MODEL_CHARS - instruction_budget)
    requested_chunk = args.chunk_size or 0
    chunk_size = (
        min(requested_chunk, model_limit) if requested_chunk > 0 else model_limit
    )

    try:
        if len(text_to_speak) <= chunk_size:
            chunks = [text_to_speak.strip()]
        else:
            if requested_chunk <= 0:
                raise RuntimeError(
                    f"Input text is {len(text_to_speak)} characters, exceeding the "
                    f"model limit (~{model_limit}). Re-run with --chunk-size (default "
                    f"{DEFAULT_CHUNK_SIZE}) or shorten the text."
                )
            chunks = chunk_text_for_model(text_to_speak, chunk_size)
        if not chunks:
            raise RuntimeError("Unable to create chunks from the provided text.")

        if fmt == "wav" and len(chunks) > 1:
            raise RuntimeError(
                "Automatic chunking currently supports MP3 output only. "
                "Use --format mp3 for long texts or shorten your input."
        )

        total_chunks = len(chunks)
        print_progress(0, total_chunks)

        if provider == "openai":
            synthesize = lambda payload: synthesize_openai(
                api_key, model, voice, payload, fmt, project
            )
        else:
            synthesize = lambda payload: synthesize_elevenlabs(
                api_key, model, voice, payload, fmt, project, elevenlabs_voice_settings
            )

        for idx, chunk in enumerate(chunks, start=1):
            payload_text = (
                f"{PODCAST_INSTRUCTION}\n\n{chunk}" if provider == "openai" else chunk
            )
            audio_bytes = synthesize(payload_text)

            mode = "wb" if idx == 1 else "ab"
            with output_path.open(mode) as f:
                f.write(audio_bytes)

            print_progress(idx, total_chunks)
    except requests.RequestException as exc:
        print(f"Failed to call {provider_label} TTS API: {exc}", file=sys.stderr)
        if exc.response is not None:
            print("API response:", exc.response.text, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Audio ready: {output_path}")
    reveal_output_folder(output_path.parent)


if __name__ == "__main__":
    main()
