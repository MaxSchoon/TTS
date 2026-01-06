#!/usr/bin/env python3
"""
Voice AI Customer Service

An experimental project exploring how voice models combined with fast, advanced
LLMs can replace traditional customer service. The goal is to create AI agents
that understand context better and answer customer questions more accurately
than human representatives.

Supports:
- Text-to-Speech: OpenAI gpt-4o-mini-tts, ElevenLabs
- LLM Understanding: Google Gemini 3 Flash (gemini-3-flash-preview)

Docs:
- OpenAI TTS: https://platform.openai.com/docs/models/gpt-4o-mini-tts
- Gemini: https://ai.google.dev/gemini-api/docs/gemini-3
"""

import argparse
import os
import re
import subprocess
import sys
import time
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
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DOC_URL = "https://platform.openai.com/docs/models/gpt-4o-mini-tts"
GEMINI_DOC_URL = "https://ai.google.dev/gemini-api/docs/gemini-3"
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env.local"
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_ELEVENLABS_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID
DEFAULT_ELEVENLABS_STABILITY = 0.55
DEFAULT_ELEVENLABS_SIMILARITY = 0.75
DEFAULT_ELEVENLABS_STYLE = 0.0
DEFAULT_ELEVENLABS_SPEAKER_BOOST = True
ELEVENLABS_MAX_RETRIES = 5
ELEVENLABS_RETRY_BASE_DELAY = 6.0
ELEVENLABS_RETRY_MAX_DELAY = 60.0
ELEVENLABS_RETRY_MULTIPLIER = 2.0
DEFAULT_FORMAT = "mp3"
DOWNLOADS_DIR = Path.home() / "Downloads"
DEFAULT_OUTPUT = DOWNLOADS_DIR / "tts-output.mp3"
MAX_MODEL_CHARS = 4000
DEFAULT_CHUNK_SIZE = 3400
CUSTOMER_SERVICE_INSTRUCTION = (
    "Instruction: Read the following text as a professional, helpful customer service representative. "
    "Keep the delivery warm, clear, and at a natural pace. Be empathetic and solution-oriented. "
    "Speak as if you genuinely want to help the customer resolve their issue. "
    "Do not read this instruction aloud—only the provided content."
)

# System prompt for Gemini LLM to process customer queries
GEMINI_SYSTEM_PROMPT = (
    "You are an AI customer service agent. Your role is to understand customer queries "
    "and provide helpful, accurate, and empathetic responses. Be concise but thorough. "
    "If you don't know something, acknowledge it honestly rather than making up information. "
    "Focus on solving the customer's problem efficiently."
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

    # Load backup API key if available (optional)
    backup_api_key = (os.getenv("ELEVENLABS_API_BACKUP") or "").strip() or None

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
    return api_key, backup_api_key, model, voice, project, source


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


def load_gemini_settings(cli_api_key: str | None = None):
    """Load Gemini API settings for LLM understanding."""
    api_key = (cli_api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None, None, None  # Gemini is optional

    model = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()

    if cli_api_key:
        source = "CLI argument"
    elif os.getenv("GEMINI_API_KEY"):
        source = "environment/.env.local"
    else:
        source = "unknown source"
    return api_key, model, source


def query_gemini(
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    """
    Query Gemini LLM to process customer input and generate a response.

    Args:
        api_key: Gemini API key
        model: Model ID (e.g., gemini-3-flash-preview)
        prompt: The customer's query/input
        system_prompt: Optional system instruction for the model

    Returns:
        The generated response text
    """
    url = f"{GEMINI_API_URL}/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    contents = []
    if system_prompt:
        contents.append({
            "role": "user",
            "parts": [{"text": f"System instruction: {system_prompt}"}]
        })
        contents.append({
            "role": "model",
            "parts": [{"text": "Understood. I will act as an AI customer service agent."}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    body = {"contents": contents}

    response = requests.post(url, json=body, headers=headers, timeout=60)
    response.raise_for_status()

    result = response.json()

    # Extract text from response
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response format: {result}") from exc


def test_gemini_connection(api_key: str, model: str) -> tuple[bool, str]:
    """Test Gemini API connection with a simple query."""
    try:
        response = query_gemini(
            api_key,
            model,
            "Hello, please respond with 'Connection successful'.",
        )
        if response:
            return True, f"✓ Gemini connection successful! Model: {model}"
        return False, "✗ Gemini connection failed: Empty response."
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else "unknown"
        detail = exc.response.text[:200] if exc.response and exc.response.text else ""
        return False, f"✗ Gemini connection failed (HTTP {status}): {detail}"
    except Exception as exc:
        return False, f"✗ Gemini connection failed: {type(exc).__name__}: {exc}"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Voice AI Customer Service - Transform text to speech with LLM understanding.",
        epilog=f"TTS Docs: {DOC_URL} | Gemini Docs: {GEMINI_DOC_URL}",
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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connection with a simple TTS request. Useful for diagnosing connection issues.",
    )
    parser.add_argument(
        "--gemini-query",
        action="store_true",
        help="Process input through Gemini LLM first (customer service mode). The LLM response is then converted to speech.",
    )
    parser.add_argument(
        "--gemini-api-key",
        help="Provide Gemini API key directly instead of relying on .env.local.",
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


def expand_crypto_abbreviations(text: str) -> str:
    """
    Expand common cryptocurrency abbreviations to their full terms for better TTS pronunciation.
    Uses word boundaries to avoid replacing abbreviations within words.
    """
    # Dictionary mapping abbreviations to their full terms
    abbreviations = {
        r"\bPOS\b": "Proof of Stake",
        r"\bPOW\b": "Proof of Work",
        r"\bDAO\b": "Decentralized Autonomous Organization",
        r"\bDEX\b": "Decentralized Exchange",
        r"\bDeFi\b": "Decentralized Finance",
        r"\bNFT\b": "Non-Fungible Token",
        r"\bICO\b": "Initial Coin Offering",
        r"\bIEO\b": "Initial Exchange Offering",
        r"\bAMM\b": "Automated Market Maker",
        r"\bTVL\b": "Total Value Locked",
        r"\bAPY\b": "Annual Percentage Yield",
    }
    
    expanded_text = text
    for abbrev, full_term in abbreviations.items():
        expanded_text = re.sub(abbrev, full_term, expanded_text, flags=re.IGNORECASE)
    
    return expanded_text


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


def extract_last_words(text: str, count: int = 5) -> str:
    words = text.strip().split()
    if not words:
        return ""
    return " ".join(words[-count:])


def report_partial_audio(
    output_path: Path,
    completed_chunks: int,
    total_chunks: int,
    last_words: str,
) -> None:
    if completed_chunks <= 0 and not output_path.exists():
        return
    chunk_message = (
        f"{completed_chunks}/{total_chunks}" if total_chunks > 0 else str(completed_chunks)
    )
    print(
        f"Partial audio saved to {output_path} ({chunk_message} chunks complete).",
        file=sys.stderr,
    )
    if last_words:
        print(f"Last completed words: \"{last_words}\"", file=sys.stderr)


def mark_partial_file(output_path: Path, completed_chunks: int) -> Path:
    if completed_chunks <= 0 or not output_path.exists():
        return output_path
    stem = output_path.stem
    if stem.endswith("PARTIAL"):
        return output_path
    partial_path = output_path.with_name(f"{stem}-PARTIAL{output_path.suffix}")
    try:
        output_path.rename(partial_path)
        return partial_path
    except OSError:
        return output_path


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


def should_retry_elevenlabs_error(
    response: requests.Response | None,
) -> tuple[bool, str]:
    if response is None:
        return False, ""
    status = response.status_code
    detail = (response.text or "").strip()
    lowered = detail.lower()
    if status == 429:
        return True, detail or "hit the ElevenLabs rate limit"
    if status == 401 and (not detail or "method not allowed" in lowered or "rate limit" in lowered):
        return True, detail or "hit the ElevenLabs rate limit"
    return False, detail


def test_api_connection(
    provider: str,
    api_key: str,
    model: str,
    voice: str,
    project: str | None = None,
    elevenlabs_voice_settings: dict | None = None,
) -> tuple[bool, str]:
    """
    Test API connection with a simple TTS request.
    Returns (success: bool, message: str).
    """
    test_text = "Hello. This is a connection test."
    fmt = "mp3"
    
    try:
        if provider == "openai":
            audio_bytes = synthesize_openai(
                api_key, model, voice, test_text, fmt, project
            )
            if len(audio_bytes) > 0:
                return True, f"✓ Connection successful! Generated {len(audio_bytes)} bytes of audio."
            else:
                return False, "✗ Connection failed: Received empty response."
        else:  # elevenlabs
            audio_bytes = synthesize_elevenlabs(
                api_key, model, voice, test_text, fmt, project, elevenlabs_voice_settings
            )
            if len(audio_bytes) > 0:
                return True, f"✓ Connection successful! Generated {len(audio_bytes)} bytes of audio."
            else:
                return False, "✗ Connection failed: Received empty response."
    except requests.HTTPError as exc:
        status = "unknown"
        detail = ""
        
        if exc.response is not None:
            status = exc.response.status_code
            try:
                # Try to read response text
                detail = (exc.response.text or "").strip()
                # If empty, try reading raw content (might be compressed)
                if not detail:
                    try:
                        raw_content = exc.response.content
                        if raw_content:
                            detail = f"Response body (raw, {len(raw_content)} bytes): {raw_content[:500].decode('utf-8', errors='replace')}"
                        else:
                            detail = f"Empty response body. Status: {status}"
                    except Exception:
                        detail = f"Could not read response body. Status: {status}"
            except Exception as e:
                detail = f"Error reading response: {e}"
        else:
            # HTTPError without response - this shouldn't normally happen
            detail = f"No response object. Exception: {str(exc)}"
        
        if status == 401:
            error_msg = f"✗ Authentication failed (HTTP 401). "
            if detail and "rate limit" in detail.lower():
                error_msg += f"Details: {detail[:500]}"
                error_msg += " (This appears to be a rate limit issue, not an authentication problem.)"
            elif detail:
                error_msg += f"Details: {detail[:500]}"
            else:
                error_msg += (
                    "Possible causes:\n"
                    "  - API key is invalid or expired\n"
                    "  - Voice ID is incorrect or not accessible with your API key\n"
                    "  - Account quota has been exceeded (ElevenLabs sometimes returns 401 for this)\n"
                    "  - API key doesn't have permission for this voice/model\n\n"
                    "Try:\n"
                    "  - Verify your API key at https://elevenlabs.io/\n"
                    "  - Check your account status and quota\n"
                    "  - Verify the voice ID is correct"
                )
            return False, error_msg
        elif status == 429:
            return False, (
                f"✗ Rate limit exceeded (HTTP 429). "
                f"Please wait before trying again. "
                f"{'Details: ' + detail[:200] if detail else ''}"
            )
        elif status == 403:
            return False, (
                f"✗ Forbidden (HTTP 403). "
                f"Your API key may not have permission or your account may need upgrading. "
                f"{'Details: ' + detail[:200] if detail else ''}"
            )
        else:
            return False, (
                f"✗ Connection failed (HTTP {status}). "
                f"{'Details: ' + detail[:200] if detail else 'No details available. '}"
                f"Exception: {type(exc).__name__}: {exc}"
            )
    except requests.RequestException as exc:
        return False, f"✗ Network error: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"✗ Unexpected error: {type(exc).__name__}: {exc}"


def invoke_with_retries(
    operation,
    provider: str,
    chunk_number: int,
    total_chunks: int,
    backup_operation=None,
) -> bytes:
    if provider != "elevenlabs":
        return operation()

    delay = ELEVENLABS_RETRY_BASE_DELAY
    last_exception = None
    used_backup_key = False
    current_operation = operation
    
    for attempt in range(1, ELEVENLABS_MAX_RETRIES + 1):
        try:
            return current_operation()
        except requests.HTTPError as exc:
            last_exception = exc
            should_retry, detail = should_retry_elevenlabs_error(exc.response)
            
            # Extract full error details for better diagnostics
            status = exc.response.status_code if exc.response is not None else "unknown"
            error_text = ""
            if exc.response is not None:
                try:
                    error_text = exc.response.text[:500] if exc.response.text else ""
                except Exception:
                    pass
            
            if not should_retry:
                # This is not a retryable error, raise immediately
                error_msg = f"HTTP {status}: {detail or error_text or 'Unknown error'}"
                raise RuntimeError(
                    f"Failed to process chunk {chunk_number}/{total_chunks}: {error_msg}"
                ) from exc
            
            # Try backup key if available and we haven't used it yet, and we've had a few failures
            if (
                backup_operation is not None
                and not used_backup_key
                and attempt >= 2  # Try backup after first failure
                and (status == 401 or status == 429)  # Only for auth/rate limit issues
            ):
                print(
                    f"Primary API key failed. Trying backup API key...",
                    file=sys.stderr,
                )
                current_operation = backup_operation
                used_backup_key = True
                delay = ELEVENLABS_RETRY_BASE_DELAY  # Reset delay when switching keys
                continue  # Retry immediately with backup key
            
            # If this is the last attempt, don't retry, just raise with a clear message
            if attempt >= ELEVENLABS_MAX_RETRIES:
                reason = (detail or "rate limited").strip() or "rate limited"
                key_info = " (tried both primary and backup keys)" if used_backup_key else ""
                raise RuntimeError(
                    f"Failed to process chunk {chunk_number}/{total_chunks} after "
                    f"{ELEVENLABS_MAX_RETRIES} attempts{key_info}. Last error: HTTP {status}: {reason}. "
                    f"{'Full error: ' + error_text if error_text else ''}"
                    f"Please wait and try again later, or reduce the chunk size. "
                    f"You can also test your connection with: python3 tts.py --test --provider elevenlabs"
                ) from exc
            
            wait_time = min(delay, ELEVENLABS_RETRY_MAX_DELAY)
            reason = (detail or "rate limited").strip() or "rate limited"
            key_label = "backup" if used_backup_key else "primary"
            print(
                (
                    f"ElevenLabs temporarily rejected chunk {chunk_number}/{total_chunks} "
                    f"(HTTP {status}: {reason}). Retrying in {wait_time:.1f}s with {key_label} key... "
                    f"(attempt {attempt}/{ELEVENLABS_MAX_RETRIES})"
                ),
                file=sys.stderr,
            )
            if error_text and attempt == 1:  # Show error details on first attempt
                print(f"Error details: {error_text[:200]}", file=sys.stderr)
            time.sleep(wait_time)
            delay = min(delay * ELEVENLABS_RETRY_MULTIPLIER, ELEVENLABS_RETRY_MAX_DELAY)
    
    # This should never be reached, but just in case
    if last_exception:
        raise RuntimeError(
            f"Exceeded ElevenLabs retry attempts for chunk {chunk_number}/{total_chunks}."
        ) from last_exception
    raise RuntimeError("Exceeded ElevenLabs retry attempts.")


def main():
    args = parse_arguments()
    load_env_file()
    provider = determine_provider(args.provider)

    loader = load_openai_settings if provider == "openai" else load_elevenlabs_settings
    backup_api_key = None
    try:
        if provider == "openai":
            (
                api_key,
                default_model,
                default_voice,
                default_project,
                api_key_source,
            ) = loader(args.api_key, args.project)
        else:  # elevenlabs
            (
                api_key,
                backup_api_key,
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
    if backup_api_key:
        print(
            f"Backup API key available: {mask_api_key(backup_api_key)}",
            file=sys.stderr,
        )
    if project:
        print(f"Using {provider_label} project: {project}", file=sys.stderr)

    # Load Gemini settings if available
    gemini_api_key, gemini_model, gemini_source = load_gemini_settings(
        getattr(args, 'gemini_api_key', None)
    )
    if gemini_api_key:
        print(
            f"Gemini LLM available ({gemini_model}) from {gemini_source}: {mask_api_key(gemini_api_key)}",
            file=sys.stderr,
        )

    # Handle test mode
    if args.test:
        all_tests_passed = True

        print(f"\nTesting {provider_label} API connection...", file=sys.stderr)
        print(f"Model: {model}", file=sys.stderr)
        print(f"Voice: {voice}", file=sys.stderr)
        if project:
            print(f"Project: {project}", file=sys.stderr)
        if backup_api_key:
            print(f"Backup API key: {mask_api_key(backup_api_key)} (available)", file=sys.stderr)
        print("", file=sys.stderr)

        # Test primary key
        print("Testing primary API key...", file=sys.stderr)
        success, message = test_api_connection(
            provider,
            api_key,
            model,
            voice,
            project,
            elevenlabs_voice_settings if provider == "elevenlabs" else None,
        )

        if success:
            print(message, file=sys.stderr)
        elif backup_api_key and provider == "elevenlabs":
            print(f"\nPrimary key failed: {message}", file=sys.stderr)
            print("\nTesting backup API key...", file=sys.stderr)
            backup_success, backup_message = test_api_connection(
                provider,
                backup_api_key,
                model,
                voice,
                project,
                elevenlabs_voice_settings,
            )
            if backup_success:
                print(f"✓ Backup key works! {backup_message}", file=sys.stderr)
                print("\nNote: The script will automatically use the backup key if the primary fails during processing.", file=sys.stderr)
            else:
                print(f"✗ Backup key also failed: {backup_message}", file=sys.stderr)
                all_tests_passed = False
        else:
            print(message, file=sys.stderr)
            all_tests_passed = False

        # Also test Gemini if available
        if gemini_api_key:
            print(f"\nTesting Gemini LLM connection...", file=sys.stderr)
            gemini_success, gemini_message = test_gemini_connection(
                gemini_api_key, gemini_model
            )
            print(gemini_message, file=sys.stderr)
            if not gemini_success:
                all_tests_passed = False

        sys.exit(0 if all_tests_passed else 1)

    text_to_speak, source_path = read_input_text(args)
    if not text_to_speak:
        print("Cannot synthesize empty text.", file=sys.stderr)
        sys.exit(1)

    # Process through Gemini LLM if --gemini-query is enabled
    if getattr(args, 'gemini_query', False):
        if not gemini_api_key:
            print(
                "Error: --gemini-query requires GEMINI_API_KEY in .env.local or --gemini-api-key.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Processing customer query through Gemini LLM...", file=sys.stderr)
        try:
            llm_response = query_gemini(
                gemini_api_key,
                gemini_model,
                text_to_speak,
                GEMINI_SYSTEM_PROMPT,
            )
            print(f"Gemini response generated ({len(llm_response)} chars)", file=sys.stderr)
            text_to_speak = llm_response
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "unknown"
            detail = exc.response.text[:200] if exc.response and exc.response.text else ""
            print(f"Gemini API error (HTTP {status}): {detail}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Gemini processing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(1)

    # Expand cryptocurrency abbreviations for better pronunciation
    # This ensures correct pronunciation for both OpenAI and ElevenLabs
    text_to_speak = expand_crypto_abbreviations(text_to_speak)
    use_input_name = source_path is not None and args.output is None
    if use_input_name and fmt != "mp3":
        print(
            "Forcing mp3 output when deriving the file name from the input document.",
            file=sys.stderr,
        )
        fmt = "mp3"

    if args.output is not None:
        desired_output = args.output
    elif source_path is not None:
        desired_output = source_path.with_suffix(".mp3")
    else:
        desired_output = DEFAULT_OUTPUT
    output_path = ensure_output_path(desired_output, fmt)

    instruction_budget = len(CUSTOMER_SERVICE_INSTRUCTION) + 2 if provider == "openai" else 0
    model_limit = max(1, MAX_MODEL_CHARS - instruction_budget)
    requested_chunk = args.chunk_size or 0
    chunk_size = (
        min(requested_chunk, model_limit) if requested_chunk > 0 else model_limit
    )

    total_chunks = 0
    completed_chunks = 0
    last_completed_words = ""

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
            backup_synthesize = None  # OpenAI doesn't support backup keys
        else:
            synthesize = lambda payload: synthesize_elevenlabs(
                api_key, model, voice, payload, fmt, project, elevenlabs_voice_settings
            )
            # Create backup synthesize function if backup key is available
            if backup_api_key:
                backup_synthesize = lambda payload: synthesize_elevenlabs(
                    backup_api_key, model, voice, payload, fmt, project, elevenlabs_voice_settings
                )
            else:
                backup_synthesize = None

        for idx, chunk in enumerate(chunks, start=1):
            payload_text = (
                f"{CUSTOMER_SERVICE_INSTRUCTION}\n\n{chunk}" if provider == "openai" else chunk
            )
            backup_operation = None
            if backup_synthesize:
                backup_operation = lambda chunk_payload=payload_text: backup_synthesize(chunk_payload)
            
            audio_bytes = invoke_with_retries(
                lambda chunk_payload=payload_text: synthesize(chunk_payload),
                provider,
                idx,
                total_chunks,
                backup_operation,
            )

            mode = "wb" if idx == 1 else "ab"
            with output_path.open(mode) as f:
                f.write(audio_bytes)

            print_progress(idx, total_chunks)
            completed_chunks = idx
            last_completed_words = extract_last_words(chunk)
    except requests.RequestException as exc:
        output_path = mark_partial_file(output_path, completed_chunks)
        report_partial_audio(output_path, completed_chunks, total_chunks, last_completed_words)
        print(f"Failed to call {provider_label} TTS API: {exc}", file=sys.stderr)
        if exc.response is not None:
            print("API response:", exc.response.text, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        output_path = mark_partial_file(output_path, completed_chunks)
        report_partial_audio(output_path, completed_chunks, total_chunks, last_completed_words)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Audio ready: {output_path}")
    reveal_output_folder(output_path.parent)


if __name__ == "__main__":
    main()
