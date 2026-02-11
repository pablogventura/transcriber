"""Summscriber CLI: transcription, summarization, and reply generation."""

import argparse
import configparser
import json
import os
import sys
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel
from openai import OpenAI

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Config file name
CONFIG_FILENAME = "config.ini"


def _global_config_path() -> Path:
    """Path to config.ini in user config dir (same for all invocations)."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "summscriber" / CONFIG_FILENAME

# Language code (Whisper, ISO 639-1) -> sumy name (stopwords/stemmer)
_LANGUAGE_TO_SUMY = {
    "ar": "arabic", "zh": "chinese", "cs": "czech", "en": "english",
    "fr": "french", "de": "german", "el": "greek", "he": "hebrew",
    "it": "italian", "ja": "japanese", "ko": "korean", "pt": "portuguese",
    "sk": "slovak", "es": "spanish", "uk": "ukrainian",
}
# For OpenAI prompt: code -> language name
_LANGUAGE_CODE_TO_NAME = {
    "es": "Spanish", "en": "English", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "uk": "Ukrainian",
}


def _language_to_sumy(whisper_code: str) -> str:
    code = (whisper_code or "en")[:2].lower()
    return _LANGUAGE_TO_SUMY.get(code, "english")


def _language_name_for_prompt(whisper_code: str) -> str | None:
    code = (whisper_code or "")[:2].lower()
    return _LANGUAGE_CODE_TO_NAME.get(code)


def _load_openai_config() -> dict:
    """Read api_key, base_url and model from config.ini or environment variables."""
    # Look in: cwd, then global config dir, then next to package
    candidates = [
        Path.cwd() / CONFIG_FILENAME,
        _global_config_path(),
        Path(__file__).resolve().parent / CONFIG_FILENAME,
        Path(__file__).resolve().parent.parent / CONFIG_FILENAME,
    ]
    config_path = next((p for p in candidates if p.exists()), None)
    out = {"api_key": "", "base_url": "", "model": "gpt-4o-mini"}
    if config_path is not None:
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")
        if parser.has_section("openai"):
            out["api_key"] = parser.get("openai", "api_key", fallback="").strip()
            out["base_url"] = parser.get(
                "openai", "base_url", fallback=""
            ).strip() or "https://api.openai.com/v1"
            out["model"] = parser.get("openai", "model", fallback="gpt-4o-mini").strip()
    out["api_key"] = os.environ.get("OPENAI_API_KEY") or out["api_key"]
    out["base_url"] = os.environ.get("OPENAI_BASE_URL") or out["base_url"] or "https://api.openai.com/v1"
    return out


def _save_openai_config(
    api_key: str,
    base_url: str,
    model: str,
    config_path: Path,
) -> None:
    """Write config.ini with [openai] section."""
    parser = configparser.ConfigParser()
    if config_path.exists():
        parser.read(config_path, encoding="utf-8")
    if not parser.has_section("openai"):
        parser.add_section("openai")
    parser.set("openai", "api_key", api_key)
    parser.set("openai", "base_url", base_url)
    parser.set("openai", "model", model)
    with open(config_path, "w", encoding="utf-8") as f:
        parser.write(f)


def _ensure_nltk_sumy():
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def summarize_text(text: str, num_sentences: int = 3) -> str:
    if not text or not text.strip():
        return ""
    auto_abstractor = AutoAbstractor()
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    auto_abstractor.delimiter_list = [".", "\n", "?", "!"]
    abstractable_doc = TopNRankAbstractor()
    result_dict = auto_abstractor.summarize(text.strip(), abstractable_doc)
    sentences = result_dict.get("summarize_result", [])[:num_sentences]
    return " ".join(sentences) if sentences else ""


def summarize_text_sumy(text: str, num_sentences: int = 3, language: str = "spanish") -> str:
    if not text or not text.strip():
        return ""
    _ensure_nltk_sumy()
    try:
        parser = PlaintextParser.from_string(text.strip(), Tokenizer(language))
        stemmer = Stemmer(language)
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        sentences = summarizer(parser.document, num_sentences)
        return " ".join(str(s) for s in sentences) if sentences else ""
    except Exception:
        return ""


def summarize_text_openai(
    text: str,
    num_sentences: int = 3,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    detected_language: str | None = None,
) -> str:
    if not text or not text.strip():
        return ""
    cfg = _load_openai_config()
    key = api_key or cfg["api_key"]
    if not key:
        return ""
    base = base_url or cfg["base_url"]
    model_name = model or cfg["model"]
    client = OpenAI(api_key=key, base_url=base)
    lang_inst = f" The text is in {detected_language}." if detected_language else ""
    system = f"Summarize the text in a technical and concise way in approximately {num_sentences} sentences.{lang_inst} Use structure and hierarchy (e.g. lists) when possible. Summarize in the same language as the text."
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception:
        return ""


def reply_text_openai(
    text: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    detected_language: str | None = None,
) -> str:
    if not text or not text.strip():
        return ""
    cfg = _load_openai_config()
    key = api_key or cfg["api_key"]
    if not key:
        return ""
    base = base_url or cfg["base_url"]
    model_name = model or cfg["model"]
    client = OpenAI(api_key=key, base_url=base)
    lang_inst = f" Reply in {detected_language}." if detected_language else ""
    system = f"Reply briefly to the following message.{lang_inst} Be direct and concise."
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text.strip()},
            ],
            temperature=0.3,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper.")
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        metavar="FILE",
        help="Audio file to transcribe (required unless using --save-config)",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save OpenAI token and URL to config.ini (use --api-key and/or --base-url, or env vars).",
    )
    parser.add_argument(
        "--api-key",
        metavar="TOKEN",
        help="OpenAI API token (for --save-config). Falls back to OPENAI_API_KEY if not set.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="API base URL (for --save-config). Falls back to OPENAI_BASE_URL or default.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help="OpenAI model (default: gpt-4o-mini). Saved with --save-config.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="(Default: on) Summarize: use OpenAI if token works; otherwise shortest of pysummarization and sumy.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not generate a summary (disable default summarization).",
    )
    parser.add_argument(
        "--summary-pysummarization",
        action="store_true",
        help="Print a summary using pysummarization.",
    )
    parser.add_argument(
        "--summary-sentences",
        type=int,
        default=3,
        metavar="N",
        help="Number of sentences in the summary (default: 3).",
    )
    parser.add_argument(
        "--summary-sumy",
        action="store_true",
        help="Print a summary using sumy (LexRank).",
    )
    parser.add_argument(
        "--summary-openai",
        action="store_true",
        help="Print a summary using OpenAI API (Ollama/CCAD). Requires config.ini or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--reply",
        action="store_true",
        help="Generate a short reply to the transcribed message using OpenAI.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (text, language, summary and/or reply depending on options).",
    )
    args = parser.parse_args()

    if args.save_config:
        api_key = (args.api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (
            (args.base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
            or "https://api.openai.com/v1"
        )
        if not api_key:
            print("Error: provide --api-key or set OPENAI_API_KEY to save configuration.")
            sys.exit(1)
        config_path = _global_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _save_openai_config(api_key, base_url, args.model or "gpt-4o-mini", config_path)
        print(f"Configuration saved to {config_path.resolve()}")
        return

    if args.audio is None:
        parser.error("the following arguments are required: FILE (unless --save-config)")

    try:
        gpu_count = ctranslate2.get_cuda_device_count()
        use_cuda = gpu_count > 0
    except Exception:
        use_cuda = False

    if use_cuda:
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    if not args.json:
        print("Using GPU (CUDA)" if use_cuda else "Using CPU")

    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    segments, info = model.transcribe(args.audio)

    language_code = getattr(info, "language", None) or ""
    sumy_language = _language_to_sumy(language_code)
    language_for_prompt = _language_name_for_prompt(language_code)

    full_text = " ".join(s.text for s in segments).strip()

    output = {}
    if args.json:
        output["text"] = full_text
        output["language"] = language_code
        output["device"] = "cuda" if use_cuda else "cpu"
    else:
        print(full_text)
        if full_text:
            print()

    n = args.summary_sentences

    if full_text and not args.no_summary:
        cfg = _load_openai_config()
        summary_openai = ""
        if cfg["api_key"]:
            summary_openai = summarize_text_openai(
                full_text, num_sentences=n, detected_language=language_for_prompt
            )
        if summary_openai:
            if args.json:
                output["summary"] = summary_openai
                output["summary_source"] = "openai"
            else:
                print("--- Summary (openai) ---")
                print(summary_openai)
        else:
            summary_py = summarize_text(full_text, num_sentences=n)
            summary_sumy = summarize_text_sumy(
                full_text, num_sentences=n, language=sumy_language
            )
            candidates = [
                (summary_py, "pysummarization"),
                (summary_sumy, "sumy"),
            ]
            candidates = [(t, name) for t, name in candidates if t]
            if candidates:
                shortest = min(candidates, key=lambda x: len(x[0]))
                summary_text, name = shortest
                if args.json:
                    output["summary"] = summary_text
                    output["summary_source"] = name
                else:
                    print(f"--- Summary ({name}, shortest) ---")
                    print(summary_text)
            elif not args.json:
                print("(Text too short to generate summary)")

    if args.summary_pysummarization and full_text:
        summary = summarize_text(full_text, num_sentences=n)
        if summary:
            if args.json:
                output["summary_pysummarization"] = summary
            else:
                print("--- Summary (pysummarization) ---")
                print(summary)
        elif not args.json:
            print("(Text too short to generate summary)")

    if args.summary_sumy and full_text:
        summary = summarize_text_sumy(
            full_text, num_sentences=n, language=sumy_language
        )
        if summary:
            if args.json:
                output["summary_sumy"] = summary
            else:
                print("--- Summary (sumy) ---")
                print(summary)
        elif not args.json:
            print("(Text too short to generate summary with sumy)")

    if args.summary_openai and full_text:
        cfg = _load_openai_config()
        if not cfg["api_key"]:
            if not args.json:
                print(
                    "Error: --summary-openai requires api_key in config.ini ([openai] section) "
                    "or OPENAI_API_KEY environment variable."
                )
            else:
                output["error_summary_openai"] = "Missing api_key in config.ini or OPENAI_API_KEY"
        else:
            summary = summarize_text_openai(
                full_text, num_sentences=n, detected_language=language_for_prompt
            )
            if summary:
                if args.json:
                    output["summary_openai"] = summary
                else:
                    print("--- Summary (openai) ---")
                    print(summary)
            elif not args.json:
                print("(Could not generate summary with OpenAI)")

    if args.reply and full_text:
        cfg = _load_openai_config()
        if not cfg["api_key"]:
            if not args.json:
                print(
                    "Error: --reply requires api_key in config.ini ([openai] section) "
                    "or OPENAI_API_KEY environment variable."
                )
            else:
                output["error_reply"] = "Missing api_key in config.ini or OPENAI_API_KEY"
        else:
            reply = reply_text_openai(
                full_text, detected_language=language_for_prompt
            )
            if reply:
                if args.json:
                    output["reply"] = reply
                else:
                    print("--- Reply ---")
                    print(reply)
            elif not args.json:
                print("(Could not generate reply with OpenAI)")

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
