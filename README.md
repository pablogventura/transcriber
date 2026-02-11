# Transcriber

Transcribe audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper), with summarization options (pysummarization, sumy, OpenAI) and short reply generation via OpenAI API.

**Repository:** [github.com/pablogventura/transcriber](https://github.com/pablogventura/transcriber)

## Installation

### With pipx (recommended: isolated env, global command)

```bash
pipx install git+https://github.com/pablogventura/transcriber.git
```

To upgrade:

```bash
pipx upgrade transcriber
```

### With pip (from project directory)

```bash
pip install .
```

Or in editable mode (development):

```bash
pip install -e .
```

Or from the repository:

```bash
pip install git+https://github.com/pablogventura/transcriber.git
```

## Usage

After installing, the `transcriber` command is available:

```bash
transcriber FILE [options]
```

Examples:

```bash
transcriber recording.mp3
transcriber interview.ogg --summary
transcriber audio.wav --summary --reply --json
```

### Main options

- **FILE**: audio file to transcribe (required).
- `--summary`: summarize (OpenAI if token works; otherwise shortest of pysummarization and sumy).
- `--summary-pysummarization` / `--summary-sumy` / `--summary-openai`: use a specific summarization backend.
- `--summary-sentences N`: number of sentences in the summary (default 3).
- `--reply`: generate a short reply to the message with OpenAI.
- `--json`: output as JSON.

For summarization and reply with OpenAI, use `config.ini` (section `[openai]`) or environment variables `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`. See `config.ini.example`. You can save your token and URL with:

```bash
transcriber --save-config --api-key YOUR_TOKEN --base-url https://...
```

## Development

From the repo root without installing:

```bash
python -m transcriber FILE [options]
```
