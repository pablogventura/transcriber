# Summscriber

Transcribe audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper), with summarization options (pysummarization, sumy, OpenAI) and short reply generation via OpenAI API.

**Repository:** [github.com/pablogventura/summscriber](https://github.com/pablogventura/summscriber)

## Installation

### With pipx (recommended: isolated env, global command)

```bash
pipx install git+https://github.com/pablogventura/summscriber.git
```

To upgrade:

```bash
pipx upgrade summscriber
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
pip install git+https://github.com/pablogventura/summscriber.git
```

Or from PyPI (once published):

```bash
pip install summscriber
```

## Usage

After installing, the `summscriber` command is available:

```bash
summscriber FILE [options]
```

Examples:

```bash
summscriber recording.mp3
summscriber interview.ogg --summary
summscriber audio.wav --summary --reply --json
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
summscriber --save-config --api-key YOUR_TOKEN --base-url https://...
```

## Development

From the repo root without installing:

```bash
python -m summscriber FILE [options]
```

## Publishing to PyPI

1. Install build tools: `pip install build twine` (or `pip install ".[publish]"`).
2. Create a PyPI account and an API token at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/).
3. From the project root run:

   ```bash
   ./publish.sh
   ```

   Or manually:

   ```bash
   rm -rf build dist *.egg-info
   python -m build
   twine upload dist/*
   ```

   When prompted, use username `__token__` and password your PyPI token. Or set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=pypi-your-token`.
