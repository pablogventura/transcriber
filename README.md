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
summscriber interview.ogg --reply
summscriber audio.wav --reply --json
summscriber audio.wav --no-summary
```

### Main options

- **FILE**: audio file to transcribe (required).
- Summarization is **on by default**: OpenAI if configured, otherwise the shortest of pysummarization and sumy.
- `--no-summary`: do not generate a summary (transcription only, or with `--reply`/`--json` if you pass those).
- `--summary-pysummarization` / `--summary-sumy` / `--summary-openai`: also print a summary from a specific backend.
- `--summary-sentences N`: number of sentences in the summary (default 3).
- `--reply`: generate a short reply to the message with OpenAI.
- `--json`: output as JSON.

For summarization and reply with OpenAI, use a config file or environment variables. Configuration is read from the current directory, then from the user config dir (`~/.config/summscriber/config.ini` on Linux/macOS, `%APPDATA%\summscriber\config.ini` on Windows). See `config.ini.example`. Save your token and URL once (stored in the user config dir so it works from any directory):

```bash
summscriber --save-config --api-key YOUR_TOKEN --base-url https://...
```

### Open .ogg (and other audio) by default with the GUI

You can set **summscriber-gui** as the default application for `.ogg` files (and other audio formats your system associates with it). Then:

1. Download WhatsApp voice messages (or any audio) manually to a folder.
2. Click on an `.ogg` (or associated) file: it will open with the GUI, which transcribes the audio and shows the summary and suggested reply; the reply is copied to the clipboard so you can paste it straight into WhatsApp.

To set it as default: right‑click an `.ogg` file → “Open with” / “Abrir con” → choose **summscriber-gui** → tick “Use as default” / “Usar como predeterminado” (wording depends on your OS). On GNOME you can also use **Settings → Applications → Default applications** (or **Files → File types**).

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
