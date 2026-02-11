import argparse
import configparser
import os
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

# Archivo de configuración (en el directorio del script o del cwd)
CONFIG_FILENAME = "config.ini"

# Código de idioma (Whisper, ISO 639-1) -> nombre para sumy (stopwords/stemmer)
# Sumy tiene: arabic, chinese, czech, english, french, german, greek, hebrew,
# italian, japanese, korean, portuguese, slovak, spanish, ukrainian
_IDIOMA_WHISPER_A_SUMY = {
    "ar": "arabic", "zh": "chinese", "cs": "czech", "en": "english",
    "fr": "french", "de": "german", "el": "greek", "he": "hebrew",
    "it": "italian", "ja": "japanese", "ko": "korean", "pt": "portuguese",
    "sk": "slovak", "es": "spanish", "uk": "ukrainian",
}
# Para el prompt de OpenAI: código -> nombre del idioma
_IDIOMA_CODIGO_A_NOMBRE = {
    "es": "español", "en": "inglés", "fr": "francés", "de": "alemán",
    "it": "italiano", "pt": "portugués", "ar": "árabe", "zh": "chino",
    "ja": "japonés", "ko": "coreano", "ru": "ruso",     "uk": "ucraniano",
}


def _idioma_sumy(codigo_whisper: str) -> str:
    """Convierte código de idioma de Whisper al nombre que usa sumy. Si no hay soporte, usa english."""
    codigo = (codigo_whisper or "en")[:2].lower()
    return _IDIOMA_WHISPER_A_SUMY.get(codigo, "english")


def _idioma_nombre_para_prompt(codigo_whisper: str) -> str | None:
    """Nombre del idioma para el prompt de OpenAI; None si no se conoce."""
    codigo = (codigo_whisper or "")[:2].lower()
    return _IDIOMA_CODIGO_A_NOMBRE.get(codigo)


def _cargar_config_openai() -> dict:
    """Lee api_key, base_url y model desde config.ini o variables de entorno.
    Las variables de entorno tienen prioridad sobre el archivo."""
    config_dir = Path(__file__).resolve().parent
    config_path = config_dir / CONFIG_FILENAME
    # También buscar en cwd por si se ejecuta desde otro directorio
    if not config_path.exists():
        config_path = Path.cwd() / CONFIG_FILENAME
    out = {"api_key": "", "base_url": "", "model": "gpt-oss:20b"}
    if config_path.exists():
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="utf-8")
        if parser.has_section("openai"):
            out["api_key"] = parser.get("openai", "api_key", fallback="").strip()
            out["base_url"] = parser.get(
                "openai", "base_url", fallback=""
            ).strip() or "https://chat.ccad.unc.edu.ar/ollama/v1"
            out["model"] = parser.get("openai", "model", fallback="gpt-oss:20b").strip()
    out["api_key"] = os.environ.get("OPENAI_API_KEY") or out["api_key"]
    out["base_url"] = os.environ.get("OPENAI_BASE_URL") or out["base_url"] or "https://chat.ccad.unc.edu.ar/ollama/v1"
    return out


def _asegurar_nltk_sumy():
    """Descarga datos NLTK necesarios para sumy (español)."""
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def resumir_texto(texto: str, num_oraciones: int = 3) -> str:
    """Genera un resumen del texto con pysummarization."""
    if not texto or not texto.strip():
        return ""
    auto_abstractor = AutoAbstractor()
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    auto_abstractor.delimiter_list = [".", "\n", "?", "!"]
    abstractable_doc = TopNRankAbstractor()
    result_dict = auto_abstractor.summarize(texto.strip(), abstractable_doc)
    oraciones = result_dict.get("summarize_result", [])[:num_oraciones]
    return " ".join(oraciones) if oraciones else ""


def resumir_texto_sumy(texto: str, num_oraciones: int = 3, idioma: str = "spanish") -> str:
    """Genera un resumen del texto con sumy (LexRank)."""
    if not texto or not texto.strip():
        return ""
    _asegurar_nltk_sumy()
    parser = PlaintextParser.from_string(texto.strip(), Tokenizer(idioma))
    stemmer = Stemmer(idioma)
    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(idioma)
    try:
        oraciones = summarizer(parser.document, num_oraciones)
        return " ".join(str(s) for s in oraciones) if oraciones else ""
    except Exception:
        return ""


def resumir_texto_openai(
    texto: str,
    num_oraciones: int = 3,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    idioma_detectado: str | None = None,
) -> str:
    """Genera un resumen del texto con la API OpenAI (ej. Ollama en CCAD)."""
    if not texto or not texto.strip():
        return ""
    cfg = _cargar_config_openai()
    key = api_key or cfg["api_key"]
    if not key:
        return ""
    base = base_url or cfg["base_url"]
    model_name = model or cfg["model"]
    client = OpenAI(api_key=key, base_url=base)
    idioma_inst = f" El texto está en {idioma_detectado}." if idioma_detectado else ""
    system = f"Resumí el texto en forma técnica y breve en aproximadamente {num_oraciones} oraciones.{idioma_inst} Si es posible muestralo con estructura y jerarquía, en el sentido de listas. Resumí en el mismo idioma del texto."
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": texto.strip()},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio con Whisper.")
    parser.add_argument(
        "audio",
        metavar="ARCHIVO",
        help="Archivo de audio a transcribir (ej. grabacion.mp3, entrada.ogg)",
    )
    parser.add_argument(
        "--resumen",
        action="store_true",
        help="Resumir: si hay token OpenAI y responde, usa ese; si no, el más corto entre pysummarization y sumy",
    )
    parser.add_argument(
        "--resumen-pysummarization",
        action="store_true",
        help="Imprimir un resumen del texto con pysummarization",
    )
    parser.add_argument(
        "--resumen-oraciones",
        type=int,
        default=3,
        metavar="N",
        help="Número de oraciones del resumen (por defecto: 3)",
    )
    parser.add_argument(
        "--resumen-sumy",
        action="store_true",
        help="Imprimir un resumen del texto con sumy (LexRank)",
    )
    parser.add_argument(
        "--resumen-openai",
        action="store_true",
        help="Imprimir un resumen con la API OpenAI (Ollama/CCAD). Requiere OPENAI_API_KEY.",
    )
    args = parser.parse_args()

    # Detectar si hay GPU disponible
    try:
        gpu_count = ctranslate2.get_cuda_device_count()
        use_cuda = gpu_count > 0
    except Exception:
        use_cuda = False

    if use_cuda:
        device, compute_type = "cuda", "float16"
        print("Usando GPU (CUDA)")
    else:
        device, compute_type = "cpu", "int8"
        print("Usando CPU")

    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    segments, info = model.transcribe(args.audio)

    # Idioma detectado por Whisper (ej. "es", "en"); se usa en resúmenes
    idioma_codigo = getattr(info, "language", None) or ""
    sumy_idioma = _idioma_sumy(idioma_codigo)
    idioma_prompt = _idioma_nombre_para_prompt(idioma_codigo)

    texto_completo = " ".join(s.text for s in segments).strip()
    print(texto_completo)
    if texto_completo:
        print()

    n = args.resumen_oraciones

    if args.resumen and texto_completo:
        cfg = _cargar_config_openai()
        resumen_openai = ""
        if cfg["api_key"]:
            resumen_openai = resumir_texto_openai(
                texto_completo, num_oraciones=n, idioma_detectado=idioma_prompt
            )
        if resumen_openai:
            print("--- Resumen (openai) ---")
            print(resumen_openai)
        else:
            # Sin token o OpenAI no respondió: usar pysummarization y sumy, elegir el más corto
            resumen_py = resumir_texto(texto_completo, num_oraciones=n)
            resumen_sumy = resumir_texto_sumy(
                texto_completo, num_oraciones=n, idioma=sumy_idioma
            )
            candidatos = [
                (resumen_py, "pysummarization"),
                (resumen_sumy, "sumy"),
            ]
            candidatos = [(t, nombre) for t, nombre in candidatos if t]
            if candidatos:
                mas_corto = min(candidatos, key=lambda x: len(x[0]))
                texto_resumen, nombre = mas_corto
                print(f"--- Resumen ({nombre}, el más corto) ---")
                print(texto_resumen)
            else:
                print("(Texto demasiado corto para generar resumen)")

    if args.resumen_pysummarization and texto_completo:
        resumen = resumir_texto(texto_completo, num_oraciones=n)
        if resumen:
            print("--- Resumen (pysummarization) ---")
            print(resumen)
        else:
            print("(Texto demasiado corto para generar resumen)")

    if args.resumen_sumy and texto_completo:
        resumen = resumir_texto_sumy(
            texto_completo, num_oraciones=n, idioma=sumy_idioma
        )
        if resumen:
            print("--- Resumen (sumy) ---")
            print(resumen)
        else:
            print("(Texto demasiado corto para generar resumen con sumy)")

    if args.resumen_openai and texto_completo:
        cfg = _cargar_config_openai()
        if not cfg["api_key"]:
            print(
                "Error: --resumen-openai requiere api_key en config.ini (sección [openai]) "
                "o la variable de entorno OPENAI_API_KEY."
            )
        else:
            resumen = resumir_texto_openai(
                texto_completo, num_oraciones=n, idioma_detectado=idioma_prompt
            )
            if resumen:
                print("--- Resumen (openai) ---")
                print(resumen)
            else:
                print("(No se pudo generar resumen con OpenAI)")


if __name__ == "__main__":
    main()
