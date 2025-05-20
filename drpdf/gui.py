import asyncio
import cgi
import os
import shutil
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from drpdf import __version__
from drpdf.high_level import translate
from drpdf.doclayout import ModelInstance
from drpdf.config import ConfigManager
from drpdf.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    OpenRouterTranslator
)

logger = logging.getLogger(__name__)
from babeldoc.docvision.doclayout import OnnxModel

BABELDOC_MODEL = OnnxModel.load_available()

# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "OpenRouter": OpenRouterTranslator
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
    "Arabic": "ar",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["Google", "Bing"]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError(f"No services available.")
    enabled_services = default_services + enabled_services
else:
    enabled_services = list(service_map.keys())


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("drpdf_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        progress(t.n / t.total, desc=desc)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "model": ModelInstance.value,
    }

    try:
        if use_babeldoc:
            return babeldoc_translate_file(**param)
        translate(**param)
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    if kwargs["prompt"]:
        prompt = kwargs["prompt"]
    else:
        prompt = None

    from drpdf.translator import (
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        XinferenceTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
    )

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # Start translation
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
from babeldoc import __version__ as babeldoc_version

# Define a modern color palette
primary_color = "#2563EB"  # Blue
secondary_color = "#4B5563"  # Gray
success_color = "#10B981"  # Green
warning_color = "#F59E0B"  # Amber
error_color = "#EF4444"  # Red
background_color = "#F9FAFB"  # Light gray
card_color = "#FFFFFF"  # White

# Create a custom theme with the modern color palette
custom_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#EFF6FF",
        c100="#DBEAFE",
        c200="#BFDBFE",
        c300="#93C5FD",
        c400="#60A5FA",
        c500=primary_color,  # Primary color
        c600="#2563EB",
        c700="#1D4ED8",
        c800="#1E40AF",
        c900="#1E3A8A",
        c950="#172554",
    ),
    secondary_hue=gr.themes.Color(
        c50="#F9FAFB",
        c100="#F3F4F6",
        c200="#E5E7EB",
        c300="#D1D5DB",
        c400="#9CA3AF",
        c500=secondary_color,  # Secondary color
        c600="#4B5563",
        c700="#374151",
        c800="#1F2937",
        c900="#111827",
        c950="#030712",
    ),
    neutral_hue=gr.themes.Color(
        c50="#F9FAFB",
        c100="#F3F4F6",
        c200="#E5E7EB",
        c300="#D1D5DB",
        c400="#9CA3AF",
        c500="#6B7280",
        c600="#4B5563",
        c700="#374151",
        c800="#1F2937",
        c900="#111827",
        c950="#030712",
    ),
    spacing_size="md",
    radius_size="lg",
    text_size="md",
)

# Enhanced CSS for a more professional look
custom_css = """
    /* Global styles */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #F9FAFB;
    }
    
    /* Header styles */
    .app-header {
        padding: 1.5rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .app-header h1 {
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin: 0 !important;
        padding: 0 1rem !important;
    }
    
    .app-header a {
        color: white !important;
        text-decoration: none !important;
    }
    
    /* Logo styles */
    .app-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .app-logo img {
        height: 3rem;
        width: auto;
    }
    
    /* Card styles */
    .card {
        background-color: white;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .card-header {
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        color: #1F2937;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-header i {
        color: #2563EB;
    }
    
    /* Input styles */
    .input-file {
        border: 2px dashed #2563EB !important;
        border-radius: 0.75rem !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
        background-color: #F9FAFB !important;
    }
    
    .input-file:hover {
        border-color: #1D4ED8 !important;
        background-color: #EFF6FF !important;
    }
    
    .input-link {
        border-radius: 0.5rem !important;
        border: 1px solid #D1D5DB !important;
        transition: all 0.3s ease !important;
    }
    
    .input-link:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Button styles */
    .primary-button {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
    }
    
    .primary-button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-1px) !important;
    }
    
    .primary-button:active {
        transform: translateY(1px) !important;
    }
    
    .secondary-button {
        background-color: #F3F4F6 !important;
        color: #4B5563 !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
        border: 1px solid #E5E7EB !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
    }
    
    .secondary-button:hover {
        background-color: #E5E7EB !important;
        transform: translateY(-1px) !important;
    }
    
    .secondary-button:active {
        transform: translateY(1px) !important;
    }
    
    /* Progress bar styles */
    .progress-bar-wrap {
        border-radius: 0.5rem !important;
        overflow: hidden !important;
        height: 0.75rem !important;
        background-color: #E5E7EB !important;
    }
    
    .progress-bar {
        border-radius: 0.5rem !important;
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%) !important;
        height: 100% !important;
        transition: width 0.3s ease !important;
    }
    
    /* Status indicators */
    .status-success {
        color: #10B981 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .status-warning {
        color: #F59E0B !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .status-error {
        color: #EF4444 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* PDF preview */
    .pdf-preview {
        border-radius: 0.75rem !important;
        overflow: hidden !important;
        border: 1px solid #E5E7EB !important;
        transition: all 0.3s ease !important;
    }
    
    .pdf-preview:hover {
        border-color: #2563EB !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    .pdf-canvas canvas {
        width: 100% !important;
        border-radius: 0.5rem !important;
    }
    
    /* Dropdown and select styles */
    select, .gr-dropdown {
        border-radius: 0.5rem !important;
        border: 1px solid #D1D5DB !important;
        padding: 0.625rem 1rem !important;
        background-color: white !important;
        transition: all 0.3s ease !important;
    }
    
    select:focus, .gr-dropdown:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Accordion styles */
    .gr-accordion {
        border-radius: 0.5rem !important;
        overflow: hidden !important;
        border: 1px solid #E5E7EB !important;
        margin-top: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-accordion:hover {
        border-color: #D1D5DB !important;
    }
    
    .gr-accordion-header {
        background-color: #F9FAFB !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        color: #4B5563 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-accordion-header:hover {
        background-color: #F3F4F6 !important;
        color: #2563EB !important;
    }
    
    /* Footer styles */
    footer {
        visibility: hidden;
    }
    
    .app-footer {
        margin-top: 2rem;
        padding: 1.5rem 0;
        border-top: 1px solid #E5E7EB;
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
    }
    
    /* Utility classes */
    .secondary-text {
        color: #6B7280 !important;
        font-size: 0.875rem !important;
    }
    
    .env-warning {
        color: #F59E0B !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .env-success {
        color: #10B981 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* Tech details styling */
    .tech-details {
        margin-top: 1rem;
        font-size: 0.875rem;
    }
    
    .tech-details summary {
        cursor: pointer;
        color: #6B7280;
        font-weight: 500;
        padding: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .tech-details summary:hover {
        color: #2563EB;
    }
    
    .tech-details .details-content {
        padding: 0.75rem;
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .tech-details a {
        color: #2563EB;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .tech-details a:hover {
        text-decoration: underline;
    }
    
    /* Result file styling */
    .result-file {
        border: 1px solid #E5E7EB !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        background-color: #F9FAFB !important;
    }
    
    .result-file:hover {
        border-color: #2563EB !important;
        background-color: #EFF6FF !important;
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: #1F2937;
        color: white;
        text-align: center;
        border-radius: 0.5rem;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Help icon */
    .help-icon {
        color: #6B7280;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-left: 0.5rem;
    }
    
    .help-icon:hover {
        color: #2563EB;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .card {
            padding: 1rem;
        }
        
        .app-header h1 {
            font-size: 1.5rem !important;
        }
        
        .primary-button, .secondary-button {
            padding: 0.625rem 1rem !important;
            font-size: 0.875rem !important;
        }
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            opacity: 1;
        }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* File upload animation */
    .file-upload-animation {
        transition: all 0.5s ease;
    }
    
    .file-upload-animation.active {
        transform: scale(1.02);
        border-color: #2563EB !important;
        background-color: #EFF6FF !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F4F6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #9CA3AF;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6B7280;
    }
"""

# Add Font Awesome for icons
custom_head = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        // Add drag and drop highlight effect
        document.addEventListener('DOMContentLoaded', function() {
            const fileDropArea = document.querySelector('.input-file');
            if (fileDropArea) {
                ['dragenter', 'dragover'].forEach(eventName => {
                    fileDropArea.addEventListener(eventName, highlight, false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    fileDropArea.addEventListener(eventName, unhighlight, false);
                });
                
                function highlight(e) {
                    fileDropArea.classList.add('active');
                }
                
                function unhighlight(e) {
                    fileDropArea.classList.remove('active');
                }
            }
        });
    </script>
"""

# Add reCAPTCHA if needed
demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
<div class="tech-details">
    <details>
        <summary><i class="fas fa-info-circle"></i> Technical details</summary>
        <div class="details-content">
            <p><strong><i class="fab fa-github"></i> GitHub:</strong> <a href="https://github.com/AI-Ahmed/DRPDF" target="_blank">AI-Ahmed/DRPDF</a></p>
            <p><strong><i class="fas fa-file-pdf"></i> PDFMathTranslate:</strong> <a href="https://github.com/Byaidu/PDFMathTranslate" target="_blank">Byaidu/PDFMathTranslate</a></p>
            <p><strong><i class="fas fa-language"></i> BabelDOC:</strong> <a href="https://github.com/funstory-ai/BabelDOC" target="_blank">funstory-ai/BabelDOC</a></p>
            <p><strong><i class="fas fa-code"></i> GUI by:</strong> <a href="https://github.com/reycn" target="_blank">Rongxin</a></p>
            <p><strong><i class="fas fa-tag"></i> DRPDF Version:</strong> {__version__}</p>
            <p><strong><i class="fas fa-tag"></i> BabelDOC Version:</strong> {babeldoc_version}</p>
        </div>
    </details>
</div>
"""

# Help tooltips
tooltips = {
    "file_upload": "Upload a PDF document from your computer",
    "link_input": "Provide a URL to a PDF document online",
    "service": "Select the translation service to use",
    "lang_from": "Select the source language of your document",
    "lang_to": "Select the target language for translation",
    "page_range": "Select which pages to translate",
    "custom_page": "Specify custom page ranges (e.g., 1-5,8,11-13)",
    "threads": "Number of concurrent translation threads (higher = faster but more resource intensive)",
    "skip_fonts": "Skip font subsetting to improve performance (may affect text appearance)",
    "ignore_cache": "Force retranslation even if cached results exist",
    "babeldoc": "Use the BabelDOC engine for improved layout preservation",
    "custom_prompt": "Provide custom instructions for the translation model"
}

cancellation_event_map = {}

# The following code creates the enhanced GUI
with gr.Blocks(
    title="DrPDF - Professional PDF Translation Tool",
    theme=custom_theme,
    css=custom_css,
    head=(custom_head + (demo_recaptcha if flag_demo else "")),
) as demo:
    # App Header
    with gr.Row(elem_classes=["app-header"]):
        gr.HTML(
            """
            <div class="app-logo">
                <i class="fas fa-file-pdf fa-2x"></i>
                <h1>DrPDF - Professional PDF Translation Tool</h1>
            </div>
            """
        )

    # Main content area
    with gr.Row():
        # Left column - Input and Settings
        with gr.Column(scale=1):
            # File Input Card
            with gr.Group(elem_classes=["card"]):
                gr.HTML(
                    """
                    <div class="card-header">
                        <i class="fas fa-upload"></i> Document Input
                    </div>
                    """
                )
                
                file_type = gr.Radio(
                    choices=["File", "Link"],
                    label="Input Type",
                    value="File",
                    elem_classes=["input-type-selector"],
                    info="Choose to upload a file or provide a URL"
                )
                
                with gr.Row():
                    file_input = gr.File(
                        label="Upload PDF Document",
                        file_count="single",
                        file_types=[".pdf"],
                        type="filepath",
                        elem_classes=["input-file", "file-upload-animation"],
                    )

                link_input = gr.Textbox(
                    label="PDF URL",
                    placeholder="https://example.com/document.pdf",
                    visible=False,
                    interactive=True,
                    elem_classes=["input-link"],
                    info=tooltips["link_input"]
                )
            
            # Translation Settings Card
            with gr.Group(elem_classes=["card"]):
                gr.HTML(
                    """
                    <div class="card-header">
                        <i class="fas fa-cogs"></i> Translation Settings
                    </div>
                    """
                )
                
                service = gr.Dropdown(
                    label="Translation Service",
                    choices=enabled_services,
                    value=enabled_services[0],
                    elem_classes=["service-dropdown"],
                    info=tooltips["service"]
                )
                
                envs = []
                for i in range(3):
                    envs.append(
                        gr.Textbox(
                            visible=False,
                            interactive=True,
                        )
                    )
                
                with gr.Row():
                    lang_from = gr.Dropdown(
                        label="Source Language",
                        choices=lang_map.keys(),
                        value=ConfigManager.get("DRPDF_LANG_FROM", "English"),
                        info=tooltips["lang_from"]
                    )
                    lang_to = gr.Dropdown(
                        label="Target Language",
                        choices=lang_map.keys(),
                        value=ConfigManager.get("DRPDF_LANG_TO", "Simplified Chinese"),
                        info=tooltips["lang_to"]
                    )
                
                page_range = gr.Radio(
                    choices=page_map.keys(),
                    label="Page Selection",
                    value=list(page_map.keys())[0],
                    info=tooltips["page_range"]
                )
                
                page_input = gr.Textbox(
                    label="Custom Page Range",
                    placeholder="e.g., 1-5,8,11-13",
                    visible=False,
                    interactive=True,
                    info=tooltips["custom_page"]
                )
                
                # Advanced Options
                with gr.Accordion("Advanced Options", open=False, elem_classes=["gr-accordion"]):
                    gr.HTML(
                        """
                        <div style="font-weight: 600; margin-bottom: 0.75rem; color: #4B5563;">
                            <i class="fas fa-sliders-h"></i> Performance & Processing
                        </div>
                        """
                    )
                    
                    threads = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Processing Threads",
                        interactive=True,
                        info=tooltips["threads"]
                    )
                    
                    with gr.Row():
                        skip_subset_fonts = gr.Checkbox(
                            label="Skip Font Subsetting", 
                            interactive=True, 
                            value=False,
                            info=tooltips["skip_fonts"]
                        )
                        ignore_cache = gr.Checkbox(
                            label="Ignore Cache", 
                            interactive=True, 
                            value=False,
                            info=tooltips["ignore_cache"]
                        )
                    
                    gr.HTML(
                        """
                        <div style="font-weight: 600; margin: 1rem 0 0.75rem 0; color: #4B5563;">
                            <i class="fas fa-language"></i> Advanced Translation
                        </div>
                        """
                    )
                    
                    prompt = gr.Textbox(
                        label="Custom Prompt for LLM", 
                        placeholder="Enter custom instructions for the translation model...",
                        interactive=True, 
                        visible=False,
                        info=tooltips["custom_prompt"]
                    )
                    
                    use_babeldoc = gr.Checkbox(
                        label="Use BabelDOC Engine", 
                        interactive=True, 
                        value=False,
                        info=tooltips["babeldoc"]
                    )
                    
                    envs.append(prompt)
            
            # Action Buttons
            with gr.Row():
                translate_btn = gr.Button(
                    value="Translate Document",
                    variant="primary", 
                    elem_classes=["primary-button"],
                    elem_id="translate-button"
                )
                
                cancellation_btn = gr.Button(
                    value="Cancel",
                    variant="secondary", 
                    elem_classes=["secondary-button"],
                    elem_id="cancel-button"
                )
            
            # Status indicator (initially hidden)
            status_indicator = gr.HTML(
                """
                <div class="status-indicator" style="display: none; margin-top: 1rem;">
                    <div class="status-success">
                        <i class="fas fa-check-circle"></i> <span id="status-message">Ready</span>
                    </div>
                </div>
                """,
                visible=True
            )
            
            # Technical Details
            gr.HTML(tech_details_string, elem_classes=["secondary-text"])
            
            # Hidden elements for reCAPTCHA
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')

        # Right column - Preview and Results
        with gr.Column(scale=2):
            # Preview Card
            with gr.Group(elem_classes=["card"]):
                gr.HTML(
                    """
                    <div class="card-header">
                        <i class="fas fa-eye"></i> Document Preview
                    </div>
                    """
                )
                preview = PDF(label="", visible=True, height=800, elem_classes=["pdf-preview"])
            
            # Results Card
            with gr.Group(elem_classes=["card"]):
                output_title = gr.HTML(
                    """
                    <div class="card-header">
                        <i class="fas fa-check-circle"></i> Translation Results
                    </div>
                    """, 
                    visible=False
                )
                
                with gr.Row(visible=False) as output_row:
                    output_file_mono = gr.File(
                        label="Single Language Version", 
                        elem_classes=["result-file"],
                        file_count="single",
                        type="filepath"
                    )
                    output_file_dual = gr.File(
                        label="Dual Language Version", 
                        elem_classes=["result-file"],
                        file_count="single",
                        type="filepath"
                    )

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                // Update status indicator
                document.querySelector('.status-indicator').style.display = 'block';
                document.querySelector('.status-indicator div').className = 'status-success';
                document.getElementById('status-message').textContent = 'Document loaded successfully';
                return [a];
            }}
            """
            if flag_demo
            else """
            (a,b)=>{
                // Update status indicator
                document.querySelector('.status-indicator').style.display = 'block';
                document.querySelector('.status-indicator div').className = 'status-success';
                document.getElementById('status-message').textContent = 'Document loaded successfully';
                return [a];
            }
            """
        ),
    )

    def on_select_service(service, evt: gr.EventData):
        translator = service_map[service]
        _envs = []
        for i in range(4):
            _envs.append(gr.update(visible=False, value=""))
        for i, env in enumerate(translator.envs.items()):
            label = env[0]
            value = ConfigManager.get_env_by_translatername(
                translator, env[0], env[1]
            )
            visible = True
            if hidden_gradio_details:
                if (
                    "MODEL" not in str(label).upper()
                    and value
                    and hidden_gradio_details
                ):
                    visible = False
                # Hidden Keys From Gradio
                if "API_KEY" in label.upper():
                    value = "***"  # We use "***" Present Real API_KEY
            _envs[i] = gr.update(
                visible=visible,
                label=label,
                value=value,
            )
        _envs[-1] = gr.update(visible=translator.CustomPrompt)
        return _envs

    def on_select_filetype(file_type):
        return (
            gr.update(visible=file_type == "File"),
            gr.update(visible=file_type == "Link"),
        )

    def on_select_page(choice):
        if choice == "Others":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
            
    def update_output_visibility(mono_path, dual_path):
        if mono_path and dual_path:
            return gr.update(visible=True)
        return gr.update(visible=False)

    state = gr.State({"session_id": None})

    # Connect event handlers
    page_range.select(on_select_page, page_range, page_input)
    service.select(on_select_service, service, envs)
    file_type.select(
        on_select_filetype,
        file_type,
        [file_input, link_input],
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    # Translation process with enhanced status updates
    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            use_babeldoc,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
        js="""
        () => {
            // Update status indicator to show processing
            document.querySelector('.status-indicator').style.display = 'block';
            document.querySelector('.status-indicator div').className = 'status-warning';
            document.getElementById('status-message').textContent = 'Translation in progress...';
            
            // Disable translate button during processing
            document.getElementById('translate-button').disabled = true;
            document.getElementById('translate-button').classList.add('loading');
            
            // Return empty object for Gradio
            return {};
        }
        """
    ).then(
        lambda: gr.update(visible=True), 
        None, 
        output_row,
        js="""
        () => {
            // Update status indicator to show completion
            document.querySelector('.status-indicator').style.display = 'block';
            document.querySelector('.status-indicator div').className = 'status-success';
            document.getElementById('status-message').textContent = 'Translation completed successfully!';
            
            // Re-enable translate button
            document.getElementById('translate-button').disabled = false;
            document.getElementById('translate-button').classList.remove('loading');
            
            // Return empty object for Gradio
            return {};
        }
        """
    ).then(
        lambda: None, 
        js="()=>{grecaptcha.reset()}" if flag_demo else ""
    )

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
        js="""
        () => {
            // Update status indicator to show cancellation
            document.querySelector('.status-indicator').style.display = 'block';
            document.querySelector('.status-indicator div').className = 'status-error';
            document.getElementById('status-message').textContent = 'Translation cancelled';
            
            // Re-enable translate button
            document.getElementById('translate-button').disabled = false;
            document.getElementById('translate-button').classList.remove('loading');
            
            // Return empty object for Gradio
            return {};
        }
        """
    )

    # Add footer
    gr.HTML(
        """
        <div class="app-footer">
            <p>DrPDF - Professional PDF Translation Tool | &copy; 2025</p>
        </div>
        """,
        elem_classes=["secondary-text"]
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    user = lines[0].strip()
    passwd = lines[1].strip()
    return user, passwd


def setup_gui(
    username=None,
    password=None,
    auth_file=None,
    port=7860,
    server_name="0.0.0.0",
    share=False,
    inbrowser=False,
):
    """
    Setup and launch the GUI.
    """
    auth = None
    if auth_file:
        username, password = parse_user_passwd(auth_file)
    if username and password:
        auth = (username, password)

    demo.queue()
    demo.launch(
        auth=auth,
        server_name=server_name,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DrPDF GUI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the GUI on")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Server name to run the GUI on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share the GUI with a public URL"
    )
    parser.add_argument(
        "--inbrowser", action="store_true", help="Open the GUI in a browser"
    )
    parser.add_argument("--username", type=str, help="Username for authentication")
    parser.add_argument("--password", type=str, help="Password for authentication")
    parser.add_argument(
        "--auth-file", type=str, help="File containing username and password"
    )
    args = parser.parse_args()

    setup_gui(
        username=args.username,
        password=args.password,
        auth_file=args.auth_file,
        port=args.port,
        server_name=args.server_name,
        share=args.share,
        inbrowser=args.inbrowser,
    )