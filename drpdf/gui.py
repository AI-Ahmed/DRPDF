import asyncio
import os
import shutil
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T
from email.parser import Parser

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from drpdf import __version__
from drpdf.high_level import translate
from drpdf.doclayout import ModelInstance, OnnxModel as DrpdfOnnxModel
from drpdf.config import ConfigManager

# Initialize the model instance for document layout detection
try:
    if ModelInstance.value is None:
        ModelInstance.value = DrpdfOnnxModel.load_available()
        print("Document layout model initialized successfully")
except Exception as e:
    print(f"Failed to initialize document layout model: {e}")
    # Set a fallback or handle the error appropriately
    ModelInstance.value = None
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
client_key = None
server_key = None

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
    result = requests.post(recaptcha_url, data=data ).json()
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
        filename = None
        try:  # filename from header
            if content:
                parser = Parser()
                parsed = parser.parsestr(f'Content-Disposition: {content}')
                filename = parsed.get_param('filename')
        except Exception:
            pass
        
        # Fallback to URL basename if header parsing failed or returned None
        if not filename:
            filename = os.path.basename(url)
        
        # Ensure filename is not empty and has .pdf extension
        if not filename or filename == "/":
            filename = "document.pdf"
        else:
            # Remove any existing extension and add .pdf
            filename = os.path.splitext(filename)[0] + ".pdf"
        
        # Clean filename from any path separators
        filename = os.path.basename(filename)
        
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


def identify_file_or_link(*args, **kwargs):
    """
    Helper function to identify file or link inputs when arguments are broken
    Returns: (file_type, file_input, link_input)
    """
    # First, check if we have valid arguments
    file_type_val = None
    file_input_val = None
    link_input_val = None
    
    # Try to get values from args
    if args and len(args) > 0:
        file_type_val = args[0]
        if len(args) > 1:
            file_input_val = args[1]
        if len(args) > 2:
            link_input_val = args[2]
    
    # If any are None, try to get from UI components
    if file_type_val is None and 'file_type' in globals() and hasattr(file_type, 'value'):
        file_type_val = file_type.value
    
    if file_input_val is None and 'file_input' in globals() and hasattr(file_input, 'value'):
        file_input_val = file_input.value
        
    if link_input_val is None and 'link_input' in globals() and hasattr(link_input, 'value'):
        link_input_val = link_input.value
    
    # Finally, infer file_type if still None
    if file_type_val is None:
        if file_input_val and isinstance(file_input_val, str) and file_input_val.strip():
            file_type_val = "File"
        elif link_input_val and isinstance(link_input_val, str) and link_input_val.strip():
            file_type_val = "Link"
        else:
            file_type_val = "File"  # Default
    
    return file_type_val, file_input_val, link_input_val


def translate_file(
    file_type,
    file_input,  # This is actually file_path_state in the UI handler
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
    vfont,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=None,  # Changed back to None as default
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate (actually file_path_state in the UI)
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
        - progress: The progress bar (default is None)
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    # Debug the actual file_type received
    global current_file_type
    print(f"translate_file received file_type: {file_type}, current_file_type: {current_file_type}")

    # Explicitly set file_type to the current global value for consistency
    file_type = current_file_type
    print(f"Using file_type: {file_type}")
    
    # If arguments are corrupted, try to fix them
    if file_type is None or (file_type == "File" and (file_input is None or file_input == "")) or \
       (file_type == "Link" and (link_input is None or link_input == "")):
        print("WARNING: Detected corrupted arguments, attempting to fix")
        file_type, file_input, link_input = identify_file_or_link(file_type, file_input, link_input)
        print(f"After fixing: file_type={file_type}, file_input={file_input}, link_input={link_input}")

    # Import Gradio Progress here to ensure it's loaded
    try:
        import gradio as gr
        # Check if progress is None or not callable and create a new Progress object
        if progress is None or not callable(progress):
            # Create a dummy progress object if we're outside Gradio
            try:
                progress = gr.Progress()
            except Exception as e:
                print(f"Could not create Gradio Progress object: {e}")
                # Fallback to a simple function if Gradio Progress is not available
                progress = lambda value, desc=None: print(f"Progress: {value * 100:.0f}% - {desc}")
    except ImportError:
        # If gradio is not available, create a simple progress function
        progress = lambda value, desc=None: print(f"Progress: {value * 100:.0f}% - {desc}")

    session_id = uuid.uuid4()
    # Initialize state as a dictionary if it's None
    if state is None:
        state = {}
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")
    
    # Update progress to show starting
    try:
        progress(0, desc="Starting translation...")
    except Exception as e:
        print(f"Warning: Could not update progress: {e}")

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)

    # Debug what we received
    print(f"File type: {file_type}")
    print(f"File input: {file_input}")
    print(f"Link input: {link_input}")

    # Handle the case where file_type is None by examining inputs
    if file_type is None:
        print("WARNING: file_type is None, attempting to determine type from inputs")
        if file_input and (isinstance(file_input, str) and file_input.strip() != ""):
            print(f"Inferring file_type='File' based on file_input: {file_input}")
            file_type = "File"
        elif link_input and (isinstance(link_input, str) and link_input.strip() != ""):
            print(f"Inferring file_type='Link' based on link_input: {link_input}")
            file_type = "Link"
        else:
            print("Could not infer file_type, using default 'File'")
            file_type = "File"

    if file_type == "File":
        print(f"Processing file upload: {file_input}")
        if file_input is None or file_input == "":
            raise gr.Error("No input file provided")
        
        # Add more debug info about the file
        try:
            if not os.path.exists(file_input):
                print(f"File does not exist at path: {file_input}")
                raise gr.Error(f"File not found: {file_input}")
            else:
                print(f"File exists. Size: {os.path.getsize(file_input)} bytes")
                # Check if file is accessible
                with open(file_input, 'rb') as f:
                    # Just read a few bytes to confirm accessibility
                    first_bytes = f.read(10)
                    print(f"File is accessible. First few bytes: {first_bytes}")
                
            file_path = shutil.copy(file_input, output)
            print(f"Copied file to: {file_path}")
            try:
                progress(0.1, desc="File copied successfully")
            except Exception as e:
                print(f"Warning: Could not update progress: {e}")
        except Exception as e:
            print(f"Error handling file: {str(e)}")
            import traceback
            traceback.print_exc()
            raise gr.Error(f"Error processing file: {str(e)}")
    elif file_type == "Link":
        print(f"Processing link: {link_input}")
        if not link_input:
            raise gr.Error("No input link provided")
        try:
            try:
                progress(0.05, desc="Downloading file...")
            except Exception as e:
                print(f"Warning: Could not update progress: {e}")
            file_path = download_with_limit(
                link_input,
                output,
                5 * 1024 * 1024 if flag_demo else None,
            )
            print(f"Downloaded file to: {file_path}")
            try:
                progress(0.1, desc="File downloaded successfully")
            except Exception as e:
                print(f"Warning: Could not update progress: {e}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            raise gr.Error(f"Error downloading file: {e}")
    else:
        raise gr.Error(f"Unknown file type: {file_type}")

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"
    
    # Check if file_raw exists
    if not file_raw.exists():
        print(f"File does not exist: {file_raw}")
        raise gr.Error(f"File does not exist: {file_raw}")
    else:
        print(f"File ready for translation: {file_raw} (Size: {os.path.getsize(file_raw)} bytes)")
        try:
            progress(0.15, desc="Preparing for translation...")
        except Exception as e:
            print(f"Warning: Could not update progress: {e}")

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
    # Get the environment variable keys in the order they appear in the class
    env_keys = list(translator.envs.keys())

    # Map the GUI values to the correct environment variables
    # The GUI passes values in the order they were displayed, which should match env_keys order
    for i, env_key in enumerate(env_keys):
        if i < len(envs):
            value = envs[i]
            # Handle empty strings and None values
            if value == "" or value is None:
                # Use default value from class if available
                _envs[env_key] = translator.envs[env_key]
            else:
                _envs[env_key] = value
        else:
            # Use default value if not provided
            _envs[env_key] = translator.envs[env_key]
    
    # Handle masked API keys and validate required keys
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY"):
            if str(v) == "***":
                # Load real API key from config
                real_key = ConfigManager.get_env_by_translatername(translator, k, None)
                if real_key:
                    _envs[k] = real_key
                else:
                    # Fallback to environment variables
                    env_value = os.environ.get(k)
                    if env_value and env_value.strip():
                        _envs[k] = env_value
                    else:
                        error_msg = f"❌ API key required: {k} must be provided. Please enter your API key in the '{k}' field."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            elif not v or str(v).strip() == "":
                # Empty field - check environment variables as fallback
                env_value = os.environ.get(k)
                if env_value and env_value.strip():
                    _envs[k] = env_value
                else:
                    error_msg = f"❌ API key required: {k} must be provided. Please enter your API key in the '{k}' field."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            else:
                # Store the provided API key directly
                _envs[k] = v

    try:
        progress(0.2, desc="Starting translation process...")
    except Exception as e:
        print(f"Warning: Could not update progress: {e}")

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        # Calculate progress between 20% and 90% based on tqdm progress
        progress_value = 0.2 + (t.n / t.total) * 0.7 if t.total > 0 else 0.5
        try:
            progress(progress_value, desc=desc)
        except Exception as e: # pragma: no cover
            # Silently ignore progress update errors during translation
            pass

    try:
        threads = int(threads) if threads is not None else 4
    except ValueError:
        threads = 4

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
        "vfont": vfont,  # 添加自定义公式字体正则表达式
        "model": ModelInstance.value,
    }

    try:
        if use_babeldoc:
            return babeldoc_translate_file(**param)
        translate(**param)
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")

    logger.debug("Files in output directory after translation: %s", os.listdir(output))

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
    try:
        print("BabelDOC: Starting BabelDOC translation...")
        from babeldoc.high_level import init as babeldoc_init

        babeldoc_init()
        print("BabelDOC: Initialization complete")
        
        from babeldoc.high_level import async_translate as babeldoc_translate
        from babeldoc.translation_config import TranslationConfig as YadtConfig

        if kwargs["prompt"]:
            prompt = kwargs["prompt"]
        else:
            prompt = None
        
        print(f"BabelDOC: Service: {kwargs['service']}, Lang: {kwargs['lang_in']} -> {kwargs['lang_out']}")
    except Exception as e:
        print(f"BabelDOC: Initialization error: {e}")
        raise

    from drpdf.translator import (
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
    )

    print(f"BabelDOC: Looking for translator for service: {kwargs['service']}")
    
    for translator_class in [
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
        if kwargs["service"] == translator_class.name:
            print(f"BabelDOC: Found translator class: {translator_class.name}")
            try:
                translator = translator_class(
                    kwargs["lang_in"],
                    kwargs["lang_out"],
                    "",
                    envs=kwargs["envs"],
                    prompt=kwargs["prompt"],
                    ignore_cache=kwargs["ignore_cache"],
                )
                print(f"BabelDOC: Translator initialized successfully")
                break
            except Exception as e:
                print(f"BabelDOC: Error initializing translator: {e}")
                raise
    else:
        available_services = [cls.name for cls in [
            GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator,
            OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator, OpenAITranslator,
            ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator,
            AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator,
            ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator,
            OpenAIlikedTranslator, QwenMtTranslator
        ]]
        raise ValueError(f"Unsupported translation service '{kwargs['service']}'. Available: {available_services}")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        print(f"BabelDOC: Processing file: {file}")
        
        # Convert pages list to string format for BabelDOC
        pages_list = kwargs.get("pages", [])
        if pages_list:
            pages_str = ",".join(str(x + 1) for x in pages_list)  # BabelDOC uses 1-based indexing
            print(f"BabelDOC: Selected pages: {pages_str}")
        else:
            pages_str = ""  # Empty string means all pages
            print("BabelDOC: Processing all pages")
            
        try:
            yadt_config = YadtConfig(
                input_file=file,
                font=None,
                pages=pages_str,
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
            print("BabelDOC: Configuration created successfully")
        except Exception as e:
            print(f"BabelDOC: Error creating configuration: {e}")
            raise

        async def yadt_translate_coro(yadt_config):
            # Get progress callback from kwargs
            progress_callback = kwargs.get("callback")
            
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # Start translation
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    
                    # Update progress callback if available
                    if progress_callback and callable(progress_callback):
                        try:
                            progress_callback(progress_context)
                        except Exception as e:
                            print(f"Warning: Could not update progress callback: {e}")
                        
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        print("BabelDOC: Translation completed successfully")
                        
                        # Debug: Print available attributes
                        print(f"BabelDOC: Result object type: {type(result)}")
                        print(f"BabelDOC: Available attributes: {dir(result)}")
                        
                        # Try to access common attributes safely
                        try:
                            # Check for different possible attribute names
                            if hasattr(result, 'original_pdf_path'):
                                original_path = result.original_pdf_path
                            elif hasattr(result, 'input_pdf_path'):
                                original_path = result.input_pdf_path
                            elif hasattr(result, 'original_file'):
                                original_path = result.original_file
                            else:
                                original_path = "Unknown"
                            
                            if hasattr(result, 'total_seconds'):
                                time_cost = result.total_seconds
                            elif hasattr(result, 'time_cost'):
                                time_cost = result.time_cost
                            elif hasattr(result, 'duration'):
                                time_cost = result.duration
                            else:
                                time_cost = 0
                            
                            if hasattr(result, 'mono_pdf_path'):
                                file_mono = result.mono_pdf_path
                            elif hasattr(result, 'mono_file'):
                                file_mono = result.mono_file
                            elif hasattr(result, 'single_lang_path'):
                                file_mono = result.single_lang_path
                            else:
                                file_mono = None
                                
                            if hasattr(result, 'dual_pdf_path'):
                                file_dual = result.dual_pdf_path
                            elif hasattr(result, 'dual_file'):
                                file_dual = result.dual_file
                            elif hasattr(result, 'bilingual_path'):
                                file_dual = result.bilingual_path
                            else:
                                file_dual = None
                            
                            print(f"BabelDOC: Original PDF: {original_path}")
                            print(f"BabelDOC: Time Cost: {time_cost:.2f}s")
                            print(f"BabelDOC: Mono PDF: {file_mono or 'None'}")
                            print(f"BabelDOC: Dual PDF: {file_dual or 'None'}")
                            
                        except Exception as e:
                            print(f"BabelDOC: Error accessing result attributes: {e}")
                            # Fallback: try to find output files in the output directory
                            import os
                            output_dir = kwargs["output"]
                            pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
                            print(f"BabelDOC: Found PDF files in output: {pdf_files}")
                            
                            # Try to identify mono and dual files
                            file_mono = None
                            file_dual = None
                            for pdf_file in pdf_files:
                                full_path = os.path.join(output_dir, pdf_file)
                                if 'mono' in pdf_file.lower() or 'single' in pdf_file.lower():
                                    file_mono = full_path
                                elif 'dual' in pdf_file.lower() or 'bilingual' in pdf_file.lower():
                                    file_dual = full_path
                                elif file_mono is None:  # First file as fallback
                                    file_mono = full_path
                                elif file_dual is None:  # Second file as fallback
                                    file_dual = full_path

                            print(f"BabelDOC: Fallback - Mono: {file_mono}, Dual: {file_dual}")
               
                        break

            import gc
            gc.collect()
            
            # Ensure files exist before returning
            import os
            if file_mono and not os.path.exists(file_mono):
                print(f"BabelDOC: Warning - Mono file does not exist: {file_mono}")
                file_mono = None
            if file_dual and not os.path.exists(file_dual):
                print(f"BabelDOC: Warning - Dual file does not exist: {file_dual}")
                file_dual = None
            
            if not file_mono and not file_dual:
                print("BabelDOC: Error - No output files found!")
                raise Exception("BabelDOC translation completed but no output files were generated")
            
            print(f"BabelDOC: Returning files - Mono: {file_mono}, Dual: {file_dual}")
            
            return (
                str(file_mono) if file_mono else None,
                str(file_mono) if file_mono else None,
                str(file_dual) if file_dual else None,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
from babeldoc import __version__ as babeldoc_version

# Define a modern color palette
primary_color = "#2563EB"  # Blue
secondary_color = "#4682DDFF"  # Gray
success_color = "#10B981"  # Green
warning_color = "#F59E0B"  # Amber
error_color = "#EF4444"  # Red
background_color = "#5990C6FF"  # Light gray
card_color = "#5A98EDFF"  # White

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
        c200="#5E7DBCFF",
        c300="#D1D5DB",
        c400="#9CA3AF",
        c500=secondary_color,  # Secondary color
        c600="#C2C6CCFF",
        c700="#374151",
        c800="#1F2937",
        c900="#111827",
        c950="#030712",
    ),
    neutral_hue=gr.themes.Color(
        c50="#F9FAFB",
        c100="#F3F4F6",
        c200="#5E7DBCFF",
        c300="#D1D5DB",
        c400="#9CA3AF",
        c500="#6B7280",
        c600="#C2C6CCFF",
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
    /* Enhanced CSS with no borders or backgrounds */
    /* Global styles */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #000000FF;
    }

    /* ===== CONTAINER SPACING CONTROLS ===== */
    /* Target Gradio container elements to remove their background/borders */
    .gradio-container {
        max-width: 100% !important;
        /* SPACING: Controls the overall container width */
    }

    /* Target the blue backgrounds pointed by red arrows in screenshot */
    .gradio-row, .gradio-column, .gradio-group, .gradio-box, .gradio-accordion {
        border: none !important;
        border-radius: 0.75rem !important;
        background: transparent !important;
        box-shadow: none !important;
        /* SPACING: You can add margin/padding here to control space between all Gradio containers */
        /* Example: margin: 0.25rem !important; (decrease space between containers) */
        /* Example: padding: 0.5rem !important; (decrease internal padding) */
    }

    /* SPACING: Control space between rows */
    .gradio-row {
        /* SPACING: Decrease this to reduce vertical space between rows */
        margin-bottom: 0.5rem !important; 
        /* SPACING: Adjust this to control space between elements inside a row */
        gap: 0.5rem !important;
    }

    /* SPACING: Control space between columns */
    .gradio-column {
        /* SPACING: Decrease this to reduce horizontal space between columns */
        margin-right: 0.5rem !important;
        /* SPACING: Adjust this to control space between elements inside a column */
        gap: 0.5rem !important;
    }

    /* Remove background from specific containers */
    .contain {
        background: transparent !important;
        border: none !important;
        /* SPACING: Controls padding inside containers */
        /* Example: padding: 0.25rem !important; (decrease internal padding) */
    }

    /* Header styles */
    .app-header {
        /* SPACING: Controls top/bottom padding of the header */
        padding: 1.5rem 0;
        border-bottom: none !important;
        /* SPACING: Controls space below the header */
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    .app-header h1 {
        font-weight: 700 !important;
        font-size: 2rem !important;
        /* SPACING: Controls margin around header text */
        margin: 0 !important;
        /* SPACING: Controls padding around header text */
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
        /* SPACING: Controls space between logo and text */
        gap: 1rem;
    }

    .app-logo img {
        /* SPACING: Controls logo size */
        height: 3rem;
        width: auto;
    }

    /* Card styles */
    .card {
        background-color: transparent !important;
        border-radius: 0.75rem;
        box-shadow: none !important;
        /* SPACING: Controls external padding of cards */     /* HERE YOU CAN CONTROL THE SPACING BETWEEN CARDS */
        padding: 0.01rem;
        /* SPACING: Controls space below each card */
        margin-bottom: 1.5rem;
        border: none !important;
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: none !important;
    }

    .card-header {
        font-weight: 600;
        font-size: 1.25rem;
        /* SPACING: Controls space below card headers */
        margin-bottom: 1rem;
        color: #93C5FD; /* Lighter blue for title text */
        border-bottom: none !important;
        /* SPACING: Controls padding below card headers */
        padding-bottom: 0.75rem;
        display: flex;
        align-items: center;
        /* SPACING: Controls space between icon and text in headers */
        gap: 0.5rem;
    }

    .card-header i {
        color: #2563EB;
    }

    /* Translation Settings header - keep blue */
    h2.svelte-1gqy2d3, h3.svelte-1gqy2d3 {
        color: #93C5FD !important; /* Lighter blue for titles */
        /* SPACING: You can add margin/padding here to control space around section titles */
        /* Example: margin-bottom: 0.5rem !important; */
    }

    /* Section titles */
    .block.svelte-1gqy2d3 label span {
        color: #93C5FD !important; /* Lighter blue for section titles */
        /* SPACING: You can add margin/padding here to control space around labels */
        /* Example: margin-bottom: 0.25rem !important; */
    }

    /* Input styles */
    .input-file {
        border: 2px dashed #7096E7FF !important;
        border-radius: 0.75rem !important;
        /* SPACING: Controls padding inside file upload area */
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
        border: none !important;
        transition: all 0.3s ease !important;
        /* SPACING: You can add padding here to control input field size */
        /* Example: padding: 0.5rem !important; */
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
        /* SPACING: Controls padding inside buttons */
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        /* SPACING: Controls space between icon and text in buttons */
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
        background-color: #5881D2FF !important;
        color: #C2C6CCFF !important;
        font-weight: 600 !important;
        /* SPACING: Controls padding inside buttons */
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        /* SPACING: Controls space between icon and text in buttons */
        gap: 0.5rem !important;
    }

    .secondary-button:hover {
        background-color: #5E7DBCFF !important;
        transform: translateY(-1px) !important;
    }

    .secondary-button:active {
        transform: translateY(1px) !important;
    }

    /* Progress bar styles */
    .progress-bar-wrap {
        border-radius: 0.5rem !important;
        overflow: hidden !important;
        /* SPACING: Controls height of progress bar */
        height: 0.75rem !important;
        background-color: #5E7DBCFF !important;
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
        /* SPACING: Controls space between icon and text */
        gap: 0.5rem !important;
    }

    .status-warning {
        color: #F59E0B !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        /* SPACING: Controls space between icon and text */
        gap: 0.5rem !important;
    }

    .status-error {
        color: #EF4444 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        /* SPACING: Controls space between icon and text */
        gap: 0.5rem !important;
    }

    /* PDF preview */
    .pdf-preview {
        border-radius: 0.75rem !important;
        overflow: hidden !important;
        border: none !important;
        transition: all 0.3s ease !important;
        /* SPACING: You can add margin here to control space around PDF preview */
        /* Example: margin: 0.5rem !important; */
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
        border: none !important;
        /* SPACING: Controls padding inside dropdowns */
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
        border: none !important;
        /* SPACING: Controls space above accordions */
        margin-top: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .gr-accordion:hover {
        border-color: #D1D5DB !important;
    }

    .gr-accordion-header {
        background-color: #F9FAFB !important;
        /* SPACING: Controls padding inside accordion headers */
        padding: 0.75rem 1rem !important;
        font-weight: 600 !important;
        color: #93C5FD !important; /* Lighter blue for accordion header */
        transition: all 0.3s ease !important;
    }

    .gr-accordion-header:hover {
        background-color: #5881D2FF !important;
        color: #E5E8EFFF !important;
    }

    /* Footer styles */
    footer {
        visibility: hidden;
    }

    .app-footer {
        /* SPACING: Controls space above footer */
        margin-top: 2rem;
        /* SPACING: Controls padding inside footer */
        padding: 1.5rem 0;
        border-top: none !important;
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
        /* SPACING: Controls space between icon and text */
        gap: 0.5rem !important;
    }

    .env-success {
        color: #10B981 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        /* SPACING: Controls space between icon and text */
        gap: 0.5rem !important;
    }

    /* Tech details styling */
    .tech-details {
        /* SPACING: Controls space above tech details */
        margin-top: 1rem;
        font-size: 0.875rem;
    }

    .tech-details summary {
        cursor: pointer;
        color: #93C5FD; /* Lighter blue for tech details */
        font-weight: 500;
        /* SPACING: Controls padding around summary */
        padding: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .tech-details summary:hover {
        color: #2563EB;
    }

    .tech-details .details-content {
        /* SPACING: Controls padding inside details content */
        padding: 0.75rem;
        background-color: #25282BFF;
        border-radius: 0.5rem;
        /* SPACING: Controls space above details content */
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
        border: none !important;
        border-radius: 0.5rem !important;
        /* SPACING: Controls padding inside result files */
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
        /* SPACING: Controls padding inside tooltips */
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
        /* SPACING: Controls space to the left of help icon */
        margin-left: 0.5rem;
    }

    .help-icon:hover {
        color: #2563EB;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .card {
            /* SPACING: Controls padding inside cards on mobile */
            padding: 1rem;
        }
        
        .app-header h1 {
            /* SPACING: Controls font size of header on mobile */
            font-size: 1.5rem !important;
        }
        
        .primary-button, .secondary-button {
            /* SPACING: Controls padding inside buttons on mobile */
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
        /* SPACING: Controls scrollbar width */
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

    /* ===== ADDITIONAL SPACING CONTROLS ===== */
    /* To reduce space between boxes, add these rules: */

    /* Reduce space between all Gradio elements */
    .gradio-container .block {
        /* SPACING: Decrease this value to reduce space between elements */
        margin-bottom: 0.5rem !important;
        /* SPACING: Decrease this value to reduce internal padding */
        padding: 0.25rem !important;
    }

    /* Reduce space between form elements */
    .form, .form > *, .block-item {
        /* SPACING: Decrease this value to reduce space between form elements */
        margin-bottom: 0.5rem !important;
        /* SPACING: Decrease this value to reduce space between stacked elements */
        gap: 0.5rem !important;
    }

    /* Reduce space in flex containers */
    .flex {
        /* SPACING: Decrease this value to reduce space between flex items */
        gap: 0.5rem !important;
    }

    /* Reduce margins around all blocks */
    .block:not(.default), .gradio-box {
        /* SPACING: Decrease this value to reduce margins around blocks */
        margin: 0.25rem !important;
    }

    /* Configuration info box styling */
    .config-info {
        /* SPACING: Controls space around config info */
        margin: 0.75rem 0 !important;
    }

    .config-info div {
        background-color: #1F2937 !important;
        border-left: 4px solid #E6EAF3FF !important;
        padding: 0.75rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }

    .config-info div:hover {
        background-color: #DBEAFE !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1) !important;
    }
"""

# Add Font Awesome for icons
custom_head = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script>
        // Add drag and drop highlight effect
        document.addEventListener('DOMContentLoaded', function( ) {
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
        var onVerify = function(token ) {
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
    "service": "Select the translation service to use. Configuration fields will appear based on your selection.",
    "lang_from": "Select the source language of your document",
    "lang_to": "Select the target language for translation\n\n\n",
    "page_range": "Select which pages to translate",
    "custom_page": "Specify custom page ranges (e.g., 1-5,8,11-13 )",
    "threads": "Number of concurrent translation threads (higher = faster but more resource intensive)",
    "skip_fonts": "Skip font subsetting to improve performance (may affect text appearance)",
    "ignore_cache": "Force retranslation even if cached results exist",
    "babeldoc": "Use the BabelDOC engine for improved layout preservation",
    "custom_prompt": "Provide custom instructions for the translation model",
    "api_key": "Your API key for the selected service. Required fields are marked as (Required).",
    "base_url": "API endpoint URL. Default values are shown when available.",
    "model": "Model name to use. Default models are pre-configured for each service.",
    "config_auto": "Configuration fields automatically adapt based on the selected service. Required fields are clearly marked."
}

cancellation_event_map = {}

# Add global variables to track the selected file type and inputs
current_file_type = "File"  # Default to File mode
current_link_url = None
current_file_path = None

# Add a global variable to track the selected file type
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
                
                # Configuration info
                gr.HTML(
                    """
                    <div style="background-color: #1068DBFF; border-left: 4px solid #E6EAF3FF; padding: 0.75rem; margin: 0.5rem 0; border-radius: 0.5rem;">
                        <div style="font-weight: 600; color: #C8D5FEFF; margin-bottom: 0.25rem;">
                            <i class="fas fa-info-circle"></i> Smart Configuration
                        </div>
                        <div style="font-size: 0.875rem; color: #C8D5FEFF;">
                            Configuration fields will automatically appear based on your selected service. 
                            <strong>Required</strong> fields are clearly marked, and default values are pre-filled when available.
                            Services like Google and Bing require no configuration!
                        </div>
                    </div>
                    """,
                    elem_classes=["config-info"]
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
                        label="Target Language\n",
                        choices=lang_map.keys(),
                        value=ConfigManager.get("DRPDF_LANG_TO", "Arabic"),
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
                        <div style="font-weight: 600; margin-bottom: 0.75rem; color: #C2C6CCFF;">
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
                        <div style="font-weight: 600; margin: 1rem 0 0.75rem 0; color: #DDE4EEFF;">
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
                    
                    # Hidden vfont parameter for math formulas - default to empty
                    vfont = gr.Textbox(
                        label="Math Formulas Font Pattern",
                        placeholder="Regex pattern for formulas font detection",
                        interactive=True,
                        visible=False,
                        value=""
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

    # State variable to store the file path
    state = gr.State({"session_id": None})
    file_path_state = gr.State(None)  # Add this line to store the file path
    
    # Add hidden textbox to store link URL so it can be accessed by translate button
    link_url_state = gr.Textbox(visible=False, interactive=False)

    # Event handlers
    def on_select_service(service):
        """
        Simple service selection handler based on the original implementation.
        
        Parameters:
            service (str): The selected translation service name
            
        Returns:
            list: Updates for the environment variable inputs and prompt field
        """
        translator = service_map[service]
        _envs = []
        
        # Initialize all environment fields as hidden
        for i in range(4):  # We have 3 env fields + 1 prompt field
            _envs.append(gr.update(visible=False, value=""))
        
        # Configure each environment variable
        for i, env in enumerate(translator.envs.items()):
            if i >= 3:  # Only handle first 3 environment variables
                break
                
            label = env[0]  # Environment variable name
            default_value = env[1]  # Default value
            
            # Get saved value from config or use default
            value = ConfigManager.get_env_by_translatername(translator, env[0], env[1])
            visible = True
            
            # Handle API keys - show masked value if saved, empty if not
            if "API_KEY" in label.upper():
                if value and value != default_value:
                    value = "***"  # Show masked value for saved keys
                else:
                    value = ""  # Empty field for user to fill
                    
            _envs[i] = gr.update(
                visible=visible,
                label=f"{label} {'(Required)' if 'API_KEY' in label.upper() else '(Optional)'}",
                value=value,
                placeholder=f"Enter {label.lower()}..." if "API_KEY" in label.upper() else default_value or f"Enter {label.lower()}...",
                type="password" if "API_KEY" in label.upper() else "text",
                info=f"Configuration for {service} service"
            )
        
        # Handle custom prompt visibility
        _envs[-1] = gr.update(visible=hasattr(translator, 'CustomPrompt') and translator.CustomPrompt)
        
        return _envs

    def on_select_filetype(file_type):
        """Handle file type selection to toggle appropriate inputs"""
        print(f"File type changed to: {file_type}")
        if file_type == "File":
            return gr.update(visible=True), gr.update(visible=False)
        else:  # Link
            return gr.update(visible=False), gr.update(visible=True)

    def on_select_page(choice):
        if choice == "Others":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
            
    def update_output_visibility(mono_path, dual_path):
        if mono_path and dual_path:
            return gr.update(visible=True)
        return gr.update(visible=False)

    def on_page_range_change(page_range_value):
        return on_select_page(page_range_value)
        
    def on_service_change(service_value):
        return on_select_service(service_value)
        
    def on_file_upload(file_obj):
        """Handle file upload event"""
        global current_file_type, current_file_path
        print(f"File uploaded: {file_obj}")
        if file_obj is None:
            print("Warning: File object is None")
            current_file_path = None
            return None, None
        
        # Extract file path from different possible formats
        if isinstance(file_obj, str):
            path = file_obj
            print(f"File path from string: {path}")
        elif hasattr(file_obj, "name"):
            path = file_obj.name
            print(f"File path from name attribute: {path}")
        elif isinstance(file_obj, dict) and "name" in file_obj:
            path = file_obj["name"]
            print(f"File path from dictionary name key: {path}")
        else:
            try:
                # Try to get the first attribute that might be a path
                import inspect
                attrs = inspect.getmembers(file_obj)
                for attr_name, attr_val in attrs:
                    if attr_name.lower() in ["path", "filepath", "name", "filename"] and isinstance(attr_val, str):
                        path = attr_val
                        print(f"File path from attribute {attr_name}: {path}")
                        break
                else:
                    print(f"Could not extract path from file object: {file_obj}")
                    current_file_path = None
                    return None, None
            except Exception as e:
                print(f"Error extracting file path: {e}")
                current_file_path = None
                return None, None
        
        # Check if the file exists
        import os
        if not os.path.exists(path):
            print(f"Warning: File does not exist at path: {path}")
        else:
            print(f"File exists at path: {path}, size: {os.path.getsize(path)} bytes")
        
        current_file_path = path  # Store globally
        current_file_type = "File"  # Set mode to File
        print(f"Extracted file path: {path}, stored: {current_file_path}")
        # Store in state and use for preview
        return path, path

    def on_link_change(link_url):
        """Handle link input changes"""
        global current_file_type, current_link_url
        print(f"Link changed to: {link_url}")
        current_link_url = link_url  # Store globally
        
        # If a link is entered, make sure we're in Link mode
        if link_url and link_url.strip():
            current_file_type = "Link"
            print(f"Setting mode to Link because link was entered, stored: {current_link_url}")
        
        # Return the link to both preview (None) and the hidden state (link_url)
        return None, link_url

    # Add a helper function right after all the event handler functions
    def safe_get_value(component):
        """Safely get value from a Gradio component regardless of API version"""
        if hasattr(component, 'value'):
            return component.value
        elif hasattr(component, 'get_value'):
            try:
                return component.get_value()
            except:
                return None
        else:
            try:
                # Try to access the component directly
                return component
            except:
                return None

    # Update the safe_translate function to set a default for threads
    def safe_translate(*args, **kwargs):
        # Don't pass progress from here - let translate_file handle it
        if 'progress' in kwargs:
            logger.debug("Progress parameter detected in safe_translate kwargs; removing to avoid conflicts")
            del kwargs['progress']
        
        # Use global variables as the PRIMARY source of truth
        global current_file_type, current_link_url, current_file_path
        logger.debug("safe_translate invoked")
        logger.debug("  Global file_type: %s", current_file_type)
        logger.debug("  Global link_url is set: %s", bool(current_link_url))
        logger.debug("  Global file_path present: %s", bool(current_file_path))
        
        # Start with global variables
        args_list = [
            current_file_type,      # [0] file_type
            current_file_path,      # [1] file_path  
            current_link_url,       # [2] link_url
            None,                   # [3] service
            None,                   # [4] lang_from
            None,                   # [5] lang_to
            None,                   # [6] page_range
            None,                   # [7] page_input
            None,                   # [8] prompt
            None,                   # [9] threads
            None,                   # [10] skip_subset_fonts
            None,                   # [11] ignore_cache
            None,                   # [12] vfont
            None,                   # [13] use_babeldoc
            None,                   # [14] recaptcha_response
            None,                   # [15] state
        ]
        
        # Override with args if they are not None
        for i, arg in enumerate(args):
            if i < len(args_list) and arg is not None:
                args_list[i] = arg
                logger.debug("  Overriding args_list[%d] with provided arg", i)
        
        # Now fill in the missing values from UI components
        
        # Service
        if args_list[3] is None:
            service_value = safe_get_value(service)
            if service_value:
                args_list[3] = service_value
            else:
                args_list[3] = enabled_services[0]  # Default to first service
        
        # Languages
        if args_list[4] is None:
            lang_from_value = safe_get_value(lang_from)
            args_list[4] = lang_from_value if lang_from_value else ConfigManager.get("DRPDF_LANG_FROM", "English")
        
        if args_list[5] is None:
            lang_to_value = safe_get_value(lang_to)
            args_list[5] = lang_to_value if lang_to_value else ConfigManager.get("DRPDF_LANG_TO", "Arabic")
        
        # Page range
        if args_list[6] is None:
            page_range_value = safe_get_value(page_range)
            args_list[6] = page_range_value if page_range_value else "All"
        
        # Page input
        if args_list[7] is None:
            page_input_value = safe_get_value(page_input)
            args_list[7] = page_input_value if page_input_value else ""
        
        # Prompt
        if args_list[8] is None:
            prompt_value = safe_get_value(prompt)
            args_list[8] = prompt_value if prompt_value else ""
        
        # Threads
        if args_list[9] is None:
            threads_value = safe_get_value(threads)
            args_list[9] = threads_value if threads_value else 4
        
        # Boolean values
        if args_list[10] is None:
            skip_subset_fonts_value = safe_get_value(skip_subset_fonts)
            args_list[10] = skip_subset_fonts_value if skip_subset_fonts_value else False
        
        if args_list[11] is None:
            ignore_cache_value = safe_get_value(ignore_cache)
            args_list[11] = ignore_cache_value if ignore_cache_value else False
        
        if args_list[12] is None:
            vfont_value = safe_get_value(vfont)
            args_list[12] = vfont_value if vfont_value else ""
        
        if args_list[13] is None:
            use_babeldoc_value = safe_get_value(use_babeldoc)
            args_list[13] = use_babeldoc_value if use_babeldoc_value else False
        
        # State
        if args_list[15] is None:
            args_list[15] = {"session_id": None}
        
        # Add environment variables from remaining args
        # Don't limit by env_count - just add all environment arguments that were passed
        if len(args) > 16:
            env_args = args[16:]  # All environment arguments
            logger.debug("safe_translate processing %d environment arguments", len(env_args))
            
            # translate_file has an explicit 'progress' parameter after 'state' (index 16).
            # We need to reserve a slot for it (None by default) BEFORE appending env variables
            if len(args_list) < 17:
                # Ensure index 16 (progress) exists
                args_list.insert(16, None)

            # After inserting the placeholder, env variables should start at index 17
            env_start_index = 17
            
            # Extend args_list to include all environment arguments
            while len(args_list) < env_start_index + len(env_args):
                args_list.append("")
            
            # Add all environment variables from args in correct positions
            for i, env_arg in enumerate(env_args):
                args_list[env_start_index + i] = env_arg
            # --- FIX END ---
        else:
            logger.debug("No environment arguments received by safe_translate")
        
        args = tuple(args_list)
        
        try:
            logger.debug("safe_translate final parameters prepared for translate_file; threads=%s, page_range=%s", args[9], args[6])
            logger.debug("  File type: %s | Service: %s | Source lang: %s | Target lang: %s", args[0], args[3], args[4], args[5])
            
            # Validate inputs based on file type
            if args[0] == "File":
                if not args[1] or args[1] == "":
                    raise ValueError("No input file provided")
            elif args[0] == "Link":
                if not args[2] or args[2] == "":
                    raise ValueError("No input link provided")
            else:
                raise ValueError(f"Unknown file type: {args[0]}")
                
            return translate_file(*args, **kwargs)
        except Exception as e:
            logger.error("Translation error: %s", e)
            print(f"SAFE_TRANSLATE: Translation error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty values for outputs plus error message
            return None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    # Fix the file type detection by adding a new listener that updates the button behavior
    def on_file_type_change(selected_type):
        """Update the internal state when file type changes"""
        global current_file_type
        print(f"File type changed to: {selected_type}")
        current_file_type = selected_type  # Store the selected type globally
        
        # Add additional handling beyond just UI visibility
        if selected_type == "File":
            # Force hide link input and show file input
            return gr.update(visible=True), gr.update(visible=False)
        else:  # Link
            # Force hide file input and show link input
            return gr.update(visible=False), gr.update(visible=True)

    # Update the event binding's fallback method to include direct event listeners
    # Define a helper method to safely bind events
    def safe_bind(component, event_name, fn, inputs=None, outputs=None, **kwargs):
        try:
            if hasattr(component, event_name):
                method = getattr(component, event_name)
                if callable(method):
                    return method(fn=fn, inputs=inputs, outputs=outputs, **kwargs)
            
            # Try with "on_" prefix (used in some versions)
            on_event = f"on_{event_name}"
            if hasattr(component, on_event):
                method = getattr(component, on_event)
                if callable(method):
                    return method(fn=fn, inputs=inputs, outputs=outputs, **kwargs)
            
            # Try all methods that might be event binders
            for attr_name in dir(component):
                if attr_name.startswith("on_") or attr_name in [
                    "change", "select", "click", "submit", "upload", "input"
                ]:
                    try:
                        method = getattr(component, attr_name)
                        if callable(method):
                            print(f"Trying {component.__class__.__name__}.{attr_name}...")
                            return method(fn=fn, inputs=inputs, outputs=outputs, **kwargs)
                    except:
                        pass
                    
            # Try the gradio.py direct event listener approach
            # This accesses Gradio internals but might be necessary for some versions
            try:
                print(f"Trying direct event listener for {component.__class__.__name__}.{event_name}...")
                import inspect
                import gradio as gr
                
                # Get the gradio module/version info
                gradio_version = getattr(gr, "__version__", "unknown")
                print(f"Gradio version: {gradio_version}")
                
                # Add a listener directly to the component
                if hasattr(component, "change") and callable(component.change):
                    # Modern Gradio
                    return component.change(fn=fn, inputs=inputs, outputs=outputs, **kwargs)
                elif hasattr(component, "_id"):
                    # Try to use the older API to bind events
                    component_id = component._id
                    print(f"Component ID: {component_id}")
                    # Find a way to add a listener
                    # This varies by Gradio version, so try multiple approaches
                    return True
            except Exception as e:
                print(f"Direct event binding failed: {e}")
            
            print(f"WARNING: Could not bind event {event_name} for {component.__class__.__name__}")
            return None
        except Exception as e:
            print(f"Error binding {event_name} for {component.__class__.__name__}: {e}")
            return None

    # Add back debug_translate_button after safe_get_value function
    def debug_translate_button(*args, **kwargs):
        """Debug function to log the inputs before passing to safe_translate"""
        global current_file_type, current_link_url, current_file_path
        
        print("\n==== TRANSLATE BUTTON CLICKED ====")
        print(f"Global file_type: {current_file_type}")
        print(f"Global link_url: {current_link_url}")
        print(f"Global file_path: {current_file_path}")
        print(f"UI file_type: {safe_get_value(file_type)}")
        print(f"UI file_path_state: {safe_get_value(file_path_state)}")
        print(f"UI link_url_state: {args[2] if len(args) > 2 else 'No link_url_state'}")  # This should be the link_url_state
        
        # Check if we have any non-None values in args
        has_values = any(arg is not None for arg in args)
        print(f"Has non-None values in args: {has_values}")
        
        print(f"Args received: {args}")
        print(f"Args length: {len(args)}")
        print(f"First few args: {args[:3] if len(args) >= 3 else args}")
        
        # Debug the environment variables specifically
        print(f"\n==== ENVIRONMENT VARIABLES DEBUG ====")
        if len(args) >= 16:  # We expect at least 16 args before envs
            env_args = args[16:]  # Environment variables start at index 16
            print(f"Environment args received: {env_args}")
            print(f"Environment args length: {len(env_args)}")
            
            # Try to get the current service to understand expected mapping
            service_value = args[3] if len(args) > 3 else "Unknown"
            print(f"Service: {service_value}")
            
            if service_value in service_map:
                translator = service_map[service_value]
                env_keys = list(translator.envs.keys())
                print(f"Expected env keys order: {env_keys}")
                
                # Map each received value to its expected key
                for i, env_arg in enumerate(env_args):
                    if i < len(env_keys):
                        print(f"  env_args[{i}] = '{env_arg}' -> should be {env_keys[i]}")
                    else:
                        print(f"  env_args[{i}] = '{env_arg}' -> extra argument")
        else:
            print("Not enough arguments received to extract environment variables")
        
        print("============================\n")
        
        # Always pass to safe_translate
        return safe_translate(*args, **kwargs)

    # Setup event handlers using updated Gradio API
    def setup_event_handlers():
        """Set up event handlers using the correct Gradio API"""
        
        # Create a list to store all the event handlers we set up
        event_handlers = []
        
        # Page range event
        try:
            page_range.change(
                fn=on_select_page,
                inputs=page_range,
                outputs=page_input
            )
            event_handlers.append("page_range.change")
        except AttributeError:
            # Fallback for different Gradio versions
            try:
                page_range.select(
                    fn=on_select_page,
                    inputs=page_range,
                    outputs=page_input
                )
                event_handlers.append("page_range.select")
            except AttributeError:
                print("Warning: Could not bind page_range event")
        
        # Service selection event
        try:
            service.change(
                fn=on_select_service,
                inputs=service,
                outputs=envs
            )
            event_handlers.append("service.change")
        except AttributeError:
            try:
                service.select(
                    fn=on_select_service,
                    inputs=service,
                    outputs=envs
                )
                event_handlers.append("service.select")
            except AttributeError:
                print("Warning: Could not bind service event")
        
        # File type selection event
        try:
            file_type.change(
                fn=on_file_type_change,
                inputs=file_type,
                outputs=[file_input, link_input]
            )
            event_handlers.append("file_type.change")
        except AttributeError:
            try:
                file_type.select(
                    fn=on_file_type_change,
                    inputs=file_type,
                    outputs=[file_input, link_input]
                )
                event_handlers.append("file_type.select")
            except AttributeError:
                print("Warning: Could not bind file_type event")
        
        # File upload event
        try:
            file_input.upload(
                fn=on_file_upload,
                inputs=file_input,
                outputs=[preview, file_path_state]
            )
            event_handlers.append("file_input.upload")
        except AttributeError:
            try:
                file_input.change(
                    fn=on_file_upload,
                    inputs=file_input,
                    outputs=[preview, file_path_state]
                )
                event_handlers.append("file_input.change")
            except AttributeError:
                print("Warning: Could not bind file_input event")
        
        # Link input event
        try:
            link_input.change(
                fn=on_link_change,
                inputs=link_input,
                outputs=[preview, link_url_state]
            )
            event_handlers.append("link_input.change")
        except AttributeError:
            try:
                link_input.input(
                    fn=on_link_change,
                    inputs=link_input,
                    outputs=[preview, link_url_state]
                )
                event_handlers.append("link_input.input")
            except AttributeError:
                print("Warning: Could not bind link_input event")
        
        # Translate button event
        try:
            translate_btn.click(
                fn=safe_translate,
                inputs=[
                    file_type,
                    file_path_state,
                    link_url_state,
                    service,
                    lang_from,
                    lang_to,
                    page_range,
                    page_input,
                    prompt,
                    threads,
                    skip_subset_fonts,
                    ignore_cache,
                    vfont,
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
                ]
            ).then(
                fn=lambda: gr.update(visible=True),
                inputs=None,
                outputs=output_row
            )
            event_handlers.append("translate_btn.click")
        except AttributeError:
            print("Warning: Could not bind translate_btn event")
        
        # Cancel button event
        try:
            cancellation_btn.click(
                fn=stop_translate_file,
                inputs=state
            )
            event_handlers.append("cancellation_btn.click")
        except AttributeError:
            print("Warning: Could not bind cancellation_btn event")
        
        # Log all the event handlers we were able to set up
        print(f"Set up {len(event_handlers)} event handlers:")
        for name in event_handlers:
            print(f"  - {name}")

    # Replace the try-except block with a more robust approach
    # Replace the try/except block for event handlers with a clean call
    try:
        setup_event_handlers()
    except Exception as e:
        print(f"Error setting up event handlers: {e}")
        import traceback
        traceback.print_exc()
        
        # Use alternative binding approach that works with different Gradio versions
        print("Using alternative event binding method...")
        try:
            # Bind events using a more compatible approach
            # Page range event
            if hasattr(page_range, 'change'):
                page_range.change(on_select_page, inputs=page_range, outputs=page_input)
                print("✓ Bound page_range.change")
            elif hasattr(page_range, 'select'):
                page_range.select(on_select_page, inputs=page_range, outputs=page_input)
                print("✓ Bound page_range.select")
            
            # Service selection event
            if hasattr(service, 'change'):
                service.change(on_select_service, inputs=service, outputs=envs)
                print("✓ Bound service.change")
            elif hasattr(service, 'select'):
                service.select(on_select_service, inputs=service, outputs=envs)
                print("✓ Bound service.select")
            
            # File type selection event
            if hasattr(file_type, 'change'):
                file_type.change(on_file_type_change, inputs=file_type, outputs=[file_input, link_input])
                print("✓ Bound file_type.change")
            elif hasattr(file_type, 'select'):
                file_type.select(on_file_type_change, inputs=file_type, outputs=[file_input, link_input])
                print("✓ Bound file_type.select")
            
            # File upload event
            if hasattr(file_input, 'upload'):
                file_input.upload(on_file_upload, inputs=file_input, outputs=[preview, file_path_state])
                print("✓ Bound file_input.upload")
            elif hasattr(file_input, 'change'):
                file_input.change(on_file_upload, inputs=file_input, outputs=[preview, file_path_state])
                print("✓ Bound file_input.change")
            
            # Link input event
            if hasattr(link_input, 'change'):
                link_input.change(on_link_change, inputs=link_input, outputs=[preview, link_url_state])
                print("✓ Bound link_input.change")
            elif hasattr(link_input, 'input'):
                link_input.input(on_link_change, inputs=link_input, outputs=[preview, link_url_state])
                print("✓ Bound link_input.input")
            
            # Translate button event
            if hasattr(translate_btn, 'click'):
                translate_event = translate_btn.click(
                    safe_translate,
                    inputs=[file_type, file_path_state, link_url_state, service, lang_from, lang_to, 
                           page_range, page_input, prompt, threads, skip_subset_fonts, ignore_cache, 
                           vfont, use_babeldoc, recaptcha_response, state] + envs,
                    outputs=[output_file_mono, preview, output_file_dual, output_file_mono, output_file_dual, output_title]
                )
                # Chain the output visibility update
                if hasattr(translate_event, 'then'):
                    translate_event.then(lambda: gr.update(visible=True), inputs=None, outputs=output_row)
                print("✓ Bound translate_btn.click")
            
            # Cancel button event
            if hasattr(cancellation_btn, 'click'):
                cancellation_btn.click(stop_translate_file, inputs=state)
                print("✓ Bound cancellation_btn.click")
                
            print("Alternative binding method completed!")
        except Exception as e:
            print(f"Alternative binding also failed: {e}")
            traceback.print_exc()

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

# Add back the main block at the end of the file
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