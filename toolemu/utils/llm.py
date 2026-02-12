import logging
import os
from typing import Any, Callable, Dict, List, Optional, Mapping

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_anthropic import ChatAnthropic as LangchainChatAnthropic
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models.openai import convert_dict_to_message
from langchain_openai import OpenAI
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ChatMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from .my_typing import *
from langchain_community.chat_models.anthropic import convert_messages_to_prompt_anthropic
from peft import PeftModel

# --- Model Environment Manager ---
class ModelEnvManager:
    """
    Manages environment variables for different model types.

    API Priority for GPT models:
    1. OpenAI API (https://api.openai.com/v1) - Official OpenAI service
    2. OpenRouter (https://openrouter.ai/api/v1) - Third-party service providing access to various models

    For open-weight models, uses HuggingFace transformers for local inference.
    """
    _real_openai_api_key = None
    _real_openai_api_base = None
    _real_openrouter_api_key = None
    _real_openrouter_api_base = None
    _current_model = None

    @classmethod
    def set_env_for_model(cls, model_name: str):
        # GPT models should use OpenAI API if available, otherwise OpenRouter
        if model_name.lower().startswith("gpt"):
            # Check if we have OpenAI API credentials
            if os.environ.get("OPENAI_API_KEY"):
                pass  # OpenAI API key is already set
            elif os.environ.get("OPENROUTER_API_KEY"):
                # Fall back to OpenRouter
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
            else:
                raise ValueError("No OpenAI API key or OpenRouter API key found for GPT models")
            cls._current_model = model_name
            return
            
        # Save real API keys if not already saved
        if cls._real_openai_api_key is None:
            cls._real_openai_api_key = os.environ.get("OPENAI_API_KEY")
        if cls._real_openai_api_base is None:
            cls._real_openai_api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        if cls._real_openrouter_api_key is None:
            cls._real_openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if cls._real_openrouter_api_base is None:
            cls._real_openrouter_api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
            
        # If we're already set for this model, no need to change
        if cls._current_model == model_name:
            return

        # For open-weight models, no API keys needed (local inference)
        # For API models, restore real API keys (prefer OpenRouter over OpenAI)
        if not cls.is_open_weight_model(model_name):
            if cls._real_openrouter_api_key is not None:
                os.environ["OPENAI_API_BASE"] = cls._real_openrouter_api_base
                os.environ["OPENAI_API_KEY"] = cls._real_openrouter_api_key
            elif cls._real_openai_api_key is not None:
                os.environ["OPENAI_API_BASE"] = cls._real_openai_api_base
                os.environ["OPENAI_API_KEY"] = cls._real_openai_api_key
            else:
                os.environ.pop("OPENAI_API_BASE", None)
                os.environ.pop("OPENAI_API_KEY", None)

        cls._current_model = model_name

    @staticmethod
    def is_open_weight_model(model_name: str) -> bool:
        """Check if a model name indicates an open-weight HuggingFace model."""
        open_weight_keywords = ["qwen", "llama", "mistral"]
        return any(k in model_name.lower() for k in open_weight_keywords)

    @staticmethod
    def supports_stop_parameter(model_name: str) -> bool:
        """
        Check if a model supports the stop parameter.
        GPT models (gpt-*) do not support the stop parameter.
        """
        if model_name is None:
            return False
        # GPT models don't support stop parameter
        if model_name.lower().startswith("gpt"):
            return False
        # Open-weight models generally support stop parameter
        if ModelEnvManager.is_open_weight_model(model_name):
            return True
        # For other models (including OpenRouter models), assume they support it
        return True

# --- Model Loader Functions (retain names) ---

def llm_register_args(parser, prefix=None, shortprefix=None, defaults={}):
    model_name = defaults.get("model_name", "gpt-4o-mini")
    temperature = defaults.get("temperature", 0.0)
    max_tokens = defaults.get("max_tokens", None)
    default_retries = 8 if model_name.startswith("claude") else 12
    max_retries = defaults.get("max_retries", default_retries)
    request_timeout = defaults.get("request_timeout", 300)
    if prefix is None:
        prefix = ""
        shortprefix = ""
    else:
        prefix += "-"
        shortprefix = shortprefix or prefix[0]
    parser.add_argument(
        f"--{prefix}model-name",
        f"-{shortprefix}m",
        type=str,
        default=model_name,
        help=f"Model name (OpenAI, Anthropic, or open source)",
    )
    parser.add_argument(
        f"--{prefix}temperature",
        f"-{shortprefix}t",
        type=float,
        default=temperature,
        help="Temperature for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-tokens",
        f"-{shortprefix}mt",
        type=int,
        default=max_tokens,
        help="Max tokens for sampling",
    )
    parser.add_argument(
        f"--{prefix}max-retries",
        f"-{shortprefix}mr",
        type=int,
        default=max_retries,
        help="Max retries for each request",
    )
    parser.add_argument(
        f"--{prefix}request-timeout",
        f"-{shortprefix}rt",
        type=int,
        default=request_timeout,
        help="Timeout for each request",
    )

# --- Main Loader ---
def load_openai_llm(model_name: str = "gpt-4o-mini", quantization=None, **kwargs) -> BaseLanguageModel:
    # Set environment variables for the model (this handles GPT models properly)
    ModelEnvManager.set_env_for_model(model_name)

    # Check for HuggingFace models FIRST (before any prefix checks)
    # All HuggingFace model names contain "/" (e.g., "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B")
    if "/" in model_name:
        return _load_transformers_model(
            model_name,
            quantization=quantization,
            device_map="auto",
            torch_dtype=kwargs.get("torch_dtype", "bfloat16"),
            trust_remote_code=True,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
        )

    # Handle GPT models - they should use OpenAI API or OpenRouter
    if model_name.lower().startswith("gpt"):
        # Map short names to full API model strings
        gpt_model_mapping = {
            "gpt-4o": "chatgpt-4o-latest",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "gpt-5-nano": "gpt-5-nano-2025-08-07",
            "gpt-5-mini": "gpt-5-mini-2025-08-07",
            "gpt-5": "gpt-5-2025-08-07",
            "gpt-5.1": "gpt-5.1-2025-11-13",
        }
        api_model_name = gpt_model_mapping.get(model_name.lower(), model_name)
        print(f"[INFO] Loading GPT model: {model_name} -> {api_model_name}")
        print(f"[INFO]   temperature={kwargs.get('temperature', 'NOT SET')}")
        llm = ChatOpenAI(model_name=api_model_name, **kwargs)
        print(f"[INFO]   ChatOpenAI.temperature={llm.temperature}")
        llm.model_name = model_name  # Keep original name for display/filenames
        return llm

    # OpenAI (non-GPT models), Anthropic, and other API models
    if any(model_name.lower().startswith(prefix) for prefix in ["anthropic", "claude", "text-davinci", "o1"]):
        llm = ChatOpenAI(model_name=model_name, **kwargs)
        llm.model_name = model_name
        return llm

    # Unknown model - throw error
    raise ValueError(
        f"Unknown model type: '{model_name}'. "
        f"Expected either a HuggingFace model (with '/') or a known API model "
        f"(gpt-*, claude-*, anthropic/*, o1-*, etc.)"
    )

def _load_transformers_model(
    model_name: str,
    quantization: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 2048,
    **kwargs
) -> BaseLanguageModel:
    """Load model using HuggingFace transformers with optional bitsandbytes quantization."""
    from langchain_huggingface import HuggingFacePipeline
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        BitsAndBytesConfig
    )
    import torch
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from utils.model_name_utils import is_non_thinking_mode, strip_non_thinking_suffix

    # Check for non-thinking mode (Bnt suffix) and get actual HuggingFace model name
    non_thinking = is_non_thinking_mode(model_name)
    if non_thinking:
        hf_model_name = strip_non_thinking_suffix(model_name)
        print(f"[INFO] Non-thinking mode detected: {model_name} -> {hf_model_name}")
    else:
        hf_model_name = model_name

    print(f"[INFO] Loading HuggingFace model: {hf_model_name}")

    if quantization:
        print(f"[INFO] Using quantization: {quantization}")

    # Parse model name for LoRA adapter (format: "source_model+adapter_path")
    if "+" in hf_model_name:
        source_model_name, adapter_path = hf_model_name.split("+", 1)
        print(f"[INFO] Detected LoRA adapter:")
        print(f"[INFO]   Source model: {source_model_name}")
        print(f"[INFO]   Adapter path: {adapter_path}")
    else:
        source_model_name = hf_model_name
        adapter_path = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_name,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Monkey-patch tokenizer for non-thinking mode (Qwen3 with Bnt suffix)
    if non_thinking:
        original_apply_chat_template = tokenizer.apply_chat_template
        def patched_apply_chat_template(*args, **kwargs):
            kwargs['enable_thinking'] = False
            return original_apply_chat_template(*args, **kwargs)
        tokenizer.apply_chat_template = patched_apply_chat_template
        print(f"[INFO] Patched tokenizer.apply_chat_template with enable_thinking=False")

    # Check if this is a chat/instruct model with chat template
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    if has_chat_template:
        print(f"[INFO] Model has chat template - will be applied to prompts")

    # Build model loading kwargs
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }

    # Add quantization using proper BitsAndBytesConfig
    if quantization in ["int4", "nf4"]:
        print("[INFO] Using 4-bit quantization with BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    elif quantization == "int8":
        print("[INFO] Using 8-bit quantization with BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else:
        # No quantization - use specified dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.bfloat16)

    # Add device_map
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        print(f"[INFO] Using device_map='auto' with {torch.cuda.device_count()} GPU(s)")

    # Try to use Flash Attention 2 for memory efficiency and speed
    # Falls back gracefully if not available (e.g., older GPU, flash-attn not installed)
    print(f"[INFO] Loading model weights...")
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(source_model_name, **model_kwargs)
        print(f"[INFO] Using Flash Attention 2 for faster inference")
    except Exception as e:
        # Flash Attention 2 not available - fall back to default attention
        del model_kwargs["attn_implementation"]
        print(f"[INFO] Flash Attention 2 not available ({type(e).__name__}), using default attention")
        model = AutoModelForCausalLM.from_pretrained(source_model_name, **model_kwargs)

    # Load LoRA adapter if specified
    if adapter_path is not None:
        print(f"[INFO] Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[INFO] LoRA adapter loaded successfully")

    model.eval()
    print(f"[INFO] Model loaded successfully")

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_tokens or 2048,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,  # Only return generated text
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad token
    )

    # Store model name in the pipeline object for compatibility
    pipe.model_name = model_name

    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    # If this is a chat/instruct model, wrap in ChatHuggingFace to handle chat templates
    if has_chat_template:
        from langchain_huggingface import ChatHuggingFace
        # Pass tokenizer explicitly so ChatHuggingFace uses our (possibly patched) tokenizer
        # instead of loading a fresh one via AutoTokenizer.from_pretrained()
        chat_llm = ChatHuggingFace(llm=llm, tokenizer=tokenizer)
        # Store model_name on the wrapped object's pipeline for compatibility
        chat_llm._model_name = model_name
        print(f"[INFO] Wrapped in ChatHuggingFace for chat template support")
        return chat_llm

    # Store model_name on llm for compatibility
    llm._model_name = model_name
    return llm

# --- Args Loader ---
def load_openai_llm_with_args(args, prefix=None, fixed_version=True, **kwargs):
    if prefix is None:
        prefix = ""
    # Compose argument names
    def get_arg(name, default=None):
        arg_name = f"{prefix}_" + name if prefix else name
        return getattr(args, arg_name, default)
    model_name = get_arg("model_name", kwargs.get("model_name"))
    if not model_name:
        raise ValueError(f"Model name for '{prefix or 'model'}' must be specified via --{prefix + '-' if prefix else ''}model-name!")
    temperature = get_arg("temperature", kwargs.get("temperature", 0.0))
    request_timeout = get_arg("request_timeout", kwargs.get("request_timeout", 60))
    max_tokens = get_arg("max_tokens", kwargs.get("max_tokens", None))
    quantization = get_arg("quantization", None)

    return load_openai_llm(
        model_name=model_name,
        temperature=temperature,
        request_timeout=request_timeout,
        max_tokens=max_tokens,
        quantization=quantization,
        **kwargs,
    )

# --- Model Category Utility (keep for compatibility) ---
def get_model_category(llm: BaseLanguageModel):
    # Import here to avoid circular imports
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        if isinstance(llm, (ChatHuggingFace, HuggingFacePipeline)):
            return "huggingface"
    except ImportError:
        pass

    if isinstance(llm, ChatOpenAI):
        return "openai"
    elif isinstance(llm, ChatAnthropic):
        return "claude"
    elif hasattr(llm, '_llm_type') and llm._llm_type == "mock":
        return "openai"  # Treat mock LLMs as OpenAI for testing
    else:
        raise ValueError(f"Unknown model type: {type(llm)}")

# --- ChatOpenAI Patch (keep for compatibility) ---
class ChatOpenAI(LangchainChatOpenAI):
    def _create_chat_result(self, response, generation_info=None):
        # Handle both dict and object (ChatCompletion) responses
        if isinstance(response, dict):
            choices = response["choices"]
            usage = response.get("usage")
        else:
            choices = response.choices
            usage = getattr(response, "usage", None)
        generations = []
        for res in choices:
            if isinstance(res, dict):
                message_dict = res["message"]
                finish_reason = res.get("finish_reason")
            else:
                message_obj = res.message
                if hasattr(message_obj, "to_dict"):
                    message_dict = message_obj.to_dict()
                else:
                    message_dict = dict(message_obj.__dict__)
                finish_reason = getattr(res, "finish_reason", None)
            message = convert_dict_to_message(message_dict)
            gen = ChatGeneration(
                message=message,
                generation_info=dict(
                    finish_reason=finish_reason,
                ),
            )
            generations.append(gen)
        # Ensure usage is a dict for compatibility
        if usage is not None and not isinstance(usage, dict):
            if hasattr(usage, "to_dict"):
                usage = usage.to_dict()
            else:
                usage = dict(usage.__dict__)
        llm_output = {"token_usage": usage, "model_name": self.model_name}
        if generation_info is not None:
            llm_output["generation_info"] = generation_info
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        # Get the parent's default params
        params = super()._default_params
        
        # For GPT models, remove the stop parameter as it's not supported
        if hasattr(self, 'model_name') and self.model_name and self.model_name.lower().startswith("gpt"):
            if "stop" in params:
                del params["stop"]
        
        return params

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Get the request payload for the API call."""
        # For GPT models, ignore the stop parameter completely
        if hasattr(self, 'model_name') and self.model_name and self.model_name.lower().startswith("gpt"):
            stop = None
            # Also remove stop from kwargs if it exists
            kwargs.pop("stop", None)
        
        # Call the parent method with the filtered parameters
        return super()._get_request_payload(input_, stop=stop, **kwargs)

    def __init__(self, **kwargs):
        # For GPT models, remove any stop parameter from model_kwargs
        if kwargs.get('model_name', '').lower().startswith('gpt'):
            if 'model_kwargs' in kwargs and 'stop' in kwargs['model_kwargs']:
                del kwargs['model_kwargs']['stop']
            if 'stop' in kwargs:
                del kwargs['stop']
            if 'stop_sequences' in kwargs:
                del kwargs['stop_sequences']
        
        super().__init__(**kwargs)

# --- Anthropic Patch (keep for compatibility) ---
logger = logging.getLogger(__name__)

def _anthropic_create_retry_decorator(llm: ChatOpenAI) -> Callable[[Any], Any]:
    import anthropic
    min_seconds = 1
    max_seconds = 60
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(anthropic.APITimeoutError)
            | retry_if_exception_type(anthropic.APIError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.RateLimitError)
            | retry_if_exception_type(anthropic.APIConnectionError)
            | retry_if_exception_type(anthropic.APIStatusError)
            | retry_if_exception_type(anthropic.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

class ChatAnthropic(LangchainChatAnthropic):
    max_retries: int = 6
    max_tokens_to_sample: int = 4000
    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        finish_reason = response.stop_reason
        if finish_reason == "max_tokens":
            finish_reason = "length"
        generations = [
            ChatGeneration(
                message=AIMessage(content=response.completion),
                generation_info=dict(
                    finish_reason=finish_reason,
                ),
            )
        ]
        llm_output = {"model_name": response.model}
        return ChatResult(generations=generations, llm_output=llm_output)

def parse_llm_response(
    res: ChatResult, index: int = 0, one_generation_only: bool = True
) -> str:
    res = res.generations[0]
    if one_generation_only:
        assert len(res) == 1, res
    res = res[index]

    # Handle case where generation_info might be None
    if hasattr(res, 'generation_info') and res.generation_info is not None:
        if res.generation_info.get("finish_reason", None) == "length":
            raise ValueError(f"Discard a response due to length: {res.text}")

    return res.text.strip()
