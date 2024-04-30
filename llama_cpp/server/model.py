from __future__ import annotations

import json

from typing import Dict, Optional, Union, List
from enum import Enum

import llama_cpp
import llama_cpp.llama_speculative as llama_speculative
import llama_cpp.llama_tokenizer as llama_tokenizer

from llama_cpp.server.settings import ModelSettings

class LlamaModelBundle:

    def __init__(self, settings : ModelSettings, alias : str) -> None:
        self._settings : ModelSettings = settings
        self._model_alias : str = alias
        
        self._model : Optional[llama_cpp.Llama] = None
        self._is_swapped_in : bool = False

    def __del__(self) -> None:
        if self._model is not None:
            del self._model

    def is_swapped_in(self) -> bool:
        return self._is_swapped_in
    
    def is_loaded(self) -> bool:
        return (self._model is not None)

    def load(self) -> None:
        if self._model is None:
            self._model = self.load_llama_from_model_settings(self._settings)
            self._is_swapped_in = True

    def unload(self) -> None:
        self._model = None
        self._is_swapped_in = False

    def swap_backend_buffer_out(self) -> None:
        if self._model is not None:
            self._model.backend_swap_out()
            self._is_swapped_in = False

    def swap_backend_buffer_in(self) -> None:
        if self._model is not None:
            self._model.backend_swap_in()
            self._is_swapped_in = True

    def model(self) -> llama_cpp.Llama:
        return self._model
    
    def settings(self, settings : ModelSettings = None) -> Optional[ModelSettings]:
        if settings is not None:
            self._settings = settings
        return self._settings
    
    @staticmethod
    def load_llama_from_model_settings(settings: ModelSettings) -> llama_cpp.Llama:
        chat_handler = None
        if settings.chat_format == "llava-1-5":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.Llava15ChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "obsidian":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.ObsidianChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.ObsidianChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "llava-1-6":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.Llava16ChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.Llava16ChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "moondream":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.MoondreamChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.MoondreamChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "nanollava":
            assert settings.clip_model_path is not None, "clip model not found"
            if settings.hf_model_repo_id is not None:
                chat_handler = (
                    llama_cpp.llama_chat_format.NanoLlavaChatHandler.from_pretrained(
                        repo_id=settings.hf_model_repo_id,
                        filename=settings.clip_model_path,
                        verbose=settings.verbose,
                    )
                )
            else:
                chat_handler = llama_cpp.llama_chat_format.NanoLlavaChatHandler(
                    clip_model_path=settings.clip_model_path, verbose=settings.verbose
                )
        elif settings.chat_format == "hf-autotokenizer":
            assert (
                settings.hf_pretrained_model_name_or_path is not None
            ), "hf_pretrained_model_name_or_path must be set for hf-autotokenizer"
            chat_handler = (
                llama_cpp.llama_chat_format.hf_autotokenizer_to_chat_completion_handler(
                    settings.hf_pretrained_model_name_or_path
                )
            )
        elif settings.chat_format == "hf-tokenizer-config":
            assert (
                settings.hf_tokenizer_config_path is not None
            ), "hf_tokenizer_config_path must be set for hf-tokenizer-config"
            chat_handler = llama_cpp.llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(
                json.load(open(settings.hf_tokenizer_config_path))
            )

        tokenizer: Optional[llama_cpp.BaseLlamaTokenizer] = None
        if settings.hf_pretrained_model_name_or_path is not None:
            tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                settings.hf_pretrained_model_name_or_path
            )

        draft_model = None
        if settings.draft_model is not None:
            draft_model = llama_speculative.LlamaPromptLookupDecoding(
                num_pred_tokens=settings.draft_model_num_pred_tokens
            )

        kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None
        if settings.kv_overrides is not None:
            assert isinstance(settings.kv_overrides, list)
            kv_overrides = {}
            for kv in settings.kv_overrides:
                key, value = kv.split("=")
                if ":" in value:
                    value_type, value = value.split(":")
                    if value_type == "bool":
                        kv_overrides[key] = value.lower() in ["true", "1"]
                    elif value_type == "int":
                        kv_overrides[key] = int(value)
                    elif value_type == "float":
                        kv_overrides[key] = float(value)
                    else:
                        raise ValueError(f"Unknown value type {value_type}")

        import functools

        kwargs = {}

        if settings.hf_model_repo_id is not None:
            create_fn = functools.partial(
                llama_cpp.Llama.from_pretrained,
                repo_id=settings.hf_model_repo_id,
                filename=settings.model,
            )
        else:
            create_fn = llama_cpp.Llama
            kwargs["model_path"] = settings.model

        _model = create_fn(
            **kwargs,
            # Model Params
            n_gpu_layers=settings.n_gpu_layers,
            main_gpu=settings.main_gpu,
            tensor_split=settings.tensor_split,
            vocab_only=settings.vocab_only,
            use_mmap=settings.use_mmap,
            use_mlock=settings.use_mlock,
            kv_overrides=kv_overrides,
            # Context Params
            seed=settings.seed,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_threads=settings.n_threads,
            n_threads_batch=settings.n_threads_batch,
            rope_scaling_type=settings.rope_scaling_type,
            rope_freq_base=settings.rope_freq_base,
            rope_freq_scale=settings.rope_freq_scale,
            yarn_ext_factor=settings.yarn_ext_factor,
            yarn_attn_factor=settings.yarn_attn_factor,
            yarn_beta_fast=settings.yarn_beta_fast,
            yarn_beta_slow=settings.yarn_beta_slow,
            yarn_orig_ctx=settings.yarn_orig_ctx,
            mul_mat_q=settings.mul_mat_q,
            logits_all=settings.logits_all,
            embedding=settings.embedding,
            offload_kqv=settings.offload_kqv,
            # Sampling Params
            last_n_tokens_size=settings.last_n_tokens_size,
            # LoRA Params
            lora_base=settings.lora_base,
            lora_path=settings.lora_path,
            # Backend Params
            numa=settings.numa,
            # Chat Format Params
            chat_format=settings.chat_format,
            chat_handler=chat_handler,
            # Speculative Decoding
            draft_model=draft_model,
            # KV Cache Quantization
            type_k=settings.type_k,
            type_v=settings.type_v,
            # Tokenizer
            tokenizer=tokenizer,
            # Misc
            verbose=settings.verbose,
        )
        if settings.cache:
            if settings.cache_type == "disk":
                if settings.verbose:
                    print(f"Using disk cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                if settings.verbose:
                    print(f"Using ram cache with size {settings.cache_size}")
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
            _model.set_cache(cache)
        return _model
    
class LlamaModelSwapStrategy(str, Enum):
    LLAMA_MODEL_STRATEGY_SWAP_MODELS = "LLAMA_MODEL_STRATEGY_SWAP_MODELS"
    LLAMA_MODEL_STRATEGY_SWAP_BACKEND_WEIGHTS = "LLAMA_MODEL_STRATEGY_SWAP_BACKEND_WEIGHTS"

class LlamaProxy:
    def __init__(self, models: List[ModelSettings], 
                 swap_strategy : LlamaModelSwapStrategy = 
                 LlamaModelSwapStrategy.LLAMA_MODEL_STRATEGY_SWAP_MODELS) -> None:
        assert len(models) > 0, "No models provided!"

        self._models = {}
        self._swap_strategy : LlamaModelSwapStrategy = swap_strategy

        self._default_model_alias : str = None
        for model in models:
            if not model.model_alias:
                model.model_alias = model.model
            self._models[model.model_alias] = LlamaModelBundle(model, model.model_alias)

            if self._default_model_alias is None:
                self._default_model_alias = model.model_alias

        self._current_model = self._default_model_alias

        # if models are not all loaded ahead-of-time, we default to the "old"
        # behaviour of loading the first model
        # from there on out, we'll assume that the current model always has
        # their weights present on the GPU
        if swap_strategy == \
            LlamaModelSwapStrategy.LLAMA_MODEL_STRATEGY_SWAP_BACKEND_WEIGHTS:
            for model_alias, model in self._models.items():
                model.load()
                model.swap_backend_buffer_out()

            # only swap in the current model
            self._models[self._current_model].swap_backend_buffer_in()
        elif swap_strategy == \
            LlamaModelSwapStrategy.LLAMA_MODEL_STRATEGY_SWAP_MODELS:
            self._models[self._current_model].load()
        else:
            raise Exception("Unknown model swapping strategy.")

    def __call__(self, model: Optional[str] = None) -> llama_cpp.Llama:

        if model is None:
            model = self._default_model_alias

        if model not in self._models:
            model = self._default_model_alias

        if self._swap_strategy == \
            LlamaModelSwapStrategy.LLAMA_MODEL_STRATEGY_SWAP_BACKEND_WEIGHTS:

            if not self._models[model].is_swapped_in():
                    
                # swap weights of current model out of the device
                if self._models[self._current_model].is_swapped_in():
                    self._models[self._current_model].swap_backend_buffer_out()

                # swap in weights of new model
                self._models[model].swap_backend_buffer_in()

        elif self._swap_strategy == \
            LlamaModelSwapStrategy.LLAMA_MODEL_STRATEGY_SWAP_MODELS:

            if not self._models[model].is_loaded():
            
                # remove model from RAM
                if self._models[self._current_model].is_loaded():
                    self._models[self._current_model].unload()

                # load new model into RAM
                self._models[model].load()

        else:
            raise Exception("Unknown model swapping strategy.")
        
        self._current_model = model
        return self._models[self._current_model].model()

    def __getitem__(self, model: str):
        return self._models[model].settings().model_dump()

    def __setitem__(self, model: str, settings: Union[ModelSettings, str, bytes]):
        if isinstance(settings, (bytes, str)):
            settings = ModelSettings.model_validate_json(settings)
        self._models[model].settings(settings)

    def __iter__(self):
        for model in self._models:
            yield model

    def free(self):
        for model_alias in self._models:
            del self._models[model_alias]
        
        self._models = {}
