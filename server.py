import argparse
import io
import logging
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from qwen_tts import Qwen3TTSModel


app = FastAPI(title="Qwen3-TTS API")
logger = logging.getLogger("qwen3_tts_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_DESIGN_CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEVICE = "cuda:0"
DTYPE = "bfloat16"
FLASH_ATTN = False
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MODE = "both"

model_lock = threading.Lock()
model: Qwen3TTSModel | None = None
model_kind: str | None = None
model_checkpoint: str | None = None
model_load_args: dict[str, Any] = {}
generation_defaults: dict[str, Any] = {}
voice_design_model: Qwen3TTSModel | None = None
voice_design_checkpoint: str | None = None
voice_design_load_args: dict[str, Any] = {}
voice_design_generation_defaults: dict[str, Any] = {}
server_mode: str = DEFAULT_MODE


def _mode_supports_voice_clone() -> bool:
    return server_mode in ("voice_clone", "both")


def _mode_supports_voice_design() -> bool:
    return server_mode in ("voice_design", "both")


def _dtype_from_str(s: str) -> torch.dtype:
    normalized = (s or "").strip().lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _detect_model_kind(tts: Qwen3TTSModel) -> str:
    detected = getattr(tts.model, "tts_model_type", None)
    if detected in ("base", "custom_voice", "voice_design"):
        return detected
    raise ValueError(f"Unknown Qwen-TTS model type: {detected}")


def _collect_gen_kwargs(
    max_new_tokens: int | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    subtalker_top_k: int | None,
    subtalker_top_p: float | None,
    subtalker_temperature: float | None,
) -> dict[str, Any]:
    mapping = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "subtalker_top_k": subtalker_top_k,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_temperature": subtalker_temperature,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _normalize_language(language: str | None) -> str:
    value = (language or "Auto").strip()
    if not value:
        raise HTTPException(status_code=400, detail="`language` must be non-empty when provided.")
    return value


def _ensure_model() -> Qwen3TTSModel:
    if model is None:
        raise RuntimeError("Model has not been initialized. Start the server via `uv run python server.py`.")
    return model


def _ensure_voice_design_model() -> Qwen3TTSModel:
    if not _mode_supports_voice_design():
        raise HTTPException(status_code=503, detail="Voice design mode is disabled on this server.")
    if model_kind == "voice_design":
        return _ensure_model()
    if voice_design_model is None:
        raise RuntimeError(
            "VoiceDesign model has not been initialized. Start the server with a voice design checkpoint enabled."
        )
    return voice_design_model


def _wav_to_pcm16le_bytes(wav: np.ndarray) -> bytes:
    pcm = np.asarray(wav, dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    return pcm_i16.tobytes()


def _wav_to_wav_bytes(wav: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.asarray(wav, dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)

    buffer = io.BytesIO()
    sf.write(buffer, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _error_response(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def _load_single_model(checkpoint: str, device: str, dtype: str, flash_attn: bool) -> tuple[Qwen3TTSModel, str, dict[str, Any], dict[str, Any]]:
    torch_dtype = _dtype_from_str(dtype)
    attn_implementation = "flash_attention_2" if flash_attn else None

    tts = Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    load_args = {
        "device": device,
        "dtype": dtype,
        "flash_attn": flash_attn,
    }
    defaults = dict(getattr(tts, "generate_defaults", {}) or {})
    return tts, _detect_model_kind(tts), load_args, defaults


def _load_model(checkpoint: str, voice_design_ckpt: str | None, mode: str, device: str, dtype: str, flash_attn: bool) -> None:
    global model, model_kind, model_checkpoint, model_load_args, generation_defaults, server_mode
    global voice_design_model, voice_design_checkpoint, voice_design_load_args, voice_design_generation_defaults

    server_mode = mode

    model = None
    model_kind = None
    model_checkpoint = None
    model_load_args = {}
    generation_defaults = {}
    voice_design_model = None
    voice_design_checkpoint = None
    voice_design_load_args = {}
    voice_design_generation_defaults = {}

    primary_checkpoint = checkpoint
    if mode == "voice_design":
        primary_checkpoint = voice_design_ckpt or VOICE_DESIGN_CHECKPOINT

    tts, detected_kind, load_args, defaults = _load_single_model(
        checkpoint=primary_checkpoint,
        device=device,
        dtype=dtype,
        flash_attn=flash_attn,
    )
    model = tts
    model_kind = detected_kind
    model_checkpoint = primary_checkpoint
    model_load_args = load_args
    generation_defaults = defaults

    if mode == "both" and voice_design_ckpt and model_kind != "voice_design":
        design_tts, design_kind, design_load_args, design_defaults = _load_single_model(
            checkpoint=voice_design_ckpt,
            device=device,
            dtype=dtype,
            flash_attn=flash_attn,
        )
        if design_kind != "voice_design":
            raise ValueError(
                f"Configured voice design checkpoint must load a voice_design model, got {design_kind}: {voice_design_ckpt}"
            )
        voice_design_model = design_tts
        voice_design_checkpoint = voice_design_ckpt
        voice_design_load_args = design_load_args
        voice_design_generation_defaults = design_defaults


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "checkpoint": model_checkpoint,
            "model_kind": model_kind,
            "mode": server_mode,
            "voice_design_checkpoint": voice_design_checkpoint,
            "voice_design_enabled": _mode_supports_voice_design() and (model_kind == "voice_design" or voice_design_model is not None),
            "voice_clone_enabled": _mode_supports_voice_clone(),
        }
    )


@app.get("/model")
def get_model_info() -> JSONResponse:
    tts = _ensure_model()
    supported_languages = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_languages = tts.model.get_supported_languages()

    supported_speakers = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_speakers = tts.model.get_supported_speakers()

    return JSONResponse(
        {
            "checkpoint": model_checkpoint,
            "model_kind": model_kind,
            "mode": server_mode,
            "load_args": model_load_args,
            "generation_defaults": generation_defaults,
            "supported_languages": supported_languages,
            "supported_speakers": supported_speakers,
            "voice_design": {
                "enabled": _mode_supports_voice_design() and (model_kind == "voice_design" or voice_design_model is not None),
                "checkpoint": model_checkpoint if model_kind == "voice_design" else voice_design_checkpoint,
                "load_args": model_load_args if model_kind == "voice_design" else voice_design_load_args,
                "generation_defaults": generation_defaults if model_kind == "voice_design" else voice_design_generation_defaults,
            },
            "voice_clone": {
                "enabled": _mode_supports_voice_clone(),
                "checkpoint": model_checkpoint if model_kind != "voice_design" else None,
            },
        }
    )


@app.get("/config")
def get_config() -> JSONResponse:
    return get_model_info()


@app.post("/tts")
def tts(
    text: str = Form(...),
    language: str = Form("Auto"),
    speaker: str | None = Form(default=None),
    instruct: str | None = Form(default=None),
    ref_text: str | None = Form(default=None),
    reference_text: str | None = Form(default=None),
    reference_wav: UploadFile | None = File(default=None),
    max_new_tokens: int | None = Form(default=None),
    temperature: float | None = Form(default=None),
    top_k: int | None = Form(default=None),
    top_p: float | None = Form(default=None),
    repetition_penalty: float | None = Form(default=None),
    subtalker_top_k: int | None = Form(default=None),
    subtalker_top_p: float | None = Form(default=None),
    subtalker_temperature: float | None = Form(default=None),
) -> Response:
    tts_model = _ensure_model()

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must be non-empty.")

    normalized_language = _normalize_language(language)
    normalized_instruct = (instruct or "").strip() or None
    normalized_ref_text = (ref_text or reference_text or "").strip() or None
    x_vector_only_mode = normalized_ref_text is None
    if reference_wav is not None and normalized_instruct is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide either `reference_wav` for voice cloning or `instruct` for voice design, not both.",
        )

    gen_kwargs = _collect_gen_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
    )

    tmp_reference_path: str | None = None
    logger.info(
        "Received /tts request model_kind=%s text_length=%s language=%s speaker=%s has_instruct=%s has_reference_wav=%s has_reference_text=%s x_vector_only=%s",
        model_kind,
        len(text),
        normalized_language,
        (speaker or "").strip() or None,
        bool(normalized_instruct),
        reference_wav is not None,
        bool(normalized_ref_text),
        x_vector_only_mode,
    )
    try:
        if normalized_instruct is not None:
            design_model = _ensure_voice_design_model()
            with model_lock:
                wavs, sr = design_model.generate_voice_design(
                    text=text,
                    language=normalized_language,
                    instruct=normalized_instruct,
                    **gen_kwargs,
                )
        elif model_kind == "custom_voice":
            if not _mode_supports_voice_clone():
                raise HTTPException(status_code=503, detail="Voice clone mode is disabled on this server.")
            normalized_speaker = (speaker or "").strip()
            if not normalized_speaker:
                raise HTTPException(status_code=400, detail="`speaker` is required for custom_voice models.")
            with model_lock:
                wavs, sr = tts_model.generate_custom_voice(
                    text=text,
                    language=normalized_language,
                    speaker=normalized_speaker,
                    instruct=None,
                    **gen_kwargs,
                )
        elif model_kind == "base":
            if not _mode_supports_voice_clone():
                raise HTTPException(status_code=503, detail="Voice clone mode is disabled on this server.")
            if reference_wav is None:
                raise HTTPException(
                    status_code=400,
                    detail="Provide `reference_wav` for voice cloning or `instruct` for voice design.",
                )
            uploaded = reference_wav.file.read()
            if not uploaded:
                raise HTTPException(status_code=400, detail="`reference_wav` was provided but is empty.")
            if reference_wav.filename and not reference_wav.filename.lower().endswith(".wav"):
                raise HTTPException(status_code=400, detail="`reference_wav` must be a .wav file.")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded)
                tmp_reference_path = tmp.name

            with model_lock:
                wavs, sr = tts_model.generate_voice_clone(
                    text=text,
                    language=normalized_language,
                    ref_audio=tmp_reference_path,
                    ref_text=normalized_ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs,
                )
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported model kind: {model_kind}")
    except HTTPException:
        logger.warning("Request rejected: %s", traceback.format_exc().strip())
        raise
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise _error_response(exc) from exc
    finally:
        if tmp_reference_path is not None:
            Path(tmp_reference_path).unlink(missing_ok=True)

    pcm_bytes = _wav_to_pcm16le_bytes(wavs[0])
    return Response(
        content=pcm_bytes,
        media_type="audio/pcm",
        headers={
            "X-Audio-Format": "pcm_s16le",
            "X-Sample-Rate": str(sr),
            "X-Channels": "1",
            "X-Model-Kind": str(model_kind),
        },
    )


@app.post("/voice-design")
def voice_design(
    prompt: str = Form(...),
    voice_description: str = Form(...),
    language: str | None = Form(default=None),
    max_new_tokens: int | None = Form(default=None),
    temperature: float | None = Form(default=None),
    top_k: int | None = Form(default=None),
    top_p: float | None = Form(default=None),
    repetition_penalty: float | None = Form(default=None),
    subtalker_top_k: int | None = Form(default=None),
    subtalker_top_p: float | None = Form(default=None),
    subtalker_temperature: float | None = Form(default=None),
) -> Response:
    design_model = _ensure_voice_design_model()

    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        raise HTTPException(status_code=400, detail="`prompt` must be non-empty.")

    normalized_voice_description = voice_description.strip()
    if not normalized_voice_description:
        raise HTTPException(status_code=400, detail="`voice_description` must be non-empty.")

    normalized_language = _normalize_language(language)
    gen_kwargs = _collect_gen_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
        subtalker_temperature=subtalker_temperature,
    )

    logger.info(
        "Received /voice-design request prompt_length=%s language=%s voice_description=%s",
        len(normalized_prompt),
        normalized_language,
        normalized_voice_description,
    )
    try:
        with model_lock:
            wavs, sr = design_model.generate_voice_design(
                text=normalized_prompt,
                language=normalized_language,
                instruct=normalized_voice_description,
                **gen_kwargs,
            )
    except Exception as exc:
        logger.exception("Voice design generation failed")
        raise _error_response(exc) from exc

    wav_bytes = _wav_to_wav_bytes(wavs[0], sr)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Format": "wav_pcm_s16le",
            "X-Sample-Rate": str(sr),
            "X-Channels": "1",
            "X-Model-Kind": "voice_design",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python server.py",
        description="Launch a FastAPI server for Qwen3-TTS.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--mode",
        choices=("voice_clone", "voice_design", "both"),
        default=DEFAULT_MODE,
        help="Load only voice clone, only voice design, or both models.",
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="Primary model checkpoint path or Hugging Face repo id.")
    parser.add_argument(
        "--voice-design-checkpoint",
        default=VOICE_DESIGN_CHECKPOINT,
        help="VoiceDesign model checkpoint path or Hugging Face repo id. Set empty to disable voice design routing.",
    )
    parser.add_argument("--device", default=DEVICE, help="Inference device.")
    parser.add_argument("--dtype", default=DTYPE, help="Torch dtype: bfloat16, float16, or float32.")
    parser.add_argument("--flash-attn", action="store_true", default=FLASH_ATTN, help="Enable flash_attention_2.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_model(
        checkpoint=args.checkpoint,
        voice_design_ckpt=(args.voice_design_checkpoint or "").strip() or None,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
        flash_attn=args.flash_attn,
    )

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
