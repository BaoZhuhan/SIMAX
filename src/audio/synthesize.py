"""Audio synthesis for SIMAX using MOSS-TTSD."""

import sys
import torch
import soundfile as sf
from pathlib import Path

from src.utils.io import setup_logger, load_json

logger = setup_logger("audio_synthesize")

# Try to import MOSS-TTSD from installed package first, then from local path
try:
    from moss_ttsd.generation_utils import load_model, process_batch
    from moss_ttsd import modeling_asteroid as moss_modeling
    logger.info("Using installed MOSS-TTSD package")
except ImportError:
    # Fallback to local path (for development)
    project_root = Path(__file__).resolve().parents[2]
    moss_path = project_root / "src" / "audio" / "moss-ttsd"
    sys.path.append(str(moss_path))

    try:
        from generation_utils import load_model, process_batch
    except ImportError:
        logger.warning("Could not import MOSS-TTSD. Install with: pip install moss-ttsd")
        load_model = None
        process_batch = None

    try:
        import modeling_asteroid as moss_modeling
    except ImportError:
        moss_modeling = None

# Patch MOSS-TTSD for streaming support if available
if moss_modeling and hasattr(moss_modeling, "CustomMixin"):
    _orig_sample = moss_modeling.CustomMixin._sample

    if not getattr(_orig_sample, "_patched_streamer_default", False):
        def _sample_with_streamer_default(self, *args, **kwargs):
            if "streamer" not in kwargs and len(args) < 6:
                kwargs["streamer"] = None
            return _orig_sample(self, *args, **kwargs)

        _sample_with_streamer_default._patched_streamer_default = True
        moss_modeling.CustomMixin._sample = _sample_with_streamer_default


def run(config: dict, experiment: dict, exp_id: str, results_dir: Path):
    """Synthesize audio using MOSS-TTSD."""
    logger.info(f"Synthesizing audio for {exp_id}")
    
    if load_model is None or process_batch is None:
        raise ImportError("MOSS-TTSD module not loaded. Cannot synthesize.")
    
    dialogue_path = results_dir / "dialogue.json"
    if not dialogue_path.exists():
        raise FileNotFoundError(f"Dialogue not found at {dialogue_path}")
    
    dialogue = load_json(dialogue_path)
    
    tts_config = experiment.get("tts_generation", {})
    use_cloning = tts_config.get("use_cloning", True)
    
    if not use_cloning:
        raise ValueError("MOSS-TTSD requires reference audio; set tts_generation.use_cloning=true.")
    
    ref_meta_path = results_dir / "audio_ref_metadata.json"
    if not ref_meta_path.exists():
        raise FileNotFoundError(f"Audio reference metadata not found at {ref_meta_path}")
    
    ref_meta = load_json(ref_meta_path)
    ref_text = ref_meta.get("reference_text", "")
    ref_doctor = ref_meta.get("ref_doctor")
    ref_patient = ref_meta.get("ref_patient")
    
    moss_config = config.get("paths", {}).get("moss_ttsd", {})
    model_path = moss_config.get("model_path", str(project_root / "resources" / "models" / "moss-ttsd-v0.7"))
    spt_config_path = moss_config.get("tokenizer_config", str(moss_path / "XY_Tokenizer" / "config" / "MOSS_TTSD_tokenizer.yaml"))
    spt_checkpoint_path = moss_config.get("tokenizer_checkpoint", str(moss_path / "XY_Tokenizer" / "weights" / "MOSS_TTSD_tokenizer"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = config.get("models", {}).get("moss_ttsd", {})
    dtype_str = tts_config.get("dtype") or model_cfg.get("dtype", "bf16")
    attn_impl = tts_config.get("attn_implementation") or model_cfg.get("attn_implementation", "flash_attention_2")
    use_normalize = bool(tts_config.get("use_normalize", False))
    silence_duration = float(tts_config.get("silence_duration", 0))
    
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    logger.info(f"Loading MOSS-TTSD model from {model_path} on {device}")
    tokenizer, model, spt = load_model(
        model_path, spt_config_path, spt_checkpoint_path,
        torch_dtype=torch_dtype, attn_implementation=attn_impl,
    )
    model = model.to(device)
    spt = spt.to(device)
    
    dialogue_text = _build_dialogue_text(dialogue)
    if not dialogue_text:
        raise ValueError("Dialogue text is empty after formatting.")
    
    input_format = tts_config.get("input_format", "separate_speakers")
    if input_format == "shared":
        ref_audio_path = results_dir / "ref_combined.wav"
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found at {ref_audio_path}")
        item = {
            "base_path": str(results_dir),
            "text": dialogue_text,
            "prompt_audio": "ref_combined.wav",
            "prompt_text": ref_text,
        }
    else:
        if not ref_doctor or not ref_patient:
            raise ValueError("Missing separate speaker references in audio_ref_metadata.json")
        item = {
            "base_path": str(results_dir),
            "text": dialogue_text,
            "prompt_audio_speaker1": ref_doctor,
            "prompt_text_speaker1": ref_meta.get("doc_text", ""),
            "prompt_audio_speaker2": ref_patient,
            "prompt_text_speaker2": ref_meta.get("pat_text", ""),
        }
    
    with torch.inference_mode():
        _, audio_results = process_batch(
            batch_items=[item],
            tokenizer=tokenizer,
            model=model,
            spt=spt,
            device=device,
            system_prompt="You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
            start_idx=0,
            use_normalize=use_normalize,
            silence_duration=silence_duration,
        )
    
    if not audio_results or audio_results[0] is None:
        raise RuntimeError("MOSS-TTSD generation failed: no audio produced.")
    
    final_output_path = results_dir / "audio.wav"
    _save_audio(final_output_path, audio_results[0]["audio_data"], audio_results[0]["sample_rate"])
    logger.info(f"Final audio saved to {final_output_path}")


def _build_dialogue_text(dialogue: list) -> str:
    """Build dialogue text in MOSS-TTSD format."""
    parts = []
    for turn in dialogue:
        speaker = (turn.get("speaker") or "").lower()
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        tag = "[S1]" if speaker == "doctor" else "[S2]"
        parts.append(f"{tag}{text}")
    return "".join(parts)


def _save_audio(output_path: Path, audio_tensor: torch.Tensor, sample_rate: int) -> None:
    """Save audio tensor to file."""
    audio_np = audio_tensor.cpu().detach().float().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    sf.write(str(output_path), audio_np, sample_rate)
