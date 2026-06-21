"""Wet audio post-processing for SIMAX.

This step augments synthesized dialogue audio with ambient room sounds
(background, door, keyboard, mouse) and exports a mixed wet track.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from src.utils.io import load_json, save_json, setup_logger

logger = setup_logger("audio_wet")


def run(config: dict, experiment: dict, exp_id: str, results_dir: Path) -> None:
    """Generate wet audio from dialogue audio and dialogue text."""
    logger.info(f"Generating wet audio for {exp_id}")

    wet_config = _get_wet_config(experiment)
    if not wet_config.get("enabled", True):
        logger.info("Wet generation is disabled. Skipping wet step.")
        return

    dialogue_audio_path = results_dir / "audio.wav"
    dialogue_json_path = results_dir / "dialogue.json"
    if not dialogue_audio_path.exists():
        raise FileNotFoundError(f"Dialogue audio not found: {dialogue_audio_path}")
    if not dialogue_json_path.exists():
        raise FileNotFoundError(f"Dialogue JSON not found: {dialogue_json_path}")

    assets_root = _get_assets_root(config, wet_config)
    assets = _resolve_asset_paths(assets_root, wet_config.get("assets", {}))

    aligned_dialogue = generate_timestamps(
        audio_path=dialogue_audio_path,
        dialogue_path=dialogue_json_path,
        output_path=results_dir / wet_config.get("timestamps_filename", "dialogue_with_timestamps.json"),
    )

    patient_segments = [
        (entry["start_time"], entry["end_time"])
        for entry in aligned_dialogue
        if entry.get("speaker") == "patient"
    ]

    rng = _build_rng(exp_id, wet_config.get("seed"))
    output_filename = wet_config.get("output_filename", "audio_wet.wav")
    output_path = results_dir / output_filename

    mix_final_audio(
        dialogue_path=dialogue_audio_path,
        background_path=assets["background"],
        door_sound_path=assets["door"],
        keyboard_sound_path=assets["keyboard"],
        mouse_sound_path=assets["mouse"],
        patient_segments=patient_segments,
        output_path=output_path,
        rng=rng,
        config=wet_config,
    )

    logger.info(f"Wet audio saved to {output_path}")


def load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file and return waveform + sample rate."""
    data, sample_rate = sf.read(str(file_path))
    return data, sample_rate


def save_audio(data: np.ndarray, sample_rate: int, file_path: Path) -> None:
    """Save an audio file to disk."""
    sf.write(str(file_path), data, sample_rate)


def calculate_rms(audio_data: np.ndarray, window_size: int = 2048, hop_size: int = 1024) -> np.ndarray:
    """Compute RMS envelope for loudness-based segmentation."""
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    rms_values = []
    for i in range(0, len(audio_data) - window_size, hop_size):
        window = audio_data[i : i + window_size]
        rms_values.append(float(np.sqrt(np.mean(window**2))))

    return np.array(rms_values, dtype=np.float32)


def detect_speech_segments(
    rms_values: np.ndarray,
    threshold: float = 0.01,
    min_duration: float = 0.5,
    sample_rate: int = 32000,
    hop_size: int = 1024,
) -> list[tuple[float, float, float]]:
    """Detect contiguous speech-like segments from an RMS envelope."""
    speech_mask = rms_values > threshold
    segments = []
    current_start = None

    for i, is_speech in enumerate(speech_mask):
        if is_speech and current_start is None:
            current_start = i
        elif not is_speech and current_start is not None:
            duration = (i - current_start) * hop_size / sample_rate
            if duration >= min_duration:
                start_time = current_start * hop_size / sample_rate
                end_time = i * hop_size / sample_rate
                segments.append((start_time, end_time, duration))
            current_start = None

    if current_start is not None:
        duration = (len(speech_mask) - current_start) * hop_size / sample_rate
        if duration >= min_duration:
            start_time = current_start * hop_size / sample_rate
            end_time = len(speech_mask) * hop_size / sample_rate
            segments.append((start_time, end_time, duration))

    return segments


def identify_speakers(
    segments: list[tuple[float, float, float]],
    audio_data: np.ndarray,
    sample_rate: int,
) -> list[str]:
    """Assign a coarse speaker label for each segment based on RMS."""
    speakers = []
    rms_values = []

    for start, end, _ in segments:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]

        if len(segment_audio) == 0:
            continue

        if segment_audio.ndim == 2:
            segment_audio = np.mean(segment_audio, axis=1)
        rms_values.append(float(np.sqrt(np.mean(segment_audio**2))))

    if len(rms_values) >= 2:
        mean_rms = float(np.mean(rms_values))
        for i, rms in enumerate(rms_values):
            if i == 0:
                speakers.append("doctor")
            elif rms > mean_rms:
                speakers.append("doctor")
            else:
                speakers.append("patient")

    return speakers


def align_dialogue_with_segments(
    dialogue: list[dict],
    segments: list[tuple[float, float, float]],
    speakers: list[str],
) -> list[dict]:
    """Align dialogue turns with detected speech segments."""
    aligned_dialogue = []
    segment_index = 0

    for line in dialogue:
        if segment_index >= len(segments):
            break

        start, end, _ = segments[segment_index]
        detected_speaker = speakers[segment_index] if segment_index < len(speakers) else "unknown"

        if detected_speaker == line.get("speaker") or segment_index == 0:
            aligned_dialogue.append(
                {
                    "speaker": line.get("speaker"),
                    "text": line.get("text"),
                    "start_time": start,
                    "end_time": end,
                }
            )
            segment_index += 1
        else:
            segment_index += 1
            if segment_index < len(segments):
                start, end, _ = segments[segment_index]
                aligned_dialogue.append(
                    {
                        "speaker": line.get("speaker"),
                        "text": line.get("text"),
                        "start_time": start,
                        "end_time": end,
                    }
                )
                segment_index += 1

    return aligned_dialogue


def generate_timestamps(audio_path: Path, dialogue_path: Path, output_path: Path) -> list[dict]:
    """Generate dialogue timestamps by aligning text turns to speech segments."""
    logger.info(f"Loading dialogue audio: {audio_path}")
    audio_data, sample_rate = load_audio(audio_path)
    logger.info(f"Audio sample rate: {sample_rate}, duration: {len(audio_data) / sample_rate:.2f}s")

    logger.info(f"Loading dialogue JSON: {dialogue_path}")
    dialogue = load_json(dialogue_path)
    if not isinstance(dialogue, list):
        raise ValueError("dialogue.json must be a list of dialogue turns.")
    logger.info(f"Dialogue turns: {len(dialogue)}")

    rms_values = calculate_rms(audio_data)
    segments = detect_speech_segments(rms_values, sample_rate=sample_rate)
    logger.info(f"Detected speech segments: {len(segments)}")

    speakers = identify_speakers(segments, audio_data, sample_rate)
    aligned_dialogue = align_dialogue_with_segments(dialogue, segments, speakers)

    save_json(aligned_dialogue, output_path)
    logger.info(f"Saved aligned dialogue timestamps: {output_path}")
    return aligned_dialogue


def adjust_volume(audio_data: np.ndarray, target_db: float = -20) -> np.ndarray:
    """Adjust an audio clip to a target RMS loudness in dBFS."""
    rms = float(np.sqrt(np.mean(audio_data**2)))
    if rms <= 0:
        return audio_data
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain_factor = 10 ** (gain_db / 20)
    return audio_data * gain_factor


def create_background_with_fade(
    background_data: np.ndarray,
    door_start: float,
    door_end: float,
    sample_rate: int,
    initial_gain_db: float = 5,
    final_gain_db: float = -15,
) -> np.ndarray:
    """Apply a gain fade around the door-close interval on background audio."""
    door_start_sample = int(door_start * sample_rate)
    door_end_sample = int(door_end * sample_rate)

    initial_gain = 10 ** (initial_gain_db / 20)
    final_gain = 10 ** (final_gain_db / 20)
    gain_curve = np.ones(len(background_data), dtype=np.float32) * initial_gain

    if door_start_sample < len(gain_curve):
        gain_curve[:door_start_sample] = initial_gain
        if door_end_sample > door_start_sample:
            fade_length = door_end_sample - door_start_sample
            fade_curve = np.linspace(initial_gain, final_gain, fade_length, dtype=np.float32)
            gain_curve[door_start_sample:door_end_sample] = fade_curve
        if door_end_sample < len(gain_curve):
            gain_curve[door_end_sample:] = final_gain

    if background_data.ndim == 2:
        for i in range(background_data.shape[1]):
            background_data[:, i] *= gain_curve
    else:
        background_data *= gain_curve

    return background_data


def group_segments_by_time_window(
    segments: list[tuple[float, float]], window_seconds: int
) -> list[list[tuple[float, float]]]:
    """Group segments into fixed-size time windows."""
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x[0])
    windows = []
    current_window = []
    window_start = 0

    for seg_start, seg_end in sorted_segments:
        while seg_start >= window_start + window_seconds:
            if current_window:
                windows.append(current_window)
            current_window = []
            window_start += window_seconds
        current_window.append((seg_start, seg_end))

    if current_window:
        windows.append(current_window)

    return windows


def select_segments_for_sounds(
    segments_by_window: list[list[tuple[float, float]]],
    rng: random.Random,
    min_per_window: int = 1,
    max_per_window: int = 2,
) -> list[tuple[float, float]]:
    """Randomly sample segments per time window."""
    selected = []
    for window in segments_by_window:
        if not window:
            continue
        count = rng.randint(min_per_window, min(max_per_window, len(window)))
        selected.extend(rng.sample(window, count))
    return selected


def mix_final_audio(
    dialogue_path: Path,
    background_path: Path,
    door_sound_path: Path,
    keyboard_sound_path: Path,
    mouse_sound_path: Path,
    patient_segments: list[tuple[float, float]],
    output_path: Path,
    rng: random.Random,
    config: dict,
) -> None:
    """Mix dialogue with ambient effects and save wet audio."""
    dialogue, sample_rate = load_audio(dialogue_path)
    background, sr_bg = load_audio(background_path)
    door_sound, sr_door = load_audio(door_sound_path)
    keyboard_sound, sr_keyboard = load_audio(keyboard_sound_path)
    mouse_sound, sr_mouse = load_audio(mouse_sound_path)

    logger.info(
        "Loaded tracks | dialogue=%.2fs, background=%.2fs, door=%.2fs, keyboard=%.2fs, mouse=%.2fs",
        len(dialogue) / sample_rate,
        len(background) / sr_bg,
        len(door_sound) / sr_door,
        len(keyboard_sound) / sr_keyboard,
        len(mouse_sound) / sr_mouse,
    )

    if sample_rate != sr_bg:
        background = signal.resample(background, len(dialogue))
    if sample_rate != sr_door:
        num_samples = int(len(door_sound) * sample_rate / sr_door)
        door_sound = signal.resample(door_sound, num_samples)
    if sample_rate != sr_keyboard:
        num_samples = int(len(keyboard_sound) * sample_rate / sr_keyboard)
        keyboard_sound = signal.resample(keyboard_sound, num_samples)
    if sample_rate != sr_mouse:
        num_samples = int(len(mouse_sound) * sample_rate / sr_mouse)
        mouse_sound = signal.resample(mouse_sound, num_samples)

    dialogue = _to_multichannel(dialogue)
    background = _to_multichannel(background)
    door_sound = _to_multichannel(door_sound)
    keyboard_sound = _to_multichannel(keyboard_sound)
    mouse_sound = _to_multichannel(mouse_sound)

    max_channels = max(
        dialogue.shape[1],
        background.shape[1],
        door_sound.shape[1],
        keyboard_sound.shape[1],
        mouse_sound.shape[1],
    )
    dialogue = _match_channels(dialogue, max_channels)
    background = _match_channels(background, max_channels)
    door_sound = _match_channels(door_sound, max_channels)
    keyboard_sound = _match_channels(keyboard_sound, max_channels)
    mouse_sound = _match_channels(mouse_sound, max_channels)

    door_tracks = np.zeros_like(dialogue)
    keyboard_tracks = np.zeros_like(dialogue)
    mouse_tracks = np.zeros_like(dialogue)
    sound_timings = {"door": [], "keyboard": [], "mouse": []}

    background = adjust_volume(background, float(config.get("background_target_db", -28)))
    door_sound = adjust_volume(door_sound, float(config.get("door_target_db", -20)))
    keyboard_sound = adjust_volume(keyboard_sound, float(config.get("keyboard_target_db", -25)))
    mouse_sound = adjust_volume(mouse_sound, float(config.get("mouse_target_db", -25)))

    door_length = len(door_sound)
    door_tracks[0:door_length] += door_sound
    sound_timings["door"].append((0.0, door_length / sample_rate))

    background = create_background_with_fade(
        background_data=background,
        door_start=0.0,
        door_end=door_length / sample_rate,
        sample_rate=sample_rate,
        initial_gain_db=float(config.get("background_initial_gain_db", 5)),
        final_gain_db=float(config.get("background_final_gain_db", -15)),
    )

    mouse_windows = group_segments_by_time_window(
        patient_segments, int(config.get("mouse_window_seconds", 60))
    )
    mouse_segments = select_segments_for_sounds(
        mouse_windows,
        rng=rng,
        min_per_window=int(config.get("mouse_min_per_window", 1)),
        max_per_window=int(config.get("mouse_max_per_window", 2)),
    )

    for start_time, end_time in mouse_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_length = end_sample - start_sample
        if segment_length <= 0:
            continue

        play_count = 2 if rng.random() < float(config.get("mouse_double_click_probability", 0.8)) else 1
        for _ in range(play_count):
            mouse_length = len(mouse_sound)
            if mouse_length <= segment_length:
                mouse_start = start_sample + rng.randint(0, segment_length - mouse_length)
                mouse_tracks[mouse_start : mouse_start + mouse_length] += mouse_sound
                sound_timings["mouse"].append(
                    (mouse_start / sample_rate, (mouse_start + mouse_length) / sample_rate)
                )

    keyboard_windows = group_segments_by_time_window(
        patient_segments, int(config.get("keyboard_window_seconds", 120))
    )
    keyboard_segments = select_segments_for_sounds(
        keyboard_windows,
        rng=rng,
        min_per_window=int(config.get("keyboard_min_per_window", 1)),
        max_per_window=int(config.get("keyboard_max_per_window", 2)),
    )

    for start_time, end_time in keyboard_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_length = end_sample - start_sample
        if segment_length <= 0:
            continue

        min_duration = int(float(config.get("keyboard_min_duration_sec", 0.5)) * sample_rate)
        max_duration = int(float(config.get("keyboard_max_duration_sec", 6.0)) * sample_rate)
        max_available = min(max_duration, len(keyboard_sound), segment_length)
        if max_available < min_duration:
            continue

        keyboard_length = rng.randint(min_duration, max_available)
        start_pos = rng.randint(0, len(keyboard_sound) - keyboard_length)
        keyboard_segment = keyboard_sound[start_pos : start_pos + keyboard_length]
        keyboard_tracks[start_sample : start_sample + keyboard_length] += keyboard_segment
        sound_timings["keyboard"].append((start_time, start_time + keyboard_length / sample_rate))

    mixed = dialogue + background + door_tracks + keyboard_tracks + mouse_tracks
    max_amplitude = float(np.max(np.abs(mixed)))
    if max_amplitude > 1.0:
        mixed = mixed / max_amplitude

    save_audio(mixed, sample_rate, output_path)

    logger.info(
        "Sound events | door=%d, keyboard=%d, mouse=%d",
        len(sound_timings["door"]),
        len(sound_timings["keyboard"]),
        len(sound_timings["mouse"]),
    )


def _to_multichannel(audio_data: np.ndarray) -> np.ndarray:
    """Ensure shape is (samples, channels)."""
    if audio_data.ndim == 1:
        return np.expand_dims(audio_data, axis=1)
    return audio_data


def _match_channels(audio_data: np.ndarray, channels: int) -> np.ndarray:
    """Duplicate mono channels to match target channel count."""
    if audio_data.shape[1] >= channels:
        return audio_data
    return np.tile(audio_data, (1, channels))


def _get_wet_config(experiment: dict) -> dict:
    """Return wet step config, with backward-compatible fallback key."""
    wet_config = experiment.get("wet_generation")
    if wet_config is not None:
        return wet_config
    return experiment.get("reverb_generation", {})


def _get_assets_root(config: dict, wet_config: dict) -> Path:
    """Resolve the root directory for wet assets."""
    configured_assets_dir = wet_config.get("assets_dir")
    if configured_assets_dir:
        assets_root = Path(configured_assets_dir)
    else:
        resources_dir = Path(config.get("paths", {}).get("resources_dir", "./resources"))
        assets_root = resources_dir / "audio_assets"

    if not assets_root.exists():
        raise FileNotFoundError(f"Wet assets directory not found: {assets_root}")
    return assets_root


def _resolve_asset_paths(assets_root: Path, assets_config: dict) -> dict[str, Path]:
    """Resolve required wet assets from config or defaults."""
    defaults = {
        "background": "201638__bassboybg__hospital-waiting-room.wav",
        "door": "126044__mhtaylor67__office-door-closing.wav",
        "keyboard": "331428__m4taiori__mechanical-keyboard-typing.mp3",
        "mouse": "545960__metrowned__mouse_click.wav",
    }

    resolved = {}
    for key, default_file in defaults.items():
        filename = assets_config.get(key, default_file)
        path = assets_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing wet asset '{key}': {path}")
        resolved[key] = path

    return resolved


def _build_rng(exp_id: str, configured_seed: int | None) -> random.Random:
    """Create a deterministic RNG for wet event placement."""
    if configured_seed is not None:
        return random.Random(int(configured_seed))
    seed_bytes = hashlib.sha256(exp_id.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(seed_bytes, 16))
