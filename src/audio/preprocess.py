"""Audio preprocessing for SIMAX."""

import pandas as pd
import shutil
from pathlib import Path

from src.utils.io import setup_logger, save_json

logger = setup_logger("audio_preprocess")


def run(config: dict, experiment: dict, exp_id: str, results_dir: Path):
    """Preprocess audio references for TTS synthesis."""
    logger.info(f"Preprocessing audio for {exp_id}")
    
    tts_config = experiment.get("tts_generation", {})
    use_cloning = tts_config.get("use_cloning", True)
    if not use_cloning:
        logger.info("Audio cloning is disabled. Skipping reference audio processing.")
        return
    
    cv_config = config.get("paths", {}).get("common_voice", {})
    cv_clips = Path(cv_config.get("clips", ""))
    cv_tsv = Path(cv_config.get("tsv", ""))
    
    doctor_audio_id = experiment["audio_reference"]["doctor_audio"]
    patient_audio_id = experiment["audio_reference"]["patient_audio"]
    
    doc_audio_path = cv_clips / doctor_audio_id
    pat_audio_path = cv_clips / patient_audio_id
    
    if not doc_audio_path.exists():
        raise FileNotFoundError(f"Doctor audio not found: {doc_audio_path}")
    if not pat_audio_path.exists():
        raise FileNotFoundError(f"Patient audio not found: {pat_audio_path}")
    
    target_doc_path = results_dir / "ref_doctor.mp3"
    target_pat_path = results_dir / "ref_patient.mp3"
    
    shutil.copy(doc_audio_path, target_doc_path)
    shutil.copy(pat_audio_path, target_pat_path)
    
    logger.info(f"Copied reference audio to {results_dir}")
    
    df = pd.read_csv(cv_tsv, sep="\t")
    
    def get_text(audio_filename: str) -> str:
        row = df[df["path"] == audio_filename]
        if not row.empty:
            return row.iloc[0]["sentence"]
        return ""
    
    doc_text = get_text(doctor_audio_id)
    pat_text = get_text(patient_audio_id)
    
    if not doc_text:
        logger.warning(f"Transcript not found for {doctor_audio_id}")
    if not pat_text:
        logger.warning(f"Transcript not found for {patient_audio_id}")
    
    ref_text = f"[S1] {doc_text} [S2] {pat_text}"
    save_json(
        {
            "reference_text": ref_text,
            "doc_text": doc_text,
            "pat_text": pat_text,
            "ref_doctor": str(target_doc_path.name),
            "ref_patient": str(target_pat_path.name),
        },
        results_dir / "audio_ref_metadata.json",
    )
    
    input_format = tts_config.get("input_format", "separate_speakers")
    if input_format == "shared":
        from pydub import AudioSegment
        
        s1 = AudioSegment.from_mp3(target_doc_path)
        s2 = AudioSegment.from_mp3(target_pat_path)
        
        combined = s1 + s2
        combined.export(results_dir / "ref_combined.wav", format="wav")
        
        logger.info("Created combined reference audio (shared format).")
    else:
        logger.info("Prepared separate speaker references for MOSS-TTSD.")
