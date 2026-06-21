"""Dialogue generation for SIMAX."""

import json
import time
from pathlib import Path
from openai import OpenAI

from src.utils.io import load_json, save_json, setup_logger
from src.prompts import (
    PROMPT_DIALOGUE,
    build_global_scores_section,
    build_wiser_counts_section,
    build_bias_scores_section,
    build_bias_counts_section,
    build_patient_scores_section,
)

logger = setup_logger("dialogue_gen")


def get_openai_client(config: dict) -> OpenAI:
    """Get OpenAI client from config (local vLLM only)."""
    gpt_config = config.get("models", {}).get("gpt", {})
    return OpenAI(
        base_url=gpt_config.get("base_url", "http://localhost:8000/v1"),
        api_key="EMPTY"
    )


def get_model_name(config: dict) -> str:
    """Get model name from config."""
    return config.get("models", {}).get("gpt", {}).get("model_name", "gpt-oss-20b")


def run(config: dict, experiment: dict, exp_id: str, results_dir: Path) -> None:
    """Generate dialogue text based on role documents."""
    logger.info(f"Generating dialogue for {exp_id}")
    
    client = get_openai_client(config)
    model_name = get_model_name(config)
    
    role_path = results_dir / "role.json"
    if not role_path.exists():
        raise FileNotFoundError(f"Role document not found at {role_path}")
    
    roles = load_json(role_path)
    codebooks = _load_codebooks(config, experiment)
    
    dialogue_config = experiment.get("text_generation", {}).get("dialogue_generation", {})
    text_config = experiment.get("text_generation", {})
    max_retries = text_config.get("max_retries", 3)
    
    prompt = _build_dialogue_prompt(roles, dialogue_config, experiment, codebooks)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant generating clinical dialogues. Output only the dialogue text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=text_config.get("temperature", 0.7),
                max_tokens=text_config.get("max_tokens", 4096)
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            content = _clean_content(content)
            dialogue_text = content.strip()
            
            with open(results_dir / "dialogue.txt", "w", encoding="utf-8") as f:
                f.write(dialogue_text)
            logger.info(f"Dialogue text saved to {results_dir / 'dialogue.txt'}")
            
            dialogue_json = _parse_dialogue(dialogue_text)
            if not dialogue_json:
                raise ValueError("No valid dialogue lines found")
            
            save_json(dialogue_json, results_dir / "dialogue.json")
            logger.info(f"Dialogue JSON saved to {results_dir / 'dialogue.json'}")
            
            return
            
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)


def _build_dialogue_prompt(roles: dict, dialogue_config: dict, experiment: dict, codebooks: dict) -> str:
    """Build the dialogue generation prompt."""
    max_turns = dialogue_config.get("max_turns", 50)
    min_turns = dialogue_config.get("min_turns", 10)
    behavior_constraints_section = _build_dialogue_behavior_constraints(experiment, codebooks)

    return PROMPT_DIALOGUE.format(
        doctor_profile=json.dumps(roles.get("doctor", {}), indent=2),
        patient_profile=json.dumps(roles.get("patient", {}), indent=2),
        condition_profile=json.dumps(roles.get("condition", {}), indent=2),
        behavior_constraints_section=behavior_constraints_section,
        min_turns=min_turns,
        max_turns=max_turns,
    )


def _build_dialogue_behavior_constraints(experiment: dict, codebooks: dict) -> str:
    """
    Build dialogue-level behavioral constraints from experiment settings.
    Only labels present in experiment settings are injected.
    """
    sections = [
        build_global_scores_section(experiment.get("global_scores", {}), codebooks),
        build_wiser_counts_section(experiment.get("wiser_counts", {}), codebooks),
        build_bias_scores_section(experiment.get("bias_scores", {}), codebooks),
        build_bias_counts_section(experiment.get("bias_counts", {}), codebooks),
        build_patient_scores_section(experiment.get("patient_scores", {}), codebooks),
    ]
    sections = [s for s in sections if s]
    if not sections:
        return ""

    return (
        "The following codebook-grounded targets MUST be reflected in the dialogue content. "
        "For count-based targets, aim to satisfy the requested frequencies across the full encounter.\n\n"
        + "\n\n".join(sections)
    )


def _load_codebooks(config: dict, experiment: dict) -> dict:
    """Load only explicitly selected codebook files for dialogue constraints."""
    resources_dir = Path(config.get("paths", {}).get("resources_dir", "./resources"))
    codebooks_dir = resources_dir / "codebooks"
    codebook_files = experiment.get("codebook_files", [])
    codebooks = {}
    for cb_file in codebook_files:
        cb_path = codebooks_dir / cb_file
        if cb_path.exists():
            codebooks[cb_path.stem] = load_json(cb_path)
        else:
            logger.warning(f"Codebook file not found for dialogue generation: {cb_path}")
    return codebooks


def _clean_content(content: str) -> str:
    """Clean markdown code blocks from content."""
    if "```" in content:
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines)
    return content


def _parse_dialogue(text: str) -> list:
    """Parse dialogue text to JSON format."""
    dialogue_json = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("[S1]"):
            speaker = "doctor"
            text_content = line[4:].strip()
        elif line.startswith("[S2]"):
            speaker = "patient"
            text_content = line[4:].strip()
        else:
            logger.warning(f"Skipping malformed line: {line}")
            continue
        
        dialogue_json.append({"speaker": speaker, "text": text_content})
    
    return dialogue_json
