"""Role document generation for SIMAX."""

import json
import re
import time
from pathlib import Path
from openai import OpenAI

from src.utils.io import load_json, save_json, setup_logger
from src.json_schemas import get_schema_for_role
from src.prompts import (
    SYSTEM_ROLE_GENERATOR,
    PROMPT_DOCTOR_PROFILE,
    PROMPT_PATIENT_PROFILE,
    PROMPT_MEDICAL_CASE,
    build_context_section,
    build_global_scores_section,
    build_wiser_counts_section,
    build_bias_scores_section,
    build_bias_counts_section,
    build_patient_scores_section,
    build_guidance_section,
)

logger = setup_logger("role_doc")


def extract_and_parse_json(content: str, role_type: str = "content") -> dict:
    """
    Robustly extract and parse JSON from LLM output.
    
    Handles:
    - JSON wrapped in markdown code blocks
    - Control characters in JSON strings
    - Various JSON formatting issues
    """
    if not content or not content.strip():
        raise ValueError(f"Empty content returned for {role_type}")
    
    # Extract JSON from markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    # Try parsing with strict=False to handle control characters
    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError as e:
        # If still fails, try cleaning the content
        logger.debug(f"First parse attempt failed for {role_type}: {e}")
        
        # Remove or escape problematic control characters
        # Keep \n, \r, \t but remove other control chars
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
        
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to parse JSON for {role_type} after cleaning")
            logger.error(f"Original error: {e}")
            logger.error(f"Content preview: {content[:500]}")
            raise


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


def load_codebooks(codebooks_dir: Path, codebook_files: list) -> dict:
    """Load codebooks from the specified directory."""
    codebooks = {}
    for cb_file in codebook_files:
        cb_path = codebooks_dir / cb_file
        if cb_path.exists():
            codebooks[cb_path.stem] = load_json(cb_path)
        else:
            logger.warning(f"Codebook file not found: {cb_path}")
    return codebooks


def load_templates(templates_dir: Path) -> dict:
    """Load role templates from the templates directory."""
    templates = {}
    for role in ["doctor", "patient", "medical_case"]:
        path = templates_dir / f"{role}.json"
        if path.exists():
            templates[role] = load_json(path)
        else:
            logger.warning(f"Template not found: {path}")
    return templates


def generate_role(client: OpenAI, model_name: str, role_type: str, experiment: dict, 
                  codebooks: dict, templates: dict, context: dict = None, 
                  text_config: dict = None, max_retries: int = 3,
                  use_guided_json: bool = True, gender: str = None, 
                  age: str = None, name: str = None) -> dict:
    """Generate a role document using LLM with optional guided JSON.
    
    Args:
        client: OpenAI client for LLM API.
        model_name: Name of the model to use.
        role_type: Type of role to generate ("doctor", "patient", "condition").
        experiment: Experiment configuration dictionary.
        codebooks: Loaded codebooks for behavior definitions.
        templates: Loaded role templates.
        context: Optional context dictionary (condition, patient, etc.).
        text_config: Text generation configuration.
        max_retries: Maximum number of retry attempts.
        use_guided_json: Whether to use vLLM guided JSON decoding.
        gender: Optional gender constraint for doctor/patient roles.
        age: Optional age constraint for doctor/patient roles.
        name: Optional name constraint for doctor/patient roles.
    """
    template = templates.get(role_type, {})
    
    if role_type == "doctor":
        prompt = _build_doctor_prompt(experiment, codebooks, template, context, gender, age, name)
    elif role_type == "patient":
        prompt = _build_patient_prompt(experiment, codebooks, template, context, gender, age, name)
    elif role_type == "condition":
        prompt = _build_case_prompt(experiment, template, context)
    else:
        raise ValueError(f"Unknown role type: {role_type}")
    
    text_config = text_config or {}
    
    for attempt in range(max_retries):
        try:
            # Prepare request parameters
            request_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_ROLE_GENERATOR},
                    {"role": "user", "content": prompt}
                ],
                "temperature": text_config.get("temperature", 0.7),
                "max_tokens": text_config.get("max_tokens", 2048)
            }
            
            # Use vLLM guided JSON if enabled
            if use_guided_json:
                schema = get_schema_for_role(role_type, gender, age, name)
                request_params["extra_body"] = {"guided_json": schema}
                constraints = []
                if gender: constraints.append(f"gender={gender}")
                if age: constraints.append(f"age={age}")
                if name: constraints.append(f"name={name}")
                constraint_str = f" with {', '.join(constraints)}" if constraints else ""
                logger.debug(f"Using vLLM guided JSON for {role_type}{constraint_str}")
            
            response = client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Model returned no content")
            
            # Parse the JSON (should be valid with guided JSON)
            return extract_and_parse_json(content, role_type)
        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed to parse JSON for {role_type}: {e}")
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            logger.error(f"Attempt {attempt+1}/{max_retries} failed for {role_type}: {e}")
            if attempt == max_retries - 1:
                raise
        time.sleep(1)


def _build_doctor_prompt(experiment: dict, codebooks: dict, template: dict, context: dict = None, 
                         gender: str = None, age: str = None, name: str = None) -> str:
    """Build the prompt for generating doctor profile.
    
    Args:
        experiment: Experiment configuration.
        codebooks: Loaded codebooks.
        template: Doctor template.
        context: Context dictionary.
        gender: Optional gender constraint.
        age: Optional age constraint.
        name: Optional name constraint.
    """
    context_section = build_context_section(
        context.get("condition") if context else None,
        context.get("patient") if context else None
    )
    global_scores_section = build_global_scores_section(
        experiment.get("global_scores", {}), codebooks
    )
    wiser_counts_section = build_wiser_counts_section(
        experiment.get("wiser_counts", {}), codebooks
    )
    bias_scores_section = build_bias_scores_section(
        experiment.get("bias_scores", {}), codebooks
    )
    bias_counts_section = build_bias_counts_section(
        experiment.get("bias_counts", {}), codebooks
    )
    guidance_section = build_guidance_section(
        experiment.get("role_guidance", {}).get("doctor", "")
    )
    
    # Add gender requirement if specified
    if gender:
        guidance_section += f"\n\n## Gender Requirement:\nThe doctor must be {gender}. Ensure the name and all pronouns (he/him/his or she/her/hers) are consistent with this gender."
    
    # Add age requirement if specified
    if age:
        guidance_section += f"\n\n## Age Requirement:\nThe doctor's age must be {age}. Ensure this is reflected in the profile."
    
    # Add name requirement if specified
    if name:
        guidance_section += f"\n\n## Name Requirement:\nThe doctor's name must be {name}."
    
    return PROMPT_DOCTOR_PROFILE.format(
        context_section=context_section,
        global_scores_section=global_scores_section,
        wiser_counts_section=wiser_counts_section,
        bias_scores_section=bias_scores_section,
        bias_counts_section=bias_counts_section,
        guidance_section=guidance_section,
        template=json.dumps(template, indent=2)
    )


def _build_patient_prompt(experiment: dict, codebooks: dict, template: dict, context: dict = None, 
                          gender: str = None, age: str = None, name: str = None) -> str:
    """Build the prompt for generating patient profile.
    
    Args:
        experiment: Experiment configuration.
        codebooks: Loaded codebooks.
        template: Patient template.
        context: Context dictionary.
        gender: Optional gender constraint.
        age: Optional age constraint.
        name: Optional name constraint.
    """
    context_section = ""
    if context and "condition" in context:
        context_section = f"## Medical Case Context:\n{json.dumps(context['condition'], indent=2)}\n"
    
    patient_scores_section = build_patient_scores_section(
        experiment.get("patient_scores", {}), codebooks
    )
    
    guidance_section = build_guidance_section(
        experiment.get("role_guidance", {}).get("patient", "")
    )
    
    if not context_section and experiment.get("role_guidance", {}).get("medical_case"):
        guidance_section += f"\n## Medical Context:\n{experiment['role_guidance']['medical_case']}\n"
    
    # Add gender requirement if specified
    if gender:
        guidance_section += f"\n\n## Gender Requirement:\nThe patient must be {gender}. Ensure the name and all pronouns (he/him/his or she/her/hers) are consistent with this gender."
    
    # Add age requirement if specified
    if age:
        guidance_section += f"\n\n## Age Requirement:\nThe patient's age must be {age}. Ensure the medical case is age-appropriate (e.g., no occupational injuries for children, no pregnancy for elderly)."
    
    # Add name requirement if specified
    if name:
        guidance_section += f"\n\n## Name Requirement:\nThe patient's name must be {name}."
    
    return PROMPT_PATIENT_PROFILE.format(
        context_section=context_section,
        patient_scores_section=patient_scores_section,
        guidance_section=guidance_section,
        template=json.dumps(template, indent=2)
    )


def _build_case_prompt(experiment: dict, template: dict, context: dict = None) -> str:
    """Build the prompt for generating medical case.
    
    Args:
        experiment: Experiment configuration.
        template: Medical case template.
        context: Context dictionary (not used, kept for compatibility).
    """
    guidance_section = build_guidance_section(
        experiment.get("role_guidance", {}).get("medical_case", "")
    )
    
    # Use medical stage and scenario from experiment config directly (strong constraint)
    medical_stage = experiment.get("role_generation", {}).get("medical_stage")
    medical_scenario = experiment.get("role_generation", {}).get("medical_scenario")
    
    stage_section = f"## Medical Stage:\n{medical_stage}\n" if medical_stage else ""
    scenario_section = f"## Medical Scenario:\n{medical_scenario}\n" if medical_scenario else ""
    
    return PROMPT_MEDICAL_CASE.format(
        guidance_section=guidance_section,
        stage_section=stage_section,
        candidate_section=scenario_section,
        template=json.dumps(template, indent=2)
    )


def run(config: dict, experiment: dict, exp_id: str, results_dir: Path) -> None:
    """Main role generation step."""
    logger.info(f"Generating role documents for {exp_id}")
    
    client = get_openai_client(config)
    model_name = get_model_name(config)
    
    resources_dir = Path(config["paths"]["resources_dir"])
    codebooks = load_codebooks(
        resources_dir / "codebooks",
        experiment.get("codebook_files", [])
    )
    templates = load_templates(resources_dir / "templates")
    
    max_retries = experiment.get("text_generation", {}).get("max_retries", 3)
    text_config = experiment.get("text_generation", {})
    
    roles = {}
    
    # Get medical stage and scenario from experiment config (required)
    medical_stage = experiment.get("role_generation", {}).get("medical_stage")
    medical_scenario = experiment.get("role_generation", {}).get("medical_scenario")
    
    if not medical_stage:
        raise ValueError("medical_stage is required in experiment.yml under role_generation")
    if not medical_scenario:
        raise ValueError("medical_scenario is required in experiment.yml under role_generation")
    
    logger.info(f"Using medical stage: {medical_stage}")
    logger.info(f"Using medical scenario: {medical_scenario}")
    
    context = {
        "stage": medical_stage,
        "candidate": medical_scenario
    }
    
    # Get gender, age, and name constraints from experiment config
    doctor_gender = experiment.get("role_generation", {}).get("doctor_gender")
    patient_gender = experiment.get("role_generation", {}).get("patient_gender")
    doctor_age = experiment.get("role_generation", {}).get("doctor_age")
    patient_age = experiment.get("role_generation", {}).get("patient_age")
    doctor_name = experiment.get("role_generation", {}).get("doctor_name")
    patient_name = experiment.get("role_generation", {}).get("patient_name")
    
    if doctor_gender:
        logger.info(f"Doctor gender constraint: {doctor_gender}")
    if patient_gender:
        logger.info(f"Patient gender constraint: {patient_gender}")
    if doctor_age:
        logger.info(f"Doctor age constraint: {doctor_age}")
    if patient_age:
        logger.info(f"Patient age constraint: {patient_age}")
    if doctor_name:
        logger.info(f"Doctor name constraint: {doctor_name}")
    if patient_name:
        logger.info(f"Patient name constraint: {patient_name}")
    
    # Check if guided JSON is enabled
    use_guided_json = config.get("models", {}).get("gpt", {}).get("use_guided_json", True)
    if use_guided_json:
        logger.info("vLLM guided JSON enabled for role generation")
    
    logger.info("Generating Medical Case (Condition)...")
    roles["condition"] = generate_role(
        client, model_name, "condition", experiment, codebooks, templates, 
        context=context, text_config=text_config, max_retries=max_retries,
        use_guided_json=use_guided_json
    )
    
    logger.info("Generating Patient...")
    roles["patient"] = generate_role(
        client, model_name, "patient", experiment, codebooks, templates,
        context=roles, text_config=text_config, max_retries=max_retries,
        use_guided_json=use_guided_json,
        gender=patient_gender,
        age=patient_age,
        name=patient_name
    )
    
    logger.info("Generating Doctor...")
    roles["doctor"] = generate_role(
        client, model_name, "doctor", experiment, codebooks, templates,
        context=roles, text_config=text_config, max_retries=max_retries,
        use_guided_json=use_guided_json,
        gender=doctor_gender,
        age=doctor_age,
        name=doctor_name
    )
    
    save_json(roles, results_dir / "role.json")
    logger.info(f"Role documents saved to {results_dir / 'role.json'}")
