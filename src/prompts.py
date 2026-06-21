"""
Unified prompts for LLM interactions.
All prompts are organized by execution order and function.
"""

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_ROLE_GENERATOR = "You are an expert medical simulation designer."

SYSTEM_DIALOGUE_GENERATOR = (
    "You are a helpful assistant generating clinical dialogues. "
    "Do not call any tools. Output only the dialogue text."
)

# =============================================================================
# ROLE DOCUMENT GENERATION (Step 2)
# =============================================================================

PROMPT_DOCTOR_PROFILE = """Generate a Doctor profile for a clinical dialogue simulation based on the following criteria:

{context_section}

{global_scores_section}

{wiser_counts_section}

{bias_scores_section}

{bias_counts_section}

{guidance_section}

## Output Format:
Please fill in the following JSON template. **IMPORTANT**: Output ONLY valid JSON without any markdown code blocks or additional text. All string values must properly escape special characters (quotes, newlines, etc.).

{template}

Ensure the personality and behavior fields reflect the scores and counts provided above."""

PROMPT_PATIENT_PROFILE = """Generate a Patient profile for a clinical dialogue simulation.

{context_section}

{patient_scores_section}

{guidance_section}

## Output Format:
Please fill in the following JSON template. **IMPORTANT**: Output ONLY valid JSON without any markdown code blocks or additional text. All string values must properly escape special characters (quotes, newlines, etc.).

{template}"""

PROMPT_MEDICAL_CASE = """Generate a detailed Medical Case profile.

{guidance_section}

{stage_section}

{candidate_section}

## Output Format:
Please fill in the following JSON template. **IMPORTANT**: Output ONLY valid JSON without any markdown code blocks or additional text. All string values must properly escape special characters (quotes, newlines, etc.).

{template}"""

# =============================================================================
# DIALOGUE GENERATION (Step 3)
# =============================================================================

PROMPT_DIALOGUE = """Generate a dialogue between a doctor and a patient based on the following profiles:

Doctor: {doctor_profile}

Patient: {patient_profile}

Condition: {condition_profile}

{behavior_constraints_section}

The dialogue should be realistic and follow the clinical workflow.
The dialogue should have between {min_turns} and {max_turns} turns.

The dialogue must follow this five-stage sequence in order:
1. Greeting (rapport opening and agenda setting)
2. HPI (history of present illness / symptom exploration)
3. Dx (diagnostic reasoning and assessment discussion)
4. Plan (treatment plan, recommendations, and shared next steps)
5. Closing (summary, return precautions, and follow-up arrangement)

Do not print stage titles. Encode the stage progression naturally in the conversation content.

Crucially, to ensure natural-sounding speech synthesis:
1. Include natural filler words and hesitations (e.g., 'um', 'uh', 'well', 'hmm', 'ah') where appropriate, especially for the patient who might be nervous or thinking.
2. Do NOT use stage directions or sound descriptions (e.g., avoid *pause*, *coughs*, [sighs], (silence)). Only write the spoken words.
3. Keep the filler words moderate and natural, do not overuse them.

Format the output strictly following the MOSS-TTSD input format:
- Each turn must be on a new line.
- Start each line with '[S1]' for the Doctor or '[S2]' for the Patient.
- The Doctor ([S1]) must speak first.
- Do not include any other text, markdown, or JSON formatting.
Example:
[S1] Hello, how can I help you?
[S2] Um, well, I have a headache... hmm, it started yesterday."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_context_section(condition: dict = None, patient: dict = None) -> str:
    """Build context section for doctor profile prompt."""
    import json
    parts = []
    if condition:
        parts.append(f"## Medical Case Context:\n{json.dumps(condition, indent=2)}\n")
    if patient:
        parts.append(f"## Patient Profile:\n{json.dumps(patient, indent=2)}\n")
    return "\n".join(parts)


def build_patient_scores_section(patient_scores: dict, codebooks: dict) -> str:
    """Build patient scores/behaviors section for role prompts."""
    if not patient_scores:
        return ""
    
    lines = ["## Patient Personality & Behavior Targets:"]
    for key, value in patient_scores.items():
        # Try finding in bias codebook first (most likely for patient behavior)
        definition, descriptions = _get_codebook_definition(codebooks, 'bias', key)
        
        # If not found, try wiser or global (less likely but possible)
        if not definition:
             definition, descriptions = _get_codebook_definition(codebooks, 'wiser', key)
        if not definition:
             definition, descriptions = _get_codebook_definition(codebooks, 'global', key)
             
        if isinstance(value, list):
            # It's a count range [min, max]
            lines.append(f"- {key}: Target frequency {value}")
        else:
            # It's a scalar score
            lines.append(f"- {key}: {value}")
            
        if definition:
            lines.append(f"  Definition: {definition}")
            
        # Add scale description if available and applicable
        if isinstance(descriptions, dict) and str(value) in descriptions:
            lines.append(f"  Behavior Description for Score {value}: {descriptions[str(value)]}")
        # Add examples if available (for counts)
        elif isinstance(descriptions, list):
            lines.append(f"  Examples: {', '.join(descriptions[:3])}")
            
    return "\n".join(lines)


def build_global_scores_section(global_scores: dict, codebooks: dict) -> str:
    """Build global scores section for role prompts."""
    import json
    if not global_scores:
        return ""
    
    lines = ["## Personality & Attitude Scores (Scale 1-5):"]
    for key, value in global_scores.items():
        definition, descriptions = _get_codebook_definition(codebooks, 'global', key)
        lines.append(f"- {key}: {value}")
        if definition:
            lines.append(f"  Definition: {definition}")
        if isinstance(descriptions, dict) and str(value) in descriptions:
            lines.append(f"  Behavior Description for Score {value}: {descriptions[str(value)]}")
    return "\n".join(lines)


def build_wiser_counts_section(wiser_counts: dict, codebooks: dict) -> str:
    """Build WISER counts section for role prompts."""
    if not wiser_counts:
        return ""
    
    lines = ["## Communication Skills (WISER Model Targets):"]
    for key, value in wiser_counts.items():
        definition, examples = _get_codebook_definition(codebooks, 'wiser', key)
        lines.append(f"- {key}: Target frequency {value}")
        if definition:
            lines.append(f"  Definition: {definition}")
        
        if examples:
            if isinstance(examples, list):
                # Check if list is empty
                if not examples:
                    continue
                    
                # Check if it's a list of strings or list of dicts
                if isinstance(examples[0], dict):
                    # Handle list of dicts (e.g. reflection_statements)
                    formatted_ex = []
                    for ex in examples[:3]:
                        if isinstance(ex, dict):
                            if 'patient' in ex and 'clinician' in ex:
                                formatted_ex.append(f"Patient: {ex['patient']} -> Clinician: {ex['clinician']}")
                            elif 'text' in ex:
                                formatted_ex.append(str(ex['text']))
                            else:
                                # Fallback: join string values
                                parts = [str(v) for v in ex.values() if isinstance(v, (str, int, float))]
                                formatted_ex.append(" ".join(parts))
                        else:
                            formatted_ex.append(str(ex))
                    lines.append(f"  Examples: {'; '.join(formatted_ex)}")
                else:
                    # List of strings
                    lines.append(f"  Examples: {', '.join([str(e) for e in examples[:3]])}")
            
            elif isinstance(examples, dict):
                # Flatten examples if it's a dict (e.g. weight with normal/judgement keys)
                all_ex = []
                for cat_list in examples.values():
                    if isinstance(cat_list, list):
                        all_ex.extend([str(e) for e in cat_list])
                if all_ex:
                    lines.append(f"  Examples: {', '.join(all_ex[:3])}")
    return "\n".join(lines)


def build_bias_scores_section(bias_scores: dict, codebooks: dict) -> str:
    """Build bias scores section for role prompts."""
    if not bias_scores:
        return ""
    
    lines = ["## Bias & Rapport Indicators (Scale):"]
    for key, value in bias_scores.items():
        definition, descriptions = _get_codebook_definition(codebooks, 'bias', key)
        lines.append(f"- {key}: {value}")
        if definition:
            lines.append(f"  Definition: {definition}")
        if isinstance(descriptions, dict) and str(value) in descriptions:
            lines.append(f"  Behavior Description for Score {value}: {descriptions[str(value)]}")
    return "\n".join(lines)


def build_bias_counts_section(bias_counts: dict, codebooks: dict) -> str:
    """Build bias counts section for role prompts."""
    if not bias_counts:
        return ""
    
    lines = ["## Bias & Rapport Behaviors (Counts):"]
    for key, value in bias_counts.items():
        definition, examples = _get_codebook_definition(codebooks, 'bias', key)
        lines.append(f"- {key}: Target frequency {value}")
        if definition:
            lines.append(f"  Definition: {definition}")
        if examples:
            lines.append(f"  Examples: {', '.join(examples[:3])}")
    return "\n".join(lines)


def build_guidance_section(guidance: str) -> str:
    """Build guidance section for role prompts."""
    if not guidance:
        return ""
    return f"## Specific Guidance:\n{guidance}"


def build_stage_section(stage: str) -> str:
    """Build stage section for medical case prompt."""
    if not stage:
        return ""
    return f"## Medical Stage:\n{stage}\n"


def build_candidate_section(candidate: str) -> str:
    """Build candidate section for medical case prompt."""
    if not candidate:
        return ""
    return f"## Scenario candidate:\n{candidate}\n"


def _get_codebook_definition(codebooks: dict, category: str, key: str):
    """Helper to find definition and examples for a specific code in the codebooks."""
    for cb_name, cb_data in codebooks.items():
        if 'dimensions' in cb_data and key in cb_data['dimensions']:
            item = cb_data['dimensions'][key]
            return item.get('definition', ''), item.get('scale_descriptions', {})
        
        if 'scale_behaviors' in cb_data and key in cb_data['scale_behaviors']:
            item = cb_data['scale_behaviors'][key]
            return item.get('definition', ''), item.get('scale_descriptions', {})
            
        if 'behaviors' in cb_data and key in cb_data['behaviors']:
            item = cb_data['behaviors'][key]
            return item.get('definition', ''), item.get('examples', [])
            
        if 'count_behaviors' in cb_data and key in cb_data['count_behaviors']:
            item = cb_data['count_behaviors'][key]
            return item.get('definition', ''), item.get('examples', [])
            
    return "", []
