"""
JSON Schemas for vLLM Guided Decoding.

These schemas ensure vLLM generates structurally valid JSON output
by constraining the generation at the token level.
"""


def get_doctor_schema(gender: str = None, age: str = None, name: str = None) -> dict:
    """Get doctor schema with optional constraints.
    
    Args:
        gender: If specified, constrains the doctor's gender to this value.
               Must be "male" or "female".
        age: If specified, constrains the doctor's age to this value or range.
            Can be a specific age (e.g., "45") or category (e.g., "adult").
        name: If specified, constrains the doctor's name to this value.
    
    Returns:
        JSON schema for doctor profile generation.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "gender": {"type": "string"},
            "age": {"type": ["string", "integer"]},
            "title": {"type": "string"},
            "personality": {
                "type": "object",
                "properties": {
                    "empathy": {"type": "string"},
                    "communication_flow": {"type": "string"},
                    "emotional_attention": {"type": "string"},
                    "attitude": {"type": "string"},
                    "interrupts_patient": {"type": "boolean"},
                    "bias_or_stereotype": {"type": "string"},
                    "tailoring_ability": {"type": "string"},
                    "rapport_building": {"type": "string"},
                    "concern_handling": {"type": "string"}
                },
                "required": ["empathy", "communication_flow", "emotional_attention", 
                            "attitude", "interrupts_patient"]
            },
            "behavior": {
                "type": "object",
                "properties": {
                    "questioning_style": {"type": "string"},
                    "gives_advice": {"type": "boolean"},
                    "assessment_initiative": {"type": "string"},
                    "goal_setting_assistance": {"type": "boolean"},
                    "followup_arrangement": {"type": "boolean"}
                },
                "required": ["questioning_style", "gives_advice"]
            }
        },
        "required": ["name", "gender", "personality", "behavior"]
    }
    
    if gender:
        schema["properties"]["gender"]["enum"] = [gender]
    if age:
        schema["properties"]["age"]["enum"] = [age]
    if name:
        schema["properties"]["name"]["enum"] = [name]
    
    return schema


def get_patient_schema(gender: str = None, age: str = None, name: str = None) -> dict:
    """Get patient schema with optional constraints.
    
    Args:
        gender: If specified, constrains the patient's gender to this value.
               Must be "male" or "female".
        age: If specified, constrains the patient's age to this value or range.
            Can be a specific age (e.g., "45") or category (e.g., "adult").
        name: If specified, constrains the patient's name to this value.
    
    Returns:
        JSON schema for patient profile generation.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "gender": {"type": "string"},
            "age": {"type": ["string", "integer"]},
            "personality": {
                "type": "object",
                "properties": {
                    "trust_tendency": {"type": "string"},
                    "emotional_expression": {"type": "string"},
                    "emotional_state": {"type": "string"},
                    "questioning_initiative": {"type": "boolean"},
                    "response_to_advice": {"type": "string"},
                    "assertiveness": {"type": "string"}
                },
                "required": ["trust_tendency", "emotional_expression", "emotional_state"]
            },
            "behavior": {
                "type": "object",
                "properties": {
                    "empathy_needs": {"type": "string"},
                    "background_disclosure": {"type": "string"},
                    "goal_setting_initiative": {"type": "boolean"},
                    "response_to_doctor_behavior": {"type": "string"}
                },
                "required": ["empathy_needs", "background_disclosure"]
            }
        },
        "required": ["name", "gender", "personality", "behavior"]
    }
    
    if gender:
        schema["properties"]["gender"]["enum"] = [gender]
    if age:
        schema["properties"]["age"]["enum"] = [age]
    if name:
        schema["properties"]["name"]["enum"] = [name]
    
    return schema


def get_medical_case_schema() -> dict:
    """Get medical case schema."""
    return {
        "type": "object",
        "properties": {
            "chief_complaint": {"type": "string"},
            "present_illness_history": {"type": "string"},
            "past_medical_history": {"type": ["string", "array"]},
            "family_history": {"type": "string"},
            "social_background": {"type": "string"},
            "current_health_issues": {"type": ["string", "array", "object"]},
            "patient_concerns": {"type": ["string", "array"]},
            "context_for_required_behaviors": {"type": "string"}
        },
        "required": ["chief_complaint", "present_illness_history"],
        "additionalProperties": True
    }


def get_schema_for_role(role_type: str, gender: str = None, age: str = None, name: str = None) -> dict:
    """Get the appropriate JSON schema for a given role type.
    
    Args:
        role_type: Type of role ("doctor", "patient", "condition").
        gender: Optional gender constraint for doctor/patient roles.
        age: Optional age constraint for doctor/patient roles.
        name: Optional name constraint for doctor/patient roles.
    
    Returns:
        JSON schema for the specified role type.
    """
    if role_type == "doctor":
        return get_doctor_schema(gender, age, name)
    elif role_type == "patient":
        return get_patient_schema(gender, age, name)
    elif role_type == "condition":
        return get_medical_case_schema()
    else:
        raise ValueError(f"Unknown role type: {role_type}")
