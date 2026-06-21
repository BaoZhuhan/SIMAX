import csv
import yaml
import os
import random
from pathlib import Path
import re

# Name Library (Gender: Male/Female, Age Group: Child/Teen/Adult/Senior, Region: US/UK/India/AUS/Africa)
NAME_LIBRARY = {
    "US": {
        "M": ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"],
        "F": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"]
    },
    "UK": {
        "M": ["Oliver", "George", "Harry", "Jack", "Jacob", "Charlie", "Thomas", "Oscar", "William", "James"],
        "F": ["Olivia", "Amelia", "Isla", "Ava", "Emily", "Isabella", "Mia", "Poppy", "Jessica", "Lily"]
    },
    "India": {
        "M": ["Aarav", "Vihaan", "Vivaan", "Ansh", "Ishaan", "Arjun", "Sai", "Ayaan", "Reyansh", "Krishna"],
        "F": ["Aadya", "Ananya", "Diya", "Isha", "Myra", "Pari", "Riya", "Saanvi", "Aditi", "Kiara"]
    },
    "AUS": {
        "M": ["Oliver", "Noah", "Jack", "William", "Leo", "Lucas", "Thomas", "Henry", "Charlie", "James"],
        "F": ["Charlotte", "Olivia", "Mia", "Ava", "Amelia", "Willow", "Matilda", "Isla", "Ruby", "Chloe"]
    },
    "Africa": { # General African names (mix of regions for simplicity)
        "M": ["Kwame", "Kofi", "Ade", "Chike", "Tunde", "Emeka", "Jabari", "Sekou", "Malik", "Amara"],
        "F": ["Aisha", "Fatima", "Zainab", "Nia", "Zola", "Amara", "Chioma", "Zuri", "Imani", "Ayana"]
    },
    "Other": {
        "M": ["Alex", "Sam", "Jordan", "Casey", "Taylor", "Jamie", "Robin", "Drew", "Morgan", "Riley"],
        "F": ["Alex", "Sam", "Jordan", "Casey", "Taylor", "Jamie", "Robin", "Drew", "Morgan", "Riley"]
    }
}

def get_random_name(region, gender):
    """Selects a random name based on region and gender."""
    region_names = NAME_LIBRARY.get(region, NAME_LIBRARY["Other"])
    gender_names = region_names.get(gender, region_names.get("M")) # Fallback to Male if gender unknown
    return random.choice(gender_names)

# Scenario Templates
# Structure: {AgeGroup: {Gender: [List of Scenarios]}}
# Using "All" for gender-neutral scenarios
SCENARIO_TEMPLATES = {
    "child": {
        "All": [
            "Fell from a slide at the playground",
            "Injured while playing soccer",
            "Fell out of a bunk bed",
            "Tripped while running in the park",
            "Injured wrist during gymnastics practice"
        ]
    },
    "teen": {
        "All": [
            "Sports injury during a basketball game",
            "Skateboard accident",
            "Bicycle accident on the way to school",
            "Twisted ankle while hiking",
            "Fell during dance practice"
        ]
    },
    "adult": {
        "All": [
            "Slip and fall on wet floor at work",
            "Car accident resulting in potential fracture",
            "Fell from a ladder while doing home repairs",
            "Tripped on an uneven sidewalk",
            "Heavy object fell on foot at a construction site"
        ]
    },
    "senior": {
        "All": [
            "Slip and fall in the bathroom",
            "Tripped over a rug at home",
            "Fell while gardening",
            "Lost balance while walking down stairs",
            "Hip pain after a minor fall"
        ]
    }
}

# Medical Stages
MEDICAL_STAGES_OBSTETRICS = [
    "First Trimester Visit (Initial)",
    "Routine Prenatal Visit (20 weeks)",
    "Third Trimester Check-up",
    "Postpartum Follow-up (6 weeks)",
    "Urgent Visit (Pain/Bleeding)"
]

MEDICAL_STAGES_ORTHOPEDICS = [
    "Initial diagnosis (Emergency Room)",
    "Follow-up consultation (2 weeks post-injury)",
    "Treatment planning (Surgery vs Conservative)",
    "Post-treatment review (Cast removal/Rehabilitation)"
]

# Scenario Templates
SCENARIO_TEMPLATES_OBSTETRICS = {
    "adult": {
        "female": [
             "Routine check-up, no complaints",
             "Complaining of nausea and vomiting",
             "Concerned about decreased fetal movement",
             "Vaginal bleeding concern",
             "Abdominal pain and cramping",
             "High blood pressure symptoms (headache, vision changes)",
             "Gestational diabetes screening discussion",
             "Birth plan discussion",
             "Breastfeeding questions"
        ]
    }
}

SCENARIO_TEMPLATES_ORTHOPEDICS = {
    "child": {
        "All": [
            "Fell from a slide at the playground",
            "Injured while playing soccer",
            "Fell out of a bunk bed",
            "Tripped while running in the park",
            "Injured wrist during gymnastics practice"
        ]
    },
    "teen": {
        "All": [
            "Sports injury during a basketball game",
            "Skateboard accident",
            "Bicycle accident on the way to school",
            "Twisted ankle while hiking",
            "Fell during dance practice"
        ]
    },
    "adult": {
        "All": [
            "Slip and fall on wet floor at work",
            "Car accident resulting in potential fracture",
            "Fell from a ladder while doing home repairs",
            "Tripped on an uneven sidewalk",
            "Heavy object fell on foot at a construction site"
        ]
    },
    "senior": {
        "All": [
            "Slip and fall in the bathroom",
            "Tripped over a rug at home",
            "Fell while gardening",
            "Lost balance while walking down stairs",
            "Hip pain after a minor fall"
        ]
    }
}

def get_scenario_ob(age_group, gender):
    """Selects a random scenario appropriate for Obstetrics."""
    # Strict fallback for safety
    if age_group != "adult" or gender != "female":
         return "Standard prenatal consultation"
         
    group_templates = SCENARIO_TEMPLATES_OBSTETRICS.get(age_group, {})
    scenarios = group_templates.get(gender, [])
    
    if not scenarios:
        return "Standard prenatal consultation"
        
    return random.choice(scenarios)

def get_scenario_ortho(age_group, gender):
    """Selects a random scenario appropriate for Orthopedics."""
    group_templates = SCENARIO_TEMPLATES_ORTHOPEDICS.get(age_group, SCENARIO_TEMPLATES_ORTHOPEDICS["adult"])
    
    # Try specific gender first, then fall back to "All"
    scenarios = group_templates.get(gender, []) + group_templates.get("All", [])
    
    if not scenarios:
        return "Patient presenting with a suspected fracture after a fall"
        
    return random.choice(scenarios)

def get_random_stage():
    """Selects a random medical stage."""
    return random.choice(MEDICAL_STAGES)


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def get_doctor_audio(csv_path):
    """
    Finds a middle-aged North American doctor audio.
    Target: United States English, Middle Age (Male).
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Locate "United States English" row
    us_row_idx = -1
    for i, row in enumerate(rows):
        if row and "United States" in row[0]:
            us_row_idx = i
            break
    
    if us_row_idx == -1:
        # Fallback search if exact string match fails
        for i, row in enumerate(rows):
             if row and "US" in row[0]:
                us_row_idx = i
                break
    
    if us_row_idx == -1:
        raise ValueError("Could not find 'United States English' in CSV")

    # Columns based on header row 1 (0-indexed):
    # 1: Kids M, 2: Kids F, 3: Teen M, 4: Teen F, 5: Middle M, 6: Middle F, 7: Old M, 8: Old F
    
    doctor_audio = None
    
    # We look at the rows starting from us_row_idx until the next accent section
    for i in range(us_row_idx, len(rows)):
        row = rows[i]
        if not row: continue
        
        # Stop if we hit another accent section (but skip the first one which is US)
        if i > us_row_idx and row[0].strip() != "":
            break
            
        # Check Middle Age Male (Col 5)
        if len(row) > 5 and row[5].strip().endswith('.mp3'):
            doctor_audio = row[5].strip()
            break
            
    if not doctor_audio:
        # Fallback to hardcoded if CSV lookup fails to be robust
        doctor_audio = "common_voice_en_39477765.mp3" # From observed CSV snippet
        print(f"Warning: Could not dynamically find doctor audio, using fallback: {doctor_audio}")
        
    return doctor_audio

def get_patient_metadata(csv_path):
    """
    Extracts metadata for all patient voices from the CSV.
    Returns a dict: {filename: {'region': str, 'age': str, 'gender': str}}
    """
    metadata = {}
    current_region = "Unknown"
    
    # Mapping for columns based on header:
    # 1: Kids M, 2: Kids F
    # 3: Teen M, 4: Teen F
    # 5: Middle M, 6: Middle F
    # 7: Old M, 8: Old F
    col_map = {
        1: ("child", "male"), 2: ("child", "female"),
        3: ("teen", "male"), 4: ("teen", "female"),
        5: ("adult", "male"), 6: ("adult", "female"),
        7: ("senior", "male"), 8: ("senior", "female")
    }

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    # Skip headers (row 0 and 1)
    for i in range(2, len(rows)):
        row = rows[i]
        if not row: continue
        
        # Check for new region in col 0
        if row[0].strip():
            raw_region = row[0].strip()
            if "United States" in raw_region: current_region = "US"
            elif "England" in raw_region: current_region = "UK"
            elif "India" in raw_region: current_region = "India"
            elif "Australian" in raw_region: current_region = "AUS"
            elif "African" in raw_region: current_region = "Africa"
            else: current_region = "Other"
            
        for col_idx in range(1, 9):
            if col_idx >= len(row): break
            cell = row[col_idx].strip()
            if cell.endswith('.mp3'):
                age_group, gender = col_map.get(col_idx, ("adult", "male")) # Default
                
                # Assign specific age based on group
                age = "35" # Default adult
                if age_group == "child": age = str(random.randint(8, 12))
                elif age_group == "teen": age = str(random.randint(13, 19))
                elif age_group == "adult": age = str(random.randint(30, 50))
                elif age_group == "senior": age = str(random.randint(65, 80))

                metadata[cell] = {
                    "region": current_region,
                    "age": age,          # Specific number
                    "age_group": age_group, # Group name for filename
                    "gender": gender
                }
    return metadata

def main():
    root_dir = Path(__file__).resolve().parents[1]
    data_csv = root_dir / "data.csv"
    template_yml = root_dir / "experiment.yml" # Use the actual updated experiment.yml as template
    output_dir = root_dir / "experiments" / "obstetrics_study"
    
    if not data_csv.exists():
        print(f"Error: {data_csv} not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get Doctor Audio
    try:
        doctor_audio = get_doctor_audio(data_csv)
    except Exception as e:
        print(f"Warning: {e}")
        doctor_audio = "common_voice_en_12345.mp3" # Fallback

    doctor_name = "Dr. John Smith"
    doctor_age = "45"
    doctor_gender = "male"
    
    print(f"Selected Doctor Audio: {doctor_audio}")
    
    # 2. Get Patient Metadata
    patient_meta = get_patient_metadata(data_csv)
    print(f"Found {len(patient_meta)} patient voices.")
    
    # 3. Load Template
    if template_yml.exists():
        base_config = load_yaml(template_yml)
    else:
        # Fallback if file missing (though we expect it)
        base_config = {
             "role_generation": {},
             "role_guidance": {},
             "audio_reference": {},
             "global_scores": {},
             "wiser_counts": {},
             "bias_scores": {},
             "bias_counts": {},
             "patient_scores": {}
        }

    # Ensure keys exist
    if 'patient_scores' not in base_config:
        base_config['patient_scores'] = {"guarded_open": 3, "trust_patient": [1, 3]}
    if 'social_determinants_of_health' not in base_config.get('wiser_counts', {}):
        if 'wiser_counts' not in base_config: base_config['wiser_counts'] = {}
        base_config['wiser_counts']['social_determinants_of_health'] = [0, 1]
    if 'weight' not in base_config.get('wiser_counts', {}):
        base_config['wiser_counts']['weight'] = [0, 1]

    # 4. Generate Experiments
    # We will vary global scores (empathy levels)
    scores = [1, 3, 5]
    
    count_ob = 0
    count_ortho = 0
    
    for patient_audio, meta in patient_meta.items():
        # Skip if patient audio is same as doctor
        if patient_audio == doctor_audio:
            continue
            
        region = meta['region']
        age = meta['age'] 
        age_group = meta['age_group']
        gender = meta['gender']
        
        # Generate names
        patient_name_female = get_random_name(region, "F")
        patient_name_male = get_random_name(region, "M")
        patient_name = patient_name_female if gender == "female" else patient_name_male
        
        # --- OBSTETRICS GENERATION (Adult Female Only) ---
        if gender == "female" and age_group == "adult":
            for score in scores:
                config = base_config.copy()
                # Deep copies
                config['role_generation'] = base_config.get('role_generation', {}).copy()
                config['role_guidance'] = base_config.get('role_guidance', {}).copy()
                config['audio_reference'] = base_config.get('audio_reference', {}).copy()
                config['global_scores'] = base_config.get('global_scores', {}).copy()
                config['wiser_counts'] = base_config.get('wiser_counts', {}).copy()
                config['patient_scores'] = base_config.get('patient_scores', {}).copy()
                
                # Setup
                config['role_generation']['medical_stage'] = random.choice(MEDICAL_STAGES_OBSTETRICS)
                config['role_generation']['medical_scenario'] = get_scenario_ob(age_group, gender)
                config['role_generation']['doctor_gender'] = doctor_gender
                config['role_generation']['doctor_age'] = doctor_age
                config['role_generation']['doctor_name'] = doctor_name
                config['role_generation']['patient_gender'] = gender
                config['role_generation']['patient_age'] = age
                config['role_generation']['patient_name'] = patient_name_female
                
                config['role_guidance']['medical_case'] = "The case should belong to the Department of Obstetrics. Ensure scenarios are medically plausible and age-appropriate."
                config['audio_reference']['doctor_audio'] = doctor_audio
                config['audio_reference']['patient_audio'] = patient_audio
                
                # Scores
                for key in ['flow', 'concerns', 'attentive', 'warmth', 'respect']:
                    config['global_scores'][key] = score
                
                if score == 1:
                    config['wiser_counts']['empathic_response'] = [0, 1]
                    config['wiser_counts']['open_ended_question'] = [1, 2]
                    config['wiser_counts']['social_determinants_of_health'] = [0, 0]
                    config['wiser_counts']['weight'] = [0, 0]
                elif score == 5:
                    config['wiser_counts']['empathic_response'] = [4, 6]
                    config['wiser_counts']['open_ended_question'] = [4, 6]
                    config['wiser_counts']['social_determinants_of_health'] = [1, 2]
                    config['wiser_counts']['weight'] = [0, 1]
                
                # Filename
                pat_id_match = re.search(r'(\d+)', patient_audio)
                pat_id = pat_id_match.group(1) if pat_id_match else "unknown"
                exp_name = f"ob_{region}_{age_group}_{gender}_pat{pat_id}_score{score}"
                
                # Output dir
                ob_dir = root_dir / "experiments" / "obstetrics"
                ob_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = ob_dir / f"exp_{exp_name}.yml"
                save_yaml(config, save_path)
                count_ob += 1

        # --- ORTHOPEDICS GENERATION (All Groups) ---
        # Generate for everyone
        for score in scores:
            config = base_config.copy()
            # Deep copies
            config['role_generation'] = base_config.get('role_generation', {}).copy()
            config['role_guidance'] = base_config.get('role_guidance', {}).copy()
            config['audio_reference'] = base_config.get('audio_reference', {}).copy()
            config['global_scores'] = base_config.get('global_scores', {}).copy()
            config['wiser_counts'] = base_config.get('wiser_counts', {}).copy()
            config['patient_scores'] = base_config.get('patient_scores', {}).copy()
            
            # Setup
            config['role_generation']['medical_stage'] = random.choice(MEDICAL_STAGES_ORTHOPEDICS)
            config['role_generation']['medical_scenario'] = get_scenario_ortho(age_group, gender)
            config['role_generation']['doctor_gender'] = doctor_gender
            config['role_generation']['doctor_age'] = doctor_age
            config['role_generation']['doctor_name'] = doctor_name
            config['role_generation']['patient_gender'] = gender
            config['role_generation']['patient_age'] = age
            config['role_generation']['patient_name'] = patient_name
            
            config['role_guidance']['medical_case'] = "The case should belong to the Department of Orthopedics (Fracture). Ensure scenarios are medically plausible and age-appropriate (e.g., no occupational injuries for children)."
            config['audio_reference']['doctor_audio'] = doctor_audio
            config['audio_reference']['patient_audio'] = patient_audio
            
            # Scores
            for key in ['flow', 'concerns', 'attentive', 'warmth', 'respect']:
                config['global_scores'][key] = score
            
            if score == 1:
                config['wiser_counts']['empathic_response'] = [0, 1]
                config['wiser_counts']['open_ended_question'] = [1, 2]
                # Less relevant for fracture but keep consistent
                config['wiser_counts']['social_determinants_of_health'] = [0, 0] 
                config['wiser_counts']['weight'] = [0, 0]
            elif score == 5:
                config['wiser_counts']['empathic_response'] = [4, 6]
                config['wiser_counts']['open_ended_question'] = [4, 6]
                # Can be relevant for fracture recovery (nutrition, home support)
                config['wiser_counts']['social_determinants_of_health'] = [0, 1] 
                config['wiser_counts']['weight'] = [0, 0]
            
            # Filename
            pat_id_match = re.search(r'(\d+)', patient_audio)
            pat_id = pat_id_match.group(1) if pat_id_match else "unknown"
            exp_name = f"fracture_{region}_{age_group}_{gender}_pat{pat_id}_score{score}"
            
            # Output dir
            ortho_dir = root_dir / "experiments" / "orthopedics"
            ortho_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = ortho_dir / f"exp_{exp_name}.yml"
            save_yaml(config, save_path)
            count_ortho += 1
            
    print(f"Generated {count_ob} Obstetrics experiments in {root_dir}/experiments/obstetrics")
    print(f"Generated {count_ortho} Orthopedics experiments in {root_dir}/experiments/orthopedics")

if __name__ == "__main__":
    main()

