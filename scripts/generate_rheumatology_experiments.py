#!/usr/bin/env python3
"""Generate Rheumatology experiments (large-scale) into experiments/rheumatology/.
Default: adult and senior patients, both genders, scores [1,3,5].
"""
import csv
import yaml
import random
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data.csv"
TEMPLATE_YML = ROOT / "experiment.yml"
OUT_DIR = ROOT / "experiments" / "rheumatology"

MEDICAL_STAGES_RHEUM = [
    "Initial diagnosis (Rheumatology clinic)",
    "Follow-up consultation (symptom monitoring)",
    "Medication monitoring (DMARD/Biologic)",
    "Flare management visit",
]

SCENARIO_TEMPLATES_RHEUM = {
    "adult": {
        "All": [
            "Chronic joint pain follow-up",
            "New-onset joint swelling and morning stiffness",
            "Rheumatoid arthritis flare",
            "Medication (DMARD) efficacy and side-effect review",
            "Suspected inflammatory arthritis referral discussion",
        ]
    },
    "senior": {
        "All": [
            "Polyarthralgia and mobility concerns",
            "Medication reconciliation and fall risk",
            "Pain management for osteoarthritis",
        ]
    }
}

SCORES = [1, 3, 5]


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_patient_metadata(csv_path):
    """Return mapping of patient_audio -> metadata dict (region, age, age_group, gender)"""
    metadata = {}
    col_map = {
        1: ("child", "male"), 2: ("child", "female"),
        3: ("teen", "male"), 4: ("teen", "female"),
        5: ("adult", "male"), 6: ("adult", "female"),
        7: ("senior", "male"), 8: ("senior", "female")
    }
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    current_region = "Other"
    for i in range(2, len(rows)):
        row = rows[i]
        if not row: continue
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
            if not cell or not cell.endswith('.mp3'): continue
            age_group, gender = col_map.get(col_idx, ("adult", "male"))
            age = "35"
            if age_group == "child": age = str(random.randint(8, 12))
            elif age_group == "teen": age = str(random.randint(13, 19))
            elif age_group == "adult": age = str(random.randint(30, 50))
            elif age_group == "senior": age = str(random.randint(65, 80))
            metadata[cell] = {
                "region": current_region,
                "age": age,
                "age_group": age_group,
                "gender": gender
            }
    return metadata


def main():
    ROOT = Path(__file__).resolve().parents[1]
    template_yml = TEMPLATE_YML
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_CSV.exists():
        print("data.csv not found; aborting")
        return

    patient_meta = parse_patient_metadata(DATA_CSV)
    print(f"Found {len(patient_meta)} patient voices")

    # Load base template
    if template_yml.exists():
        base_config = load_yaml(template_yml)
    else:
        base_config = {}

    count = 0
    doctor_audio = base_config.get('audio_reference', {}).get('doctor_audio', 'common_voice_en_39477765.mp3')
    doctor_name = "Dr. John Smith"
    doctor_age = '45'
    doctor_gender = 'male'

    for patient_audio, meta in patient_meta.items():
        region = meta['region']
        age = meta['age']
        age_group = meta['age_group']
        gender = meta['gender']

        # Only adult or senior for rheumatology
        if age_group not in ['adult', 'senior']:
            continue

        patient_name = f"Patient_{re.sub(r'[^0-9]','', patient_audio)[:6]}"

        for score in SCORES:
            config = base_config.copy()
            # Ensure nested keys exist
            config.setdefault('role_generation', {})
            config.setdefault('role_guidance', {})
            config.setdefault('audio_reference', {})
            config.setdefault('global_scores', {})
            config.setdefault('wiser_counts', {})
            config.setdefault('patient_scores', {})

            config['role_generation']['medical_stage'] = random.choice(MEDICAL_STAGES_RHEUM)
            # pick scenario based on age_group
            scenarios = SCENARIO_TEMPLATES_RHEUM.get(age_group, {}).get('All', [])
            if not scenarios:
                scenarios = SCENARIO_TEMPLATES_RHEUM['adult']['All']
            config['role_generation']['medical_scenario'] = random.choice(scenarios)
            config['role_generation']['doctor_gender'] = doctor_gender
            config['role_generation']['doctor_age'] = doctor_age
            config['role_generation']['doctor_name'] = doctor_name
            config['role_generation']['patient_gender'] = gender
            config['role_generation']['patient_age'] = age
            config['role_generation']['patient_name'] = patient_name

            config['role_guidance']['medical_case'] = "The case should belong to the Department of Rheumatology. Ensure scenarios are medically plausible and age-appropriate."
            config['audio_reference']['doctor_audio'] = doctor_audio
            config['audio_reference']['patient_audio'] = patient_audio

            for key in ['flow', 'concerns', 'attentive', 'warmth', 'respect']:
                config['global_scores'][key] = score

            # WISER defaults by score
            if score == 1:
                config['wiser_counts']['empathic_response'] = [0, 1]
                config['wiser_counts']['open_ended_question'] = [1, 2]
                config['wiser_counts']['social_determinants_of_health'] = [0, 0]
            elif score == 5:
                config['wiser_counts']['empathic_response'] = [4, 6]
                config['wiser_counts']['open_ended_question'] = [4, 6]
                config['wiser_counts']['social_determinants_of_health'] = [0, 1]

            pat_id_match = re.search(r'(\d+)', patient_audio)
            pat_id = pat_id_match.group(1) if pat_id_match else 'unknown'
            exp_name = f"rheum_{region}_{age_group}_{gender}_pat{pat_id}_score{score}"
            save_path = out_dir / f"exp_{exp_name}.yml"
            save_yaml(config, save_path)
            count += 1

    print(f"Generated {count} Rheumatology experiments in {out_dir}")

if __name__ == '__main__':
    main()
