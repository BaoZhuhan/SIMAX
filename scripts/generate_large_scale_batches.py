#!/usr/bin/env python3
"""Generate large-scale SIMAX batches for five-doctors and ruio settings."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data.csv"
TEMPLATE_YML = ROOT / "experiment.yml"
OUT_ROOT = ROOT / "experiments"

DOCTOR_POOLS = {
    "five-doctors": [
        {
            "doctor_audio": "common_voice_en_19949787.mp3",
            "doctor_region": "US",
            "doctor_gender": "male",
            "doctor_age": "45",
        },
        {
            "doctor_audio": "common_voice_en_22465949.mp3",
            "doctor_region": "US",
            "doctor_gender": "female",
            "doctor_age": "43",
        },
        {
            "doctor_audio": "common_voice_en_17281510.mp3",
            "doctor_region": "India",
            "doctor_gender": "male",
            "doctor_age": "46",
        },
        {
            "doctor_audio": "common_voice_en_20113591.mp3",
            "doctor_region": "India",
            "doctor_gender": "female",
            "doctor_age": "41",
        },
        {
            "doctor_audio": "common_voice_en_39477765.mp3",
            "doctor_region": "India",
            "doctor_gender": "male",
            "doctor_age": "44",
        },
    ],
    "ruio": [
        {
            "doctor_audio": "common_voice_en_ruio_99999999.mp3",
            "doctor_region": "US",
            "doctor_gender": "male",
            "doctor_age": "25",
        }
    ],
}

EXTRA_EXCLUDED_PATIENTS = {
    # Match the historical 888 batch by excluding ruio voice from patient pool.
    "five-doctors": {"common_voice_en_ruio_99999999.mp3"},
    "ruio": set(),
}

NAME_LIBRARY = {
    "US": {
        "M": ["James", "John", "Robert", "Michael", "William", "David"],
        "F": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan"],
    },
    "UK": {
        "M": ["Oliver", "George", "Harry", "Jack", "Jacob", "Charlie"],
        "F": ["Olivia", "Amelia", "Isla", "Ava", "Emily", "Mia"],
    },
    "India": {
        "M": ["Aarav", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna"],
        "F": ["Aadya", "Ananya", "Diya", "Isha", "Riya", "Aditi"],
    },
    "AUS": {
        "M": ["Oliver", "Noah", "Jack", "William", "Leo", "Henry"],
        "F": ["Charlotte", "Olivia", "Mia", "Ava", "Amelia", "Isla"],
    },
    "Africa": {
        "M": ["Kwame", "Kofi", "Ade", "Chike", "Tunde", "Emeka"],
        "F": ["Aisha", "Fatima", "Zainab", "Nia", "Zola", "Imani"],
    },
    "Other": {
        "M": ["Alex", "Sam", "Jordan", "Casey", "Taylor", "Morgan"],
        "F": ["Alex", "Sam", "Jordan", "Casey", "Taylor", "Morgan"],
    },
}

LAST_NAME_LIBRARY = {
    "US": ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller"],
    "India": ["Sharma", "Patel", "Singh", "Gupta", "Reddy", "Iyer"],
    "Other": ["Taylor", "Morgan", "Lee", "Jordan", "Casey", "Riley"],
}

MEDICAL_STAGES_OBSTETRICS = [
    "First Trimester Visit (Initial)",
    "Routine Prenatal Visit (20 weeks)",
    "Third Trimester Check-up",
    "Postpartum Follow-up (6 weeks)",
    "Urgent Visit (Pain/Bleeding)",
]

MEDICAL_STAGES_ORTHOPEDICS = [
    "Initial diagnosis (Emergency Room)",
    "Follow-up consultation (2 weeks post-injury)",
    "Treatment planning (Surgery vs Conservative)",
    "Post-treatment review (Cast removal/Rehabilitation)",
]

MEDICAL_STAGES_RHEUM = [
    "Initial diagnosis (Rheumatology clinic)",
    "Follow-up consultation (symptom monitoring)",
    "Medication monitoring (DMARD/Biologic)",
    "Flare management visit",
]

SCENARIOS_OB = [
    "Routine check-up, no complaints",
    "Complaining of nausea and vomiting",
    "Concerned about decreased fetal movement",
    "Vaginal bleeding concern",
    "Abdominal pain and cramping",
    "High blood pressure symptoms (headache, vision changes)",
    "Gestational diabetes screening discussion",
    "Birth plan discussion",
    "Breastfeeding questions",
]

SCENARIOS_ORTHO = {
    "child": [
        "Fell from a slide at the playground",
        "Injured while playing soccer",
        "Fell out of a bunk bed",
        "Tripped while running in the park",
    ],
    "teen": [
        "Sports injury during a basketball game",
        "Skateboard accident",
        "Bicycle accident on the way to school",
        "Twisted ankle while hiking",
    ],
    "adult": [
        "Slip and fall on wet floor at work",
        "Car accident resulting in potential fracture",
        "Fell from a ladder while doing home repairs",
        "Tripped on an uneven sidewalk",
    ],
    "senior": [
        "Slip and fall in the bathroom",
        "Tripped over a rug at home",
        "Fell while gardening",
        "Lost balance while walking down stairs",
    ],
}

SCENARIOS_RHEUM = {
    "adult": [
        "Chronic joint pain follow-up",
        "New-onset joint swelling and morning stiffness",
        "Rheumatoid arthritis flare",
        "Medication (DMARD) efficacy and side-effect review",
        "Suspected inflammatory arthritis referral discussion",
    ],
    "senior": [
        "Polyarthralgia and mobility concerns",
        "Medication reconciliation and fall risk",
        "Pain management for osteoarthritis",
    ],
}

SCORES = [1, 3, 5]

# Use all WISER labels available in resources/codebooks/wiser_codes.json.
WISER_SCORE_PROFILES = {
    1: {
        "empathic_opportunity": [2, 4],
        "empathic_response": [0, 1],
        "empathic_statement": [0, 1],
        "im_sorry": [2, 4],
        "open_ended_question": [0, 1],
        "reflective_statement": [0, 1],
        "elicit_questions": [0, 0],
        "weight": [0, 0],
        "social_determinants_of_health": [0, 0],
        "asking_questions": [0, 1],
        "assertive_responses": [0, 1],
    },
    3: {
        "empathic_opportunity": [2, 4],
        "empathic_response": [2, 4],
        "empathic_statement": [1, 2],
        "im_sorry": [1, 2],
        "open_ended_question": [3, 5],
        "reflective_statement": [2, 3],
        "elicit_questions": [1, 2],
        "weight": [0, 1],
        "social_determinants_of_health": [0, 1],
        "asking_questions": [1, 2],
        "assertive_responses": [1, 2],
    },
    5: {
        "empathic_opportunity": [1, 3],
        "empathic_response": [4, 7],
        "empathic_statement": [2, 4],
        "im_sorry": [0, 1],
        "open_ended_question": [5, 8],
        "reflective_statement": [3, 5],
        "elicit_questions": [2, 3],
        "weight": [1, 2],
        "social_determinants_of_health": [1, 2],
        "asking_questions": [2, 4],
        "assertive_responses": [2, 4],
    },
}

EXPECTED_COUNTS = {
    "five-doctors": {"ob": 69, "fracture": 546, "rheum": 273, "total": 888},
    "ruio": {"ob": 75, "fracture": 561, "rheum": 288, "total": 924},
}


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_random_name(region: str, gender: str) -> str:
    gender_key = "F" if gender == "female" else "M"
    first_names = NAME_LIBRARY.get(region, NAME_LIBRARY["Other"]).get(
        gender_key, NAME_LIBRARY["Other"]["M"]
    )
    return random.choice(first_names)


def get_doctor_name(region: str, gender: str, doctor_audio: str) -> str:
    gender_key = "F" if gender == "female" else "M"
    first_pool = NAME_LIBRARY.get(region, NAME_LIBRARY["Other"]).get(
        gender_key, NAME_LIBRARY["Other"]["M"]
    )
    last_pool = LAST_NAME_LIBRARY.get(region, LAST_NAME_LIBRARY["Other"])
    match = re.search(r"(\d+)", doctor_audio)
    numeric_id = int(match.group(1)) if match else 0
    first = first_pool[numeric_id % len(first_pool)]
    last = last_pool[(numeric_id // len(first_pool)) % len(last_pool)]
    return f"Dr. {first} {last}"


def parse_patient_metadata(csv_path: Path):
    metadata = {}
    col_map = {
        1: ("child", "male"),
        2: ("child", "female"),
        3: ("teen", "male"),
        4: ("teen", "female"),
        5: ("adult", "male"),
        6: ("adult", "female"),
        7: ("senior", "male"),
        8: ("senior", "female"),
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    current_region = "Other"
    for row in rows[2:]:
        if not row:
            continue

        if row[0].strip():
            raw_region = row[0].strip()
            if "United States" in raw_region:
                current_region = "US"
            elif "England" in raw_region:
                current_region = "UK"
            elif "India" in raw_region:
                current_region = "India"
            elif "Australian" in raw_region:
                current_region = "AUS"
            elif "African" in raw_region:
                current_region = "Africa"
            else:
                current_region = "Other"

        for col_idx in range(1, 9):
            if col_idx >= len(row):
                break
            cell = row[col_idx].strip()
            if not cell.endswith(".mp3"):
                continue

            age_group, gender = col_map[col_idx]
            age = "35"
            if age_group == "child":
                age = str(random.randint(8, 12))
            elif age_group == "teen":
                age = str(random.randint(13, 19))
            elif age_group == "adult":
                age = str(random.randint(30, 50))
            elif age_group == "senior":
                age = str(random.randint(65, 80))

            metadata[cell] = {
                "region": current_region,
                "age": age,
                "age_group": age_group,
                "gender": gender,
            }
    return metadata


def pick_rotating_doctor(pool, cursor: int, patient_audio: str):
    pool_size = len(pool)
    for _ in range(pool_size):
        doctor = deepcopy(pool[cursor % pool_size])
        cursor += 1
        if doctor["doctor_audio"] != patient_audio:
            doctor["doctor_name"] = get_doctor_name(
                doctor.get("doctor_region", "Other"),
                doctor["doctor_gender"],
                doctor["doctor_audio"],
            )
            return doctor, cursor
    raise ValueError(f"No doctor audio available different from patient audio: {patient_audio}")


def build_base_case(template: dict, doctor: dict, patient_audio: str, patient_meta: dict):
    cfg = deepcopy(template)
    cfg.setdefault("role_generation", {})
    cfg.setdefault("role_guidance", {})
    cfg.setdefault("audio_reference", {})
    cfg.setdefault("global_scores", {})
    cfg.setdefault("wiser_counts", {})
    cfg.setdefault("patient_scores", {})

    cfg["role_generation"]["doctor_gender"] = doctor["doctor_gender"]
    cfg["role_generation"]["doctor_age"] = doctor["doctor_age"]
    cfg["role_generation"]["doctor_name"] = doctor["doctor_name"]
    cfg["role_generation"]["patient_gender"] = patient_meta["gender"]
    cfg["role_generation"]["patient_age"] = patient_meta["age"]
    cfg["role_generation"]["patient_name"] = get_random_name(
        patient_meta["region"], patient_meta["gender"]
    )

    cfg["audio_reference"]["doctor_audio"] = doctor["doctor_audio"]
    cfg["audio_reference"]["patient_audio"] = patient_audio
    return cfg


def apply_score_profile(cfg: dict, score: int):
    for key in ["flow", "concerns", "attentive", "warmth", "respect"]:
        cfg["global_scores"][key] = score
    cfg["wiser_counts"] = deepcopy(WISER_SCORE_PROFILES[score])


def render_name(specialty: str, region: str, age_group: str, gender: str, patient_audio: str, score: int):
    pat_id_match = re.search(r"(\d+)", patient_audio)
    pat_id = pat_id_match.group(1) if pat_id_match else "unknown"
    return f"exp_{specialty}_{region}_{age_group}_{gender}_pat{pat_id}_score{score}.yml"


def generate_batch(batch: str, clean: bool, seed: int):
    random.seed(seed)
    template = load_yaml(TEMPLATE_YML)
    out_dir = OUT_ROOT / batch
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_meta = parse_patient_metadata(DATA_CSV)
    pool = DOCTOR_POOLS[batch]
    excluded = {d["doctor_audio"] for d in pool} | EXTRA_EXCLUDED_PATIENTS[batch]
    patient_meta = {k: v for k, v in patient_meta.items() if k not in excluded}
    patient_items = sorted(patient_meta.items())

    cursor = {"ob": 0, "fracture": 0, "rheum": 0}
    counts = {"ob": 0, "fracture": 0, "rheum": 0}

    for patient_audio, meta in patient_items:
        region = meta["region"]
        age_group = meta["age_group"]
        gender = meta["gender"]

        # Obstetrics: adult female only.
        if age_group == "adult" and gender == "female":
            doctor, cursor["ob"] = pick_rotating_doctor(pool, cursor["ob"], patient_audio)
            case = build_base_case(template, doctor, patient_audio, meta)
            case["role_generation"]["medical_stage"] = random.choice(MEDICAL_STAGES_OBSTETRICS)
            case["role_generation"]["medical_scenario"] = random.choice(SCENARIOS_OB)
            case["role_guidance"][
                "medical_case"
            ] = "The case should belong to the Department of Obstetrics. Ensure scenarios are medically plausible and age-appropriate."
            for score in SCORES:
                cfg = deepcopy(case)
                apply_score_profile(cfg, score)
                out_name = render_name("ob", region, age_group, gender, patient_audio, score)
                save_yaml(cfg, out_dir / out_name)
                counts["ob"] += 1

        # Orthopedics (fracture): all age groups and genders.
        doctor, cursor["fracture"] = pick_rotating_doctor(pool, cursor["fracture"], patient_audio)
        case = build_base_case(template, doctor, patient_audio, meta)
        case["role_generation"]["medical_stage"] = random.choice(MEDICAL_STAGES_ORTHOPEDICS)
        case["role_generation"]["medical_scenario"] = random.choice(
            SCENARIOS_ORTHO.get(age_group, SCENARIOS_ORTHO["adult"])
        )
        case["role_guidance"][
            "medical_case"
        ] = "The case should belong to the Department of Orthopedics (Fracture). Ensure scenarios are medically plausible and age-appropriate."
        for score in SCORES:
            cfg = deepcopy(case)
            apply_score_profile(cfg, score)
            out_name = render_name("fracture", region, age_group, gender, patient_audio, score)
            save_yaml(cfg, out_dir / out_name)
            counts["fracture"] += 1

        # Rheumatology: adult and senior only.
        if age_group in {"adult", "senior"}:
            doctor, cursor["rheum"] = pick_rotating_doctor(pool, cursor["rheum"], patient_audio)
            case = build_base_case(template, doctor, patient_audio, meta)
            case["role_generation"]["medical_stage"] = random.choice(MEDICAL_STAGES_RHEUM)
            case["role_generation"]["medical_scenario"] = random.choice(
                SCENARIOS_RHEUM.get(age_group, SCENARIOS_RHEUM["adult"])
            )
            case["role_guidance"][
                "medical_case"
            ] = "The case should belong to the Department of Rheumatology. Ensure scenarios are medically plausible and age-appropriate."
            for score in SCORES:
                cfg = deepcopy(case)
                apply_score_profile(cfg, score)
                out_name = render_name("rheum", region, age_group, gender, patient_audio, score)
                save_yaml(cfg, out_dir / out_name)
                counts["rheum"] += 1

    total = sum(counts.values())
    expected = EXPECTED_COUNTS[batch]
    ok = counts == {k: expected[k] for k in ("ob", "fracture", "rheum")} and total == expected["total"]

    print(f"[{batch}] output_dir={out_dir}")
    print(f"[{batch}] counts: ob={counts['ob']} fracture={counts['fracture']} rheum={counts['rheum']} total={total}")
    print(f"[{batch}] expected: ob={expected['ob']} fracture={expected['fracture']} rheum={expected['rheum']} total={expected['total']}")
    if not ok:
        raise RuntimeError(f"Generated counts for {batch} do not match expected totals.")


def main():
    parser = argparse.ArgumentParser(description="Generate 888+924 large-scale experiment batches.")
    parser.add_argument(
        "--batch",
        choices=["five-doctors", "ruio", "all"],
        default="all",
        help="Which batch to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing experiments/<batch> before generation.",
    )
    args = parser.parse_args()

    batches = ["five-doctors", "ruio"] if args.batch == "all" else [args.batch]
    for batch in batches:
        generate_batch(batch, clean=args.clean, seed=args.seed)


if __name__ == "__main__":
    main()
