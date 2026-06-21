# SIMAX

SIMAX: A Scalable and Interpretable Framework for Multi-Fidelity and Annotated Clinician–Patient Dialogue Simulation


## Overview

SIMAX generates realistic clinician-patient dialogues through a 4-step pipeline:

1. **Role Generation** — Generate doctor/patient profiles with controlled personality traits
2. **Dialogue Generation** — Create multi-turn clinical dialogues following medical workflow
3. **Audio Synthesis** — Convert dialogues to natural speech using MOSS-TTSD
4. **Wet Mixing** — Add environmental sounds (keyboard, door, background) for realism

## Requirements

- Python 3.10+
- CUDA-compatible GPU (for vLLM and MOSS-TTSD)
- Conda environment manager

## Installation

```bash
# Clone the repository
git clone https://github.com/BaoZhuhan.git
cd SIMAX

# Create conda environment
conda env create -f environment.yml
conda activate simax

# Install MOSS-TTSD (external dependency)
pip install moss-ttsd

# Install vLLM for GPT inference
pip install vllm
```

## Quick Start

### 1. Download Models

Download the required models:
- **GPT Model**: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- **MOSS-TTSD**: [MOSS-TTSD v0.7](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.7)

### 2. Configure Paths

Edit `config.yml` to set your model paths:
```yaml
paths:
  gpt_model_path: ./models/gpt-oss-20b
  moss_ttsd:
    model_path: ./models/moss-ttsd-v0.7
```

### 3. Prepare Audio References

Download Common Voice dataset and set paths in `config.yml`:
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

### 4. Run Pipeline

```bash
# Copy example experiment config
cp experiment.yml.example experiment.yml

# Run single experiment
bash job.sh.example my-experiment config.yml experiment.yml

# Or run step by step
python main.py --config config.yml --experiment experiment.yml --exp_id test --step role
python main.py --config config.yml --experiment experiment.yml --exp_id test --step dialogue
python main.py --config config.yml --experiment experiment.yml --exp_id test --step audio
python main.py --config config.yml --experiment experiment.yml --exp_id test --step wet
```

## Project Structure

```
simax_nejm/
├── main.py                    # Main entry point
├── config.yml                 # Global configuration
├── experiment.yml.example     # Example experiment config
├── job.sh.example             # Example SLURM job script
├── environment.yml            # Conda environment
├── src/
│   ├── prompts.py             # LLM prompt templates
│   ├── json_schemas.py        # JSON schema definitions
│   ├── services.py            # vLLM service management
│   ├── text/
│   │   ├── role_doc.py        # Role document generation
│   │   └── dialogue_gen.py    # Dialogue generation
│   ├── audio/
│   │   ├── preprocess.py      # Audio preprocessing
│   │   ├── synthesize.py      # TTS synthesis
│   │   └── wet.py             # Environmental audio mixing
│   └── utils/
│       └── io.py              # I/O utilities
├── resources/
│   ├── templates/             # JSON templates for profiles
│   └── audio_assets/          # Environmental sound files
└── scripts/
    ├── generate_experiments.py
    ├── generate_rheumatology_experiments.py
    └── generate_large_scale_batches.py
```

## Experiment Configuration

See `experiment.yml.example` for the full configuration format. Key sections:

- **role_generation**: Medical stage, scenario, gender/age constraints
- **audio_reference**: Voice reference audio files
- **text_generation**: Temperature, max tokens, dialogue length
- **tts_generation**: TTS model settings
- **wet_generation**: Environmental audio mixing settings
- **Codebook scores or counts**: Behavior control settings

## Codebook Note

The behavioral codebooks (WISER, Global, Bias) used for controlling dialogue generation are **not publicly available**. The framework will gracefully handle their absence and use default behavior.We also encourage you to use custom behavioral codebooks.

## Citation

If you use SIMAX in your research, please cite:

```bibtex
will coming soon
```

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

- [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD) for the text-to-speech model
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for voice reference data
- [Freesound](https://freesound.org/) for environmental audio assets
