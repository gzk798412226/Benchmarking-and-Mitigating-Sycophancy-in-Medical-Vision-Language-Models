# Benchmarking and Mitigating Sycophancy in Medical Vision-Language Models

This repository provides code for benchmarking and mitigating sycophancy (excessive agreement) in medical vision-language models. It implements a three-stage testing protocol to evaluate how models respond to various forms of pressure and how mitigation strategies can help maintain accurate medical diagnoses.

## Overview

The codebase implements:

1. **Seven types of sycophancy pressure**: Expert correction, emotional manipulation, social consensus, ethical/economic pressure, stylistic mimicry, authority-based commands, and technological self-doubt
2. **Three-stage testing protocol**: 
   - Stage 1 (Initial): Baseline without pressure
   - Stage 2 (Sycophancy): Apply pressure to test vulnerability
   - Stage 3 (Mitigation): Apply mitigation methods under pressure
3. **Multiple mitigation strategies**: Including CoT, visual cues, pressure resistance, role-playing, and single-model dual-function methods

## Setup

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, can also pass via command line):
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # For Claude
export GOOGLE_API_KEY="your-key-here"     # For Gemini
```

## Prepare Data

We include a processed CSV file: `data/sycophancy_challenge_datasets_fixed.csv`. This file references image paths by relative names; you need to download the corresponding images and place them under your preferred structure.

If you want to rebuild from your own raw metadata:
1. Download source datasets and place metadata under `data/raw/`
2. Copy `config/datasets.example.yaml` to `config/datasets.yaml`
3. Point each entry to your CSV files and image paths
4. Build the challenge set:
```bash
PYTHONPATH=src python -m sycophancy.data_builder \
  --config config/datasets.yaml \
  --output data/processed/sycophancy_challenge.csv \
  --image-root data/raw
```

## Running Experiments

### Basic Usage

Run the three-stage experiment with default settings:
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY"
```

### Command Line Options

- `--dataset`: Path to dataset CSV file (required)
- `--output-dir`: Output directory for results (default: `runs`)
- `--client`: Model client type - `openai`, `claude`, `gemini`, or `echo` (default: `echo`)
- `--model`: Model name (default: `gpt-4o`)
- `--api-key`: API key (optional if set via environment variable)
- `--base-url`: Base URL for API (optional, mainly for OpenAI-compatible APIs)
- `--limit`: Limit number of samples to test (optional)
- `--enable-mitigation`: Enable Stage 3 mitigation testing (default: True)
- `--mitigation-methods`: List of mitigation methods to test (default: `cot_visual role_playing single_model_dual_function`)

### Example: Testing with Different Models

**OpenAI (GPT-4o):**
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY"
```

**Claude:**
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client claude \
  --model claude-3-opus-20240229 \
  --api-key "$ANTHROPIC_API_KEY"
```

**Gemini:**
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client gemini \
  --model gemini-pro-vision \
  --api-key "$GOOGLE_API_KEY"
```

**Echo mode (for testing without API calls):**
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client echo \
  --limit 10
```

### Custom Mitigation Methods

Test specific mitigation methods:
```bash
PYTHONPATH=src python -m sycophancy.experiment_runner \
  --dataset data/sycophancy_challenge_datasets_fixed.csv \
  --output-dir runs \
  --client openai \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY" \
  --mitigation-methods cot_visual single_model_dual_function role_playing
```

Available mitigation methods:
- `std_cot`: Standardized Chain-of-Thought
- `std_visual`: Standardized Visual Cues
- `std_resistance`: Standardized Pressure Resistance
- `cot_visual`: CoT + Visual Cues combination
- `cot_resistance`: CoT + Pressure Resistance combination
- `visual_resistance`: Visual Cues + Pressure Resistance combination
- `triple_combo`: CoT + Visual + Resistance triple combination
- `role_playing`: Role-playing as independent medical AI
- `single_model_dual_function`: Single model with dual function (filter + answer)
- `dual_model_filter`: Dual model method (requires two API calls)
- And more (see `src/sycophancy/mitigation_templates.py`)

## Three-Stage Testing Protocol

### Stage 1: Initial (Baseline)
- Tests model with clean prompt (no pressure)
- Only samples where initial answer is correct proceed to next stages
- Establishes baseline performance

### Stage 2: Sycophancy (Pressure Testing)
- Applies one of seven sycophancy pressure types:
  1. **Expert Correction**: Strong expert authority pressure
  2. **Emotional**: Emotional manipulation from patient perspective
  3. **Social Consensus**: Peer pressure and social isolation
  4. **Ethical/Economic**: Moral and financial pressure
  5. **Stylistic Mimicry**: Subtle suggestion to change answer
  6. **Authority-Based**: Hierarchical command pressure
  7. **Technological Doubt**: Technical uncertainty pressure
- Records whether model changes answer (sycophancy occurred)

### Stage 3: Mitigation (Resistance Testing)
- Applies mitigation methods to sycophancy prompts
- Evaluates effectiveness:
  - `restored`: Mitigation restored correct answer
  - `maintained`: Mitigation maintained correct answer
  - `failed`: Mitigation failed to resist pressure

## Output Format

Results are saved to `{output_dir}/sycophancy_results.csv` with the following columns:

- Basic info: `dataset`, `sample_id`, `split`, `question`, `ground_truth`, `correct_label`
- Options: `option_A`, `option_B`, `option_C`, `option_D`
- Stage 1: `initial_choice`
- Stage 2: For each sycophancy type:
  - `{type}_choice`: Answer under pressure
  - `{type}_occurred`: Whether sycophancy occurred (boolean)
- Stage 3: For each sycophancy type and mitigation method:
  - `{type}_mitigation_{method}_choice`: Answer with mitigation
  - `{type}_mitigation_{method}_effect`: Mitigation effect (`restored`/`maintained`/`failed`)

## Code Structure

- `src/sycophancy/prompt_templates.py`: Seven sycophancy prompt templates
- `src/sycophancy/mitigation_templates.py`: Mitigation method templates
- `src/sycophancy/experiment_runner.py`: Three-stage experiment runner
- `src/sycophancy/model_clients.py`: API client implementations
- `src/sycophancy/data_builder.py`: Dataset construction utilities

## Requirements

See `requirements.txt` for full list. Key dependencies:
- pandas
- openai (for OpenAI API)
- anthropic (for Claude API, optional)
- google-generativeai (for Gemini API, optional)

## Citation

If you use this code in your research, please cite our paper (citation to be added).

## License

[License information to be added]
