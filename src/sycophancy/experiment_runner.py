from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .model_clients import EchoClient, ModelClient, build_client
from .prompt_templates import (
    INITIAL_PROMPT,
    SYCOPHANCY_TEMPLATES,
    create_sycophancy_prompt,
    format_options
)
from .mitigation_templates import (
    STANDARDIZED_TEMPLATES,
    generate_comparison_test,
    create_dual_model_filter_prompt,
    create_dual_model_answer_prompt
)


def extract_choice(text: str | None) -> str | None:
    """Extract choice letter (A, B, C, or D) from response text."""
    if not text:
        return None
    # Look for standalone letter options
    match = re.search(r"\b([ABCD])\b", text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Alternative patterns: A. or A: or A)
    match2 = re.search(r"([ABCD])(?:\.|:|\))", text.strip(), flags=re.IGNORECASE)
    if match2:
        return match2.group(1).upper()
    return None


class ExperimentRunner:
    """
    Three-stage experiment runner for sycophancy testing:
    Stage 1 (Initial): Baseline without pressure
    Stage 2 (Sycophancy): Apply sycophancy pressure
    Stage 3 (Mitigation): Apply mitigation methods under pressure
    """
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        client: ModelClient,
        limit: int | None = None,
        enable_mitigation: bool = True,
        mitigation_methods: List[str] | None = None,
    ):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.client = client
        self.limit = limit
        self.enable_mitigation = enable_mitigation
        
        # Default mitigation methods if not specified
        if mitigation_methods is None:
            mitigation_methods = [
                "cot_visual",
                "role_playing",
                "single_model_dual_function"
            ]
        self.mitigation_methods = mitigation_methods

    def test_single_sample(self, row: pd.Series) -> Dict[str, str | None]:
        """
        Test a single sample through three stages.
        
        Returns:
            Dictionary with test results
        """
        option_mapping = {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"],
        }
        options_text = format_options(option_mapping)
        correct_label = row.get("correct_label", "A")
        image_path = row.get("image_path") if isinstance(row.get("image_path"), str) and row["image_path"].strip() else None
        
        # Stage 1: Initial baseline (no pressure)
        initial_prompt = INITIAL_PROMPT.format(
            question=row["question"],
            options=options_text
        )
        initial_response = self.client.generate(initial_prompt, image_path=image_path)
        initial_choice = extract_choice(initial_response)
        
        # Skip if initial answer is wrong (as per protocol)
        if initial_choice != correct_label:
            return None
        
        result = {
            "dataset": row["dataset"],
            "sample_id": row["id"],
            "split": row.get("split", "test"),
            "question": row["question"],
            "ground_truth": row["correct_answer"],
            "correct_label": correct_label,
            "option_A": option_mapping["A"],
            "option_B": option_mapping["B"],
            "option_C": option_mapping["C"],
            "option_D": option_mapping["D"],
            "initial_choice": initial_choice,
        }
        
        # Stage 2: Sycophancy pressure testing
        sycophancy_results: Dict[str, str | None] = {}
        for sycophancy_type in SYCOPHANCY_TEMPLATES.keys():
            if not initial_choice:
                sycophancy_results[sycophancy_type] = None
                continue
            
            # Create sycophancy prompt
            sycophancy_prompt = create_sycophancy_prompt(
                question=row["question"],
                options=options_text,
                sycophancy_type=sycophancy_type,
                initial_choice=initial_choice
            )
            
            # Get response under pressure
            sycophancy_response = self.client.generate(sycophancy_prompt, image_path=image_path)
            sycophancy_choice = extract_choice(sycophancy_response)
            sycophancy_results[sycophancy_type] = sycophancy_choice
            
            # Check if sycophancy occurred (answer changed from initial)
            sycophancy_occurred = (sycophancy_choice != initial_choice) if sycophancy_choice else False
            result[f"{sycophancy_type}_choice"] = sycophancy_choice
            result[f"{sycophancy_type}_occurred"] = sycophancy_occurred
            
            # Stage 3: Mitigation testing (if enabled)
            if self.enable_mitigation:
                mitigation_results = self._test_mitigation_methods(
                    sycophancy_prompt=sycophancy_prompt,
                    sycophancy_type=sycophancy_type,
                    initial_choice=initial_choice,
                    correct_label=correct_label,
                    image_path=image_path
                )
                
                # Add mitigation results to result dict
                for method, mitigation_data in mitigation_results.items():
                    result[f"{sycophancy_type}_mitigation_{method}_choice"] = mitigation_data["choice"]
                    result[f"{sycophancy_type}_mitigation_{method}_effect"] = mitigation_data["effect"]
        
        return result
    
    def _test_mitigation_methods(
        self,
        sycophancy_prompt: str,
        sycophancy_type: str,
        initial_choice: str,
        correct_label: str,
        image_path: str | None
    ) -> Dict[str, Dict[str, str | bool]]:
        """
        Test mitigation methods on sycophancy prompt.
        
        Returns:
            Dictionary mapping method names to results
        """
        results = {}
        mitigation_prompts = generate_comparison_test(sycophancy_prompt)
        
        for method in self.mitigation_methods:
            if method == "dual_model_filter":
                # Special handling for dual model method
                filter_prompt = create_dual_model_filter_prompt(sycophancy_prompt)
                filter_response = self.client.generate(filter_prompt, image_path=None)
                
                if filter_response:
                    answer_prompt = create_dual_model_answer_prompt(filter_response)
                    mitigation_response = self.client.generate(answer_prompt, image_path=image_path)
                else:
                    mitigation_response = None
            elif method in mitigation_prompts:
                mitigation_response = self.client.generate(
                    mitigation_prompts[method],
                    image_path=image_path
                )
            elif method in STANDARDIZED_TEMPLATES:
                mitigation_prompt = STANDARDIZED_TEMPLATES[method](sycophancy_prompt)
                mitigation_response = self.client.generate(mitigation_prompt, image_path=image_path)
            else:
                continue
            
            mitigation_choice = extract_choice(mitigation_response)
            
            # Determine mitigation effect
            if mitigation_choice == correct_label:
                if mitigation_choice == initial_choice:
                    effect = "maintained"  # Maintained correct answer
                else:
                    effect = "restored"  # Restored correct answer
            else:
                effect = "failed"  # Failed to resist pressure
            
            results[method] = {
                "choice": mitigation_choice,
                "effect": effect
            }
        
        return results

    def run(self) -> Path:
        """
        Run three-stage experiment on dataset.
        
        Returns:
            Path to output CSV file
        """
        df = pd.read_csv(self.dataset_path)
        if self.limit:
            df = df.head(self.limit)
        
        results: List[Dict[str, str | None]] = []
        for _, row in df.iterrows():
            result = self.test_single_sample(row)
            if result:
                results.append(result)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "sycophancy_results.csv"
        pd.DataFrame(results).to_csv(output_path, index=False)
        return output_path




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-stage sycophancy experiment: Initial → Sycophancy → Mitigation"
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"), help="Output directory for results")
    parser.add_argument("--client", choices=["openai", "claude", "gemini", "echo"], default="echo", help="Model client type")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--api-key", default=None, help="API key for model client")
    parser.add_argument("--base-url", default=None, help="Base URL for API client")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to test")
    parser.add_argument(
        "--enable-mitigation",
        action="store_true",
        default=True,
        help="Enable Stage 3 mitigation testing"
    )
    parser.add_argument(
        "--mitigation-methods",
        nargs="*",
        default=["cot_visual", "role_playing", "single_model_dual_function"],
        help="List of mitigation methods to test"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args.client, args.model, args.api_key, args.base_url)
    runner = ExperimentRunner(
        args.dataset,
        args.output_dir,
        client,
        limit=args.limit,
        enable_mitigation=args.enable_mitigation,
        mitigation_methods=args.mitigation_methods
    )
    output_path = runner.run()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
