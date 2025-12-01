from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


@dataclass
class DatasetSpec:
    name: str
    csv: str
    question_column: str = "question"
    answer_column: str = "answer"
    image_column: str | None = None
    split_column: str | None = None
    default_split: str = "test"
    question_type_column: str | None = None
    default_question_type: str = "open"
    language: str = "en"
    id_column: str | None = None
    id_prefix: str | None = None


def load_specs(config_path: Path) -> list[DatasetSpec]:
    data = yaml.safe_load(config_path.read_text())
    items = data.get("datasets", []) if data else []
    specs: list[DatasetSpec] = []
    for item in items:
        specs.append(DatasetSpec(**item))
    return specs


def is_yes_no(question: str, answer: str, question_type: str | None) -> bool:
    q = question.lower()
    a = answer.lower()
    if question_type and "yes" in question_type.lower():
        return True
    if q.startswith(("is ", "are ", "do ", "does ", "can ", "should ")):
        return True
    if a in {"yes", "no", "true", "false"}:
        return True
    return False


def normalize_answer(text: str) -> str:
    return " ".join(str(text).split())


def normalize_yes_no(text: str) -> str:
    t = text.strip().lower()
    if t in {"true", "yes", "y"}:
        return "yes"
    if t in {"false", "no", "n"}:
        return "no"
    return t


def create_options(
    correct_answer: str,
    candidates: Iterable[str],
    rng: random.Random,
    yes_no: bool,
) -> tuple[dict[str, str], str]:
    if yes_no:
        ordered = ["yes", "no", "not sure", "cannot determine"]
        correct_norm = normalize_yes_no(correct_answer)
        if correct_norm in {"yes", "no"}:
            mapping = dict(zip("ABCD", ordered))
            label = next(k for k, v in mapping.items() if normalize_yes_no(v) == correct_norm)
            return mapping, label

    options = []
    seen = set()
    ca = normalize_answer(correct_answer)
    options.append(ca)
    seen.add(ca.lower())
    pool = [normalize_answer(c) for c in candidates if str(c).strip()]
    rng.shuffle(pool)
    for cand in pool:
        if cand.lower() in seen:
            continue
        options.append(cand)
        seen.add(cand.lower())
        if len(options) == 4:
            break
    fillers = ["not sure", "cannot determine", "other finding", "unclear"]
    for filler in fillers:
        if len(options) >= 4:
            break
        if filler.lower() in seen:
            continue
        options.append(filler)
        seen.add(filler.lower())
    while len(options) < 4:
        stub = f"option_{len(options)+1}"
        options.append(stub)
    rng.shuffle(options)
    mapping = dict(zip("ABCD", options))
    label = next(k for k, v in mapping.items() if v == ca)
    return mapping, label


def build_dataset(config_path: Path, output_path: Path, image_root: Path | None, seed: int) -> Path:
    rng = random.Random(seed)
    specs = load_specs(config_path)
    rows: list[dict[str, str]] = []
    for spec in specs:
        csv_path = (config_path.parent / spec.csv).resolve()
        df = pd.read_csv(csv_path)
        answer_pool = [str(v) for v in df[spec.answer_column].dropna().tolist()]
        for idx, (_, row) in enumerate(df.iterrows()):
            if pd.isna(row[spec.answer_column]) or pd.isna(row[spec.question_column]):
                continue
            answer = normalize_answer(row[spec.answer_column])
            question = normalize_answer(row[spec.question_column])
            qtype = spec.default_question_type
            if spec.question_type_column and spec.question_type_column in df.columns and not pd.isna(row[spec.question_type_column]):
                qtype = str(row[spec.question_type_column]).strip()
            yes_no = is_yes_no(question, answer, qtype)
            option_mapping, correct_label = create_options(answer, answer_pool, rng, yes_no)
            image_value = None
            if spec.image_column and spec.image_column in df.columns and not pd.isna(row[spec.image_column]):
                image_value = str(row[spec.image_column])
                if image_root:
                    image_value = str((image_root / image_value).as_posix())
            rid = None
            if spec.id_column and spec.id_column in df.columns and not pd.isna(row[spec.id_column]):
                rid = str(row[spec.id_column])
            if not rid:
                prefix = spec.id_prefix or spec.name
                rid = f"{prefix}_{idx:06d}"
            split = spec.default_split
            if spec.split_column and spec.split_column in df.columns and not pd.isna(row[spec.split_column]):
                split = str(row[spec.split_column])
            rows.append(
                {
                    "id": rid,
                    "dataset": spec.name,
                    "split": split,
                    "image_path": image_value or "",
                    "question": question,
                    "correct_answer": answer,
                    "question_type": qtype,
                    "option_a": option_mapping["A"],
                    "option_b": option_mapping["B"],
                    "option_c": option_mapping["C"],
                    "option_d": option_mapping["D"],
                    "correct_label": correct_label,
                    "language": spec.language,
                    "options": " ".join(f"{k}) {v}" for k, v in option_mapping.items()),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sycophancy challenge dataset from raw metadata.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.config, args.output, args.image_root, args.seed)


if __name__ == "__main__":
    main()
