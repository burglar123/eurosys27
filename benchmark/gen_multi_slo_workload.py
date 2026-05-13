#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a reproducible multi-SLO workload JSONL file.

Each line in the output JSONL file is one request:
{
  "request_id": "...",
  "arrival_offset_sec": 0.123,
  "category": "coding" | "chat" | "summarization",
  "slo_class": "tight" | "normal" | "loose",
  "slo_tpot_ms": 30.0,
  "per_request_gamma": 4,
  "max_tokens": 256,
  "prompt": "..."
}

Example:

python benchmark/gen_multi_slo_workload.py \
  --rps 4.0 \
  --duration-sec 120 \
  --mix 0.6,0.2,0.2 \
  --baseline-tpot-ms 26.3 \
  --slo-scale 1.2 \
  --max-tokens 256 \
  --seed 0 \
  --out benchmark/workloads/rps4.0_mix0.6_0.2_0.2_seed0.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def load_jsonl_prompts(path: str, role: str) -> List[str]:
    """
    Load prompts from different JSONL schemas.

    Supported common formats:
    - {"turns": ["..."]}
    - {"prompt": "..."}
    - {"instruction": "...", "input": "..."}
    - {"article": "...", "highlights": "..."} for summarization
    - {"document": "..."}
    - {"text": "..."}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please provide the correct path for role={role}."
        )

    prompts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON line {line_no} in {path}")
                continue

            prompt = extract_prompt(obj, role)
            if prompt:
                prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {path} for role={role}")

    return prompts


def extract_prompt(obj: Dict[str, Any], role: str) -> Optional[str]:
    """
    Convert a raw dataset item to a prompt string.
    """
    if role == "summarization":
        if obj.get("article"):
            return "Summarize the following article:\n\n" + str(obj["article"]).strip()
        if obj.get("document"):
            return "Summarize the following document:\n\n" + str(obj["document"]).strip()
        if obj.get("text"):
            return "Summarize the following text:\n\n" + str(obj["text"]).strip()

    if role == "chat":
        if obj.get("instruction"):
            instruction = str(obj["instruction"]).strip()
            extra_input = str(obj.get("input", "")).strip()
            if extra_input:
                return instruction + "\n\nInput:\n" + extra_input
            return instruction

    if obj.get("turns") and isinstance(obj["turns"], list) and len(obj["turns"]) > 0:
        return str(obj["turns"][0]).strip()

    if obj.get("prompt"):
        return str(obj["prompt"]).strip()

    if obj.get("instruction"):
        instruction = str(obj["instruction"]).strip()
        extra_input = str(obj.get("input", "")).strip()
        if extra_input:
            return instruction + "\n\nInput:\n" + extra_input
        return instruction

    if obj.get("question"):
        return str(obj["question"]).strip()

    if obj.get("text"):
        return str(obj["text"]).strip()

    return None


def generate_poisson_arrivals(
    rps: float,
    duration_sec: Optional[float],
    num_requests: Optional[int],
    rng: random.Random,
) -> List[float]:
    """
    Generate Poisson arrivals.

    If num_requests is provided, generate exactly num_requests arrivals.
    Otherwise, generate arrivals until duration_sec.
    """
    if rps <= 0:
        raise ValueError("--rps must be positive")

    if duration_sec is None and num_requests is None:
        raise ValueError("Either --duration-sec or --num-requests must be provided")

    arrivals: List[float] = []
    t = 0.0

    if num_requests is not None:
        for _ in range(num_requests):
            t += rng.expovariate(rps)
            arrivals.append(t)
        return arrivals

    assert duration_sec is not None
    while t < duration_sec:
        t += rng.expovariate(rps)
        if t < duration_sec:
            arrivals.append(t)

    return arrivals


def load_arrival_offsets(path: str, max_requests: Optional[int] = None) -> List[float]:
    """
    Load arrival offsets from a file.

    Supported formats:
    1. JSONL:
       {"arrival_offset_sec": 0.1}
       {"timestamp": 0.2}
       {"arrival_ts": 0.3}

    2. Plain text:
       0.1
       0.2
       0.3

    3. CSV-like first column:
       0.1,xxx
       0.2,xxx

    All values are treated as offsets in seconds.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arrival trace file not found: {path}")

    offsets: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if max_requests is not None and len(offsets) >= max_requests:
                break

            line = line.strip()
            if not line:
                continue

            value: Optional[float] = None

            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    for key in ("arrival_offset_sec", "timestamp", "arrival_ts", "time"):
                        if key in obj:
                            value = float(obj[key])
                            break
                except Exception as exc:
                    print(f"[WARN] Skip invalid JSON arrival line {line_no}: {exc}")
                    continue
            else:
                first_col = line.split(",")[0].strip()
                try:
                    value = float(first_col)
                except ValueError:
                    print(f"[WARN] Skip invalid arrival line {line_no}: {line}")
                    continue

            if value is not None:
                offsets.append(value)

    if not offsets:
        raise ValueError(f"No valid arrival offsets loaded from {path}")

    offsets.sort()
    base = offsets[0]
    offsets = [x - base for x in offsets]

    return offsets


def parse_mix(mix_str: str) -> List[float]:
    parts = [float(x.strip()) for x in mix_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--mix must have exactly 3 numbers: coding,chat,summarization")
    if any(x < 0 for x in parts):
        raise ValueError("--mix values must be non-negative")
    s = sum(parts)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"--mix must sum to 1.0, got {s}")
    return parts


def choose_prompt(prompts: List[str], rng: random.Random) -> Tuple[str, int]:
    idx = rng.randrange(len(prompts))
    return prompts[idx], idx


def build_request_id(prefix: str, idx: int) -> str:
    safe_prefix = (
        prefix.replace(".", "p")
        .replace(",", "_")
        .replace("/", "_")
        .replace(" ", "")
    )
    return f"{safe_prefix}-{idx:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible multi-SLO workload JSONL for nano-PEARL."
    )

    # Dataset paths.
    parser.add_argument(
        "--humaneval",
        default="benchmark/data/HumanEval.jsonl",
        help="JSONL file for coding requests.",
    )
    parser.add_argument(
        "--chat",
        default="benchmark/data/GSM8K.jsonl",
        help="JSONL file for normal/chat-like requests.",
    )
    parser.add_argument(
        "--cnndm",
        default="benchmark/data/CNNDM.jsonl",
        help="JSONL file for summarization requests.",
    )

    # Arrival pattern.
    parser.add_argument("--rps", type=float, required=True, help="Average request rate.")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=120.0,
        help="Trace duration in seconds. Ignored if --num-requests is set.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Generate exactly this many requests. Overrides --duration-sec.",
    )
    parser.add_argument(
        "--arrival-trace",
        type=str,
        default=None,
        help="Optional arrival trace file. If set, use it instead of Poisson arrivals.",
    )

    # Workload composition.
    parser.add_argument(
        "--mix",
        type=str,
        default="0.6,0.2,0.2",
        help="Category mix: coding,chat,summarization. Must sum to 1.",
    )
    parser.add_argument("--seed", type=int, default=0)

    # SLO settings.
    parser.add_argument(
        "--baseline-tpot-ms",
        type=float,
        default=None,
        help="Baseline TPOT in ms. Used for tight SLO = slo_scale * baseline_tpot_ms.",
    )
    parser.add_argument(
        "--tight-slo-tpot-ms",
        type=float,
        default=None,
        help="Directly set tight SLO TPOT. If set, overrides --baseline-tpot-ms and --slo-scale.",
    )
    parser.add_argument(
        "--slo-scale",
        type=float,
        default=1.2,
        help="Tight SLO scale relative to baseline TPOT.",
    )
    parser.add_argument(
        "--normal-slo-tpot-ms",
        type=float,
        default=50.0,
        help="Normal/chat TPOT SLO in ms.",
    )
    parser.add_argument(
        "--loose-slo-tpot-ms",
        type=float,
        default=150.0,
        help="Loose/summarization TPOT SLO in ms.",
    )

    # Generation settings stored per request.
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tight-gamma", type=int, default=4)
    parser.add_argument("--normal-gamma", type=int, default=4)
    parser.add_argument("--loose-gamma", type=int, default=4)

    # Output.
    parser.add_argument("--out", required=True, help="Output workload JSONL path.")
    parser.add_argument(
        "--request-id-prefix",
        default=None,
        help="Optional request id prefix. If not set, a prefix is generated from rps/seed.",
    )

    args = parser.parse_args()

    mix = parse_mix(args.mix)

    if args.tight_slo_tpot_ms is not None:
        tight_slo_tpot_ms = args.tight_slo_tpot_ms
    else:
        if args.baseline_tpot_ms is None:
            raise ValueError(
                "Either --tight-slo-tpot-ms or --baseline-tpot-ms must be provided."
            )
        tight_slo_tpot_ms = args.slo_scale * args.baseline_tpot_ms

    arrival_rng = random.Random(args.seed)
    sample_rng = random.Random(args.seed + 1)

    print("[INFO] Loading datasets...")
    coding_prompts = load_jsonl_prompts(args.humaneval, role="coding")
    chat_prompts = load_jsonl_prompts(args.chat, role="chat")
    sum_prompts = load_jsonl_prompts(args.cnndm, role="summarization")

    print(f"[INFO] Loaded coding prompts: {len(coding_prompts)}")
    print(f"[INFO] Loaded chat prompts: {len(chat_prompts)}")
    print(f"[INFO] Loaded summarization prompts: {len(sum_prompts)}")

    if args.arrival_trace:
        arrivals = load_arrival_offsets(args.arrival_trace, max_requests=args.num_requests)
        if args.duration_sec is not None:
            arrivals = [x for x in arrivals if x < args.duration_sec]
        if args.num_requests is not None:
            arrivals = arrivals[: args.num_requests]
    else:
        arrivals = generate_poisson_arrivals(
            rps=args.rps,
            duration_sec=args.duration_sec if args.num_requests is None else None,
            num_requests=args.num_requests,
            rng=arrival_rng,
        )

    categories = ["coding", "chat", "summarization"]
    workload: List[Dict[str, Any]] = []

    if args.request_id_prefix is None:
        prefix = f"rps{args.rps}-seed{args.seed}"
    else:
        prefix = args.request_id_prefix

    for idx, arrival_offset_sec in enumerate(arrivals):
        category = sample_rng.choices(categories, weights=mix, k=1)[0]

        if category == "coding":
            prompt, source_index = choose_prompt(coding_prompts, sample_rng)
            slo_class = "tight"
            slo_tpot_ms = tight_slo_tpot_ms
            per_request_gamma = args.tight_gamma
            source_dataset = os.path.basename(args.humaneval)

        elif category == "chat":
            prompt, source_index = choose_prompt(chat_prompts, sample_rng)
            slo_class = "normal"
            slo_tpot_ms = args.normal_slo_tpot_ms
            per_request_gamma = args.normal_gamma
            source_dataset = os.path.basename(args.chat)

        else:
            prompt, source_index = choose_prompt(sum_prompts, sample_rng)
            slo_class = "loose"
            slo_tpot_ms = args.loose_slo_tpot_ms
            per_request_gamma = args.loose_gamma
            source_dataset = os.path.basename(args.cnndm)

        workload.append(
            {
                "request_id": build_request_id(prefix, idx),
                "arrival_offset_sec": float(arrival_offset_sec),
                "category": category,
                "slo_class": slo_class,
                "slo_tpot_ms": float(slo_tpot_ms),
                "per_request_gamma": int(per_request_gamma),
                "max_tokens": int(args.max_tokens),
                "prompt": prompt,
                "source_dataset": source_dataset,
                "source_index": source_index,
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for req in workload:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    counts = Counter(req["category"] for req in workload)
    slo_counts = Counter(req["slo_class"] for req in workload)

    meta = {
        "rps": args.rps,
        "duration_sec": args.duration_sec,
        "num_requests": len(workload),
        "mix": {
            "coding": mix[0],
            "chat": mix[1],
            "summarization": mix[2],
        },
        "seed": args.seed,
        "baseline_tpot_ms": args.baseline_tpot_ms,
        "slo_scale": args.slo_scale,
        "tight_slo_tpot_ms": tight_slo_tpot_ms,
        "normal_slo_tpot_ms": args.normal_slo_tpot_ms,
        "loose_slo_tpot_ms": args.loose_slo_tpot_ms,
        "max_tokens": args.max_tokens,
        "tight_gamma": args.tight_gamma,
        "normal_gamma": args.normal_gamma,
        "loose_gamma": args.loose_gamma,
        "category_counts": dict(counts),
        "slo_class_counts": dict(slo_counts),
        "workload_path": args.out,
    }

    meta_path = args.out + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved workload: {args.out}")
    print(f"[OK] Saved metadata: {meta_path}")
    print(f"[OK] Number of requests: {len(workload)}")
    print(f"[OK] Category counts: {dict(counts)}")
    print(f"[OK] SLO-class counts: {dict(slo_counts)}")


if __name__ == "__main__":
    main()