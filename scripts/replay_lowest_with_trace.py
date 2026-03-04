#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("examples"), list):
        return data["examples"]
    raise ValueError(f"Unsupported JSON schema in {path}")


def _parse_image_paths(record: Dict[str, Any]) -> List[str]:
    rows = record.get("retrieved_image_paths", [])
    image_paths: List[str] = []
    for row in rows:
        if not isinstance(row, str):
            continue
        if "|" in row:
            path = row.split("|", 1)[1].strip()
        else:
            path = row.strip()
        if path:
            image_paths.append(path)
    return image_paths


def _run_one(agent: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
    images = _parse_image_paths(sample)
    query = sample.get("query", "")
    answer, trace = agent.run_agent(query=query, images_path=images, return_trace=True)
    return {
        "uid": sample.get("uid"),
        "query": query,
        "reference_answer": sample.get("reference_answer"),
        "previous_response": sample.get("response"),
        "new_response": answer,
        "meta_info": sample.get("meta_info", {}),
        "eval_result": sample.get("eval_result", {}),
        "candidate_images": images,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay lowest-score ViDoSeek samples and dump step-by-step agent traces."
    )
    parser.add_argument(
        "--input",
        default="data/ViDoSeek/results/vidorag_openbmb-EVisRAG-7B_lowest10.json",
        help="Input lowest-score sample file (JSON).",
    )
    parser.add_argument(
        "--output",
        default="data/ViDoSeek/results/vidorag_openbmb-EVisRAG-7B_lowest10_trace.jsonl",
        help="Output path for trace results (JSONL).",
    )
    parser.add_argument(
        "--model_name",
        default="openbmb/EVisRAG-7B",
        help="Generation model name.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit of samples to replay.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_records(input_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    from llms.llm import LLM
    from vidorag_agents import ViDoRAG_Agents

    vlm = LLM(model_name=args.model_name)
    agents = ViDoRAG_Agents(vlm)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in tqdm(samples, desc="Replay lowest samples"):
            try:
                result = _run_one(agents, sample)
                result["status"] = "ok"
            except Exception as e:
                result = {
                    "uid": sample.get("uid"),
                    "query": sample.get("query", ""),
                    "status": "error",
                    "error": str(e),
                }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved trace to: {output_path}")


if __name__ == "__main__":
    main()
