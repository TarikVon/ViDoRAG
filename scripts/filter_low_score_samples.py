#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        # Standard JSONL: one JSON object per line.
        records = []
        jsonl_ok = True
        for line_no, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                jsonl_ok = False
                break
        if jsonl_ok:
            return records

        # Fallback: handle concatenated multi-line JSON objects.
        records = []
        decoder = json.JSONDecoder()
        idx = 0
        n = len(content)
        while idx < n:
            while idx < n and content[idx].isspace():
                idx += 1
            if idx >= n:
                break
            try:
                obj, next_idx = decoder.raw_decode(content, idx)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Unable to decode concatenated JSON content near char {idx}: {e}"
                ) from e
            records.append(obj)
            idx = next_idx
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # support {'examples': [...]} style files
        if isinstance(data.get("examples"), list):
            return data["examples"]
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _get_score(record: Dict[str, Any]) -> float:
    eval_result = record.get("eval_result") or {}
    score = eval_result.get("score")
    if score is None:
        return float("inf")
    try:
        return float(score)
    except (TypeError, ValueError):
        return float("inf")


def _score_counter(records: Iterable[Dict[str, Any]]) -> Counter:
    c = Counter()
    for rec in records:
        score = _get_score(rec)
        if score != float("inf"):
            c[score] += 1
    return c


def _format_one(record: Dict[str, Any], rank: int) -> str:
    uid = record.get("uid", "")
    query = record.get("query", "")
    ref = record.get("reference_answer", "")
    pred = record.get("response", "")
    eval_result = record.get("eval_result") or {}
    score = eval_result.get("score", "N/A")
    passing = eval_result.get("passing", "N/A")
    judge = eval_result.get("judge", "")
    meta = record.get("meta_info") or {}
    source_type = meta.get("source_type", "")
    query_type = meta.get("query_type", "")
    retrieval_metrics = record.get("retrieval_metrics") or {}

    lines = [
        f"[{rank}] uid={uid}",
        f"score={score}, passing={passing}, source_type={source_type}, query_type={query_type}",
        f"query: {query}",
        f"reference_answer: {ref}",
        f"response: {pred}",
    ]

    if judge:
        lines.append(f"judge: {judge}")

    if retrieval_metrics:
        keys = ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"]
        metric_view = ", ".join(
            f"{k}={retrieval_metrics[k]}" for k in keys if k in retrieval_metrics
        )
        if metric_view:
            lines.append(f"retrieval_metrics: {metric_view}")

    return "\n".join(lines)

def _extract_image_paths(record: Dict[str, Any]) -> List[str]:
    recall_results = record.get("recall_results")
    if not recall_results:
        return []

    if isinstance(recall_results, dict):
        nodes = recall_results.get("source_nodes", [])
    elif isinstance(recall_results, list):
        nodes = recall_results
    else:
        return []

    image_paths: List[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_obj = node.get("node", {})
        metadata = node_obj.get("metadata", {})
        path = metadata.get("file_name") or metadata.get("filename")
        if path:
            image_paths.append(path)
    return image_paths

def _to_readable_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uid": record.get("uid"),
        "query": record.get("query"),
        "reference_answer": record.get("reference_answer"),
        "meta_info": record.get("meta_info", {}),
        "eval_result": record.get("eval_result", {}),
        "response": record.get("response"),
        "retrieved_image_paths": _extract_image_paths(record),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter lowest-score samples from eval results.")
    parser.add_argument("--input", required=True, help="Path to result file (.jsonl or .json)")
    parser.add_argument("--top_n", type=int, default=20, help="How many lowest-score samples to show")
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Optional upper bound for score filter (e.g. --min_score 2 keeps score<=2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to save filtered samples as JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = _load_records(input_path)
    if not records:
        print("No records found.")
        return

    scored = [r for r in records if _get_score(r) != float("inf")]
    missing_score = len(records) - len(scored)

    scored.sort(key=lambda r: (_get_score(r), r.get("uid", "")))
    if args.min_score is not None:
        scored = [r for r in scored if _get_score(r) <= args.min_score]

    selected = scored[: max(args.top_n, 0)]
    readable_selected = [_to_readable_record(rec) for rec in selected]

    print(f"Input: {input_path}")
    print(f"Total records: {len(records)}")
    print(f"Records with eval score: {len(records) - missing_score}")
    print(f"Records missing eval score: {missing_score}")

    dist = _score_counter(records)
    if dist:
        ordered = ", ".join(f"{k:g}:{dist[k]}" for k in sorted(dist.keys()))
        print(f"Score distribution: {ordered}")

    print(f"Selected lowest-score records: {len(selected)}")
    print("=" * 80)

    for idx, rec in enumerate(selected, 1):
        print(_format_one(rec, idx))
        print("-" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(readable_selected, f, ensure_ascii=False, indent=2)
        print(f"Saved filtered samples to: {out_path}")


if __name__ == "__main__":
    main()
