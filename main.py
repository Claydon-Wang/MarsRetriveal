import argparse
import logging
import os
import sys
import csv
from datetime import datetime
from typing import List
import torch
from configs.config_base import load_static_config
from tools.utils import random_seed, _merge_args, _configure_logging, _validate_inputs

from datasets.utils import build_dataset
from evaluators.utils import build_evaluator
from models.utils import build_image_encoder, build_text_encoder
from queries.utils import build_query
from retrievers.utils import build_retriever


def _parse_args():
    parser = argparse.ArgumentParser(description="Retrieval benchmark runner")
    # Basic experiment config
    parser.add_argument("--project_name", type=str, default=None, help="Project name for logging.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (used under logs/).")
    parser.add_argument("--config_name", type=str, required=True, help="Static config name.")
    parser.add_argument("--model_name", type=str, default=None, help="Model name override (supports prefixes like openclip/ or dinov3/).")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained tag override.")
    parser.add_argument("--resume_post_train", type=str, default=None, help="Checkpoint for pretrained weights.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store retrieval outputs.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K retrieval results to keep.")
    parser.add_argument("--db_dir", type=str, default=None, help="Optional database directory override.")

    # Query specification
    parser.add_argument("--query_mode", type=str, required=True, help="Query mode: image | text | hybrid.")
    parser.add_argument("--query_images", nargs="*", default=None, help="Paths to query images.")
    parser.add_argument("--query_text", type=str, default=None, help="Text query for text/multimodal modes.")

    # Encoders
    parser.add_argument("--image_encoder_type", type=str, default=None, help="Image encoder type (e.g., openclip, dinov3).")
    parser.add_argument("--text_encoder_type", type=str, default=None, help="Text encoder type (e.g., openclip, none).")
    parser.add_argument("--dinov3_checkpoint", type=str, default=None, help="Local repo path for DINOv3 hubconf.")

    # Evaluation
    parser.add_argument("--ground_truth_csv", type=str, default=None, help="Optional GT CSV for evaluation.")
    parser.add_argument("--radius_deg", type=float, default=None, help="Radius in degrees for coverage metric.")
    parser.add_argument("--eval_max_k", type=int, default=None, help="Max K to scan during evaluation.")

    return parser.parse_args()


def main():
    args_dynamic = _parse_args()
    args = load_static_config(args_dynamic.config_name, type="retrieval")
    args = _merge_args(args, args_dynamic)
    if args_dynamic.query_mode in ("text", "hybrid") and args.text_encoder_type == "none":
        raise ValueError("Text or hybrid query modes require a text encoder (text_encoder_type must not be 'none').")

    output_dir = args.output_dir
    log_file = _configure_logging(output_dir)
    logging.info("Writing logs to %s", log_file)

    args.output_dir = output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed(args.seed)
    logging.info("Using device: %s", device)

    image_encoder = build_image_encoder(args, device)
    text_encoder = None
    if args_dynamic.query_mode in ("text", "hybrid"):
        text_encoder = build_text_encoder(args, device)

    database = build_dataset(args, image_encoder, delta=0.2)
    retriever = build_retriever(args, database)

    query_mode = args_dynamic.query_mode
    query_images = args_dynamic.query_images or []
    query_text = args_dynamic.query_text
    _validate_inputs(query_mode, query_images, query_text)

    query_features = build_query(
        args,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        query_mode=query_mode,
        query_images=query_images,
        query_name=query_text,
    )

    results = retriever.search(query_features)
    df_results = retriever.to_dataframe(results)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_name = f"{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df_results.to_csv(csv_path, index=False)
    logging.info("Saved retrieval results to %s", csv_path)

    evaluator = build_evaluator(
        args.ground_truth_csv, radius_deg=args.radius_deg, max_k=args.eval_max_k
    )
    eval_summary = evaluator.evaluate(df_results, label=query_mode) if evaluator else {}
    best_f1 = None
    auprc_score = None
    if eval_summary:
        logging.info("Evaluation summary: %s", eval_summary["best"])
        best = eval_summary.get("best") or {}
        best_f1 = (
            round(best.get("f1", 0.0) * 100, 2),
            round(best.get("precision", 0.0) * 100, 2),
            round(best.get("recall", 0.0) * 100, 2),
            best.get("k"),
        )
        auprc_score = round(eval_summary.get("auprc", 0.0) * 100, 2) if "auprc" in eval_summary else None

    # Append summary
    summary_dir = os.path.join(args.logs, args.name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)
    with open(summary_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "query_text",
                    "query_mode",
                    "model_name",
                    "pretrained",
                    "resume_post_train",
                    "f1",
                    "precision",
                    "recall",
                    "auprc",
                    "k_at_best",
                ]
            )
        writer.writerow(
            [
                args_dynamic.query_text or "",
                args_dynamic.query_mode,
                args.model,
                args.pretrained,
                args.resume_post_train or "",
                best_f1[0] if best_f1 else "",
                best_f1[1] if best_f1 else "",
                best_f1[2] if best_f1 else "",
                auprc_score if auprc_score is not None else "",
                best_f1[3] if best_f1 else "",
            ]
        )
    logging.info("Appended run summary to %s", summary_path)

    logging.info("Benchmark run complete.")


if __name__ == "__main__":
    main()
