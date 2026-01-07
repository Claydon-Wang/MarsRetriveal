import argparse
import logging
import os
import sys
import csv
from datetime import datetime
from typing import List
import torch
from configs.config_base import load_task_config, load_model_config, merge_configs
from tools.utils import random_seed, _merge_args, _configure_logging, _validate_inputs, _silence_noisy_loggers

from datasets.utils import build_dataset
from evaluators.utils import build_evaluator
from models.utils import build_image_encoder, build_text_encoder
from queries.utils import build_query
from retrievers.utils import build_retriever


def _parse_args():
    parser = argparse.ArgumentParser(description="Retrieval benchmark runner")
    # Basic experiment config
    parser.add_argument("--project_name", type=str, default=None, help="Project name for logging.")
    parser.add_argument("--task_name", type=str, default=None, help="Task name for module dispatch.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (used under logs/).")
    parser.add_argument("--task_config", type=str, required=True, help="Task config name.")
    parser.add_argument("--model_config", type=str, required=True, help="Model config name.")
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

    # Evaluation
    parser.add_argument("--ground_truth_csv", type=str, default=None, help="Optional GT CSV for evaluation.")
    parser.add_argument("--radius_deg", type=float, default=None, help="Radius in degrees for coverage metric.")
    parser.add_argument("--eval_max_k", type=int, default=None, help="Max K to scan during evaluation.")

    return parser.parse_args()


def main():
    _silence_noisy_loggers()
    args_dynamic = _parse_args()
    task_cfg = load_task_config(args_dynamic.task_config)
    model_cfg = load_model_config(args_dynamic.model_config)
    args = merge_configs(task_cfg, model_cfg)
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
    logging.info("Using image encoder: %s (type=%s)", image_encoder.__class__.__name__, args.image_encoder_type)
    text_encoder = None
    if args.task_name == "cross_modal_matching" or args_dynamic.query_mode in ("text", "hybrid"):
        text_encoder = build_text_encoder(args, device)
        if text_encoder is not None:
            logging.info("Using text encoder: %s (type=%s)", text_encoder.__class__.__name__, args.text_encoder_type)
        else:
            logging.info("Text encoder disabled.")

    delta = args.delta_degree if args.delta_degree is not None else 0.2
    if args.task_name == "cross_modal_matching":
        delta = None
    database = build_dataset(args, image_encoder, text_encoder=text_encoder, delta=delta)
    retriever = build_retriever(args, database)

    query_mode = args_dynamic.query_mode
    if args.task_name == "cross_modal_matching":
        query_mode = "cross_modal"
    query_images = args_dynamic.query_images or []
    query_text = args_dynamic.query_text
    _validate_inputs(query_mode, query_images, query_text, task_name=args.task_name)

    if args.task_name == "cross_modal_matching":
        results = retriever.search()
        df_results = retriever.to_dataframe(results)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for suffix, df in df_results.items():
            csv_name = f"{timestamp}_{suffix}.csv"
            csv_path = os.path.join(output_dir, csv_name)
            df.to_csv(csv_path, index=False)
            logging.info("Saved retrieval results to %s", csv_path)
    else:
        query_features = build_query(
            args,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            query_mode=query_mode,
            query_images=query_images,
            query_name=query_text,
        )

        if args.task_name == "landform_retrieval":
            query_features, query_names = query_features
            results = retriever.search(query_features, query_names=query_names)
        else:
            results = retriever.search(query_features)
        df_results = retriever.to_dataframe(results)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if args.task_name != "landform_retrieval":
            csv_name = f"{timestamp}.csv"
            csv_path = os.path.join(output_dir, csv_name)
            df_results.to_csv(csv_path, index=False)
            logging.info("Saved retrieval results to %s", csv_path)

    evaluator = build_evaluator(args)
    if args.task_name == "cross_modal_matching":
        eval_summary = evaluator.evaluate(database, label=args.task_name) if evaluator else {}
    else:
        eval_summary = evaluator.evaluate(df_results, label=query_mode) if evaluator else {}
    if evaluator:
        if eval_summary:
            summary_log = eval_summary.get("best", eval_summary)
            logging.info("Evaluation summary: %s", summary_log)
        headers, row = evaluator.summary(args, args_dynamic, eval_summary)

        summary_dir = os.path.join(args.logs, args.task_name or "default_task")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "summary.csv")
        write_header = not os.path.exists(summary_path)
        with open(summary_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)
        logging.info("Appended run summary to %s", summary_path)
        if args.task_name == "landform_retrieval":
            per_class = eval_summary.get("per_class", {}) if eval_summary else {}
            if per_class:
                metrics_path = os.path.join(output_dir, f"{timestamp}.csv")
                with open(metrics_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["class", "mAP", "recall@1", "recall@5", "recall@10", "precision@10"])
                    for cls_name, metrics in sorted(per_class.items()):
                        writer.writerow(
                            [
                                cls_name,
                                round(metrics.get("mAP", 0.0) * 100, 2),
                                round(metrics.get("recall@1", 0.0) * 100, 2),
                                round(metrics.get("recall@5", 0.0) * 100, 2),
                                round(metrics.get("recall@10", 0.0) * 100, 2),
                                round(metrics.get("precision@10", 0.0) * 100, 2),
                            ]
                        )
                logging.info("Saved per-class metrics to %s", metrics_path)
    else:
        logging.info("No evaluator available; skipping summary file.")

    logging.info("Benchmark run complete.")


if __name__ == "__main__":
    main()
