import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

# Task types in `sy1998/MLVU_dev`.
TASK_TYPES = ["anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"]


hf_home = os.getenv("HF_HOME", "./~/.cache/huggingface")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "mlvu.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]

fast_cache_dir = 'data/video_instruction_tuning/mlvu/languagebind_feat'
slow_cache_dir = 'data/video_instruction_tuning/mlvu/languagebind_feat'


def mlvu_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_name"]
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    
    feature_name = doc["video_name"].split('.')[0] + '.pt'
    slow_path = os.path.join(slow_cache_dir, feature_name)
    fast_path = os.path.join(fast_cache_dir, feature_name)
    # import ipdb
    # ipdb.set_trace() # 
    
    return [slow_path, fast_path, video_path]


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    instruction = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
    question = doc["question"].strip()
    candidates = doc.get("candidates", None)
    if isinstance(candidates, list) and len(candidates) > 0:
        option_lines = "\n".join([f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(candidates)])
        prompt = f"{instruction}\n{question}\n{option_lines}\nAnswer:"
    else:
        prompt = f"{instruction}\n{question}\nAnswer:"
    return prompt


def extract_choice_letter(s: str) -> str:
    s = (s or "").strip()
    m = re.search(r"(?i)(?:^|\b|\()([ABCD])(?:\b|\)|\.)", s)
    return m.group(1).upper() if m else s[:1].upper()


def mlvu_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_choice_letter(pred)

    task_type = doc.get("task_type", "unknown")
    question_id = doc.get("question_id", doc.get("question", ""))
    data_dict = {"question_id": question_id, "task_type": task_type, "pred_answer": pred_ans, "answer": doc["answer"]}

    return {f"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    all_types = set(TASK_TYPES)
    for result in results:
        all_types.add(result.get("task_type", "unknown"))

    category2score = {t: {"correct": 0, "answered": 0} for t in sorted(all_types)}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    for task_cate in sorted(all_types):
        v = category2score[task_cate]
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {100 * v['correct'] / v['answered'] if v['answered'] > 0 else 0 : .1f}%")

    total_correct = 0
    total_answered = 0
    for k, v in category2score.items():
        total_correct += v["correct"]
        total_answered += v["answered"]
    eval_logger.info(f"Overall Performance: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    return 100 * total_correct / total_answered if total_answered > 0 else 0
