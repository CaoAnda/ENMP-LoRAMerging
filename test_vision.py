import argparse
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*peft_config.*")

import numpy as np
import torch

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    prepare_experiment_config,
    set_seed,
    parse_eval_args,
    merge_args_into_task_merge_config,
)
from tqdm import tqdm
from eval_utils.vision_pertask import validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="vitB_r16_full_ties.py",
        help="config file name"
    )
    # type=str to receive the whole list as a single string
    parser.add_argument(
        "--mask",
        type=str,
        default="[]",
        help="mask list, e.g. '[1, 2, 3]'"
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_config = get_config_from_name(args.config, device=device)
    config = prepare_experiment_config(raw_config)
    all_clip_encodings = [
        get_clip_encodings(i["clip_encodings"]).to(device)
        for i in raw_config["dataset"]
    ]
    drop_state = np.zeros((12, 8)).flatten()  # 12 layers, 8 models
    mask = eval(args.mask)
    print(f"==>> mask: {mask}")
    drop_state[mask] = 1

    norm_accs, abs_accs = validate(
        drop_state.reshape(12, 8),
        device,
        raw_config,
        all_clip_encodings,
        config,
        silent=False,
        EVAL_SPLIT="test",
        merge_cache=False,
        return_abs_acc=True
    )
    avg_norm_acc = sum(norm_accs) / len(norm_accs)
    avg_abs_acc = sum(abs_accs) / len(abs_accs)
    print("Normalized Accuracies:", [f'{acc:<.4f}' for acc in norm_accs])
    print("Absolute Accuracies:", [f'{acc:<.4f}' for acc in abs_accs])
    print(f"Average Normalized Accuracy: {avg_norm_acc:.4f}")
    print(f"Average Absolute Accuracy: {avg_abs_acc:.4f}")