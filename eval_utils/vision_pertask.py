import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*peft_config.*")

import numpy as np
import torch

from torch.utils.data import Sampler, DataLoader
from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config
from tqdm import tqdm


def validate(
    drop_state,
    device,
    raw_config,
    all_clip_encodings,
    config,
    silent=True,
    EVAL_SPLIT = "val",
    merge_cache=False,
    return_abs_acc=False
):
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], {})
    dataset_names = [i['name'] for i in raw_config['dataset']]
    config['task_merge_config']['dataset_names'] = dataset_names
    dataloaders = [i for i in config['data']]

    model_type = config['model']['base_type']
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = get_vision_accuracies(model_type, peft_type=peft_type, rank=rank)

    pred_detail = {}
    with torch.no_grad():
        all_results = deepcopy(config['task_merge_config'])
        # iniitalize merging function
        models = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        Merge.transform(config['task_merge_config'])
        # set task scaling coefficients
        Merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])
        merged_model = Merge.merge(config['task_merge_config'], drop_state, merge_cache).to(device)

        norm_accs = []
        abs_accs = []
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(
                merged_model, loader, class_vectors=all_clip_encodings[i], silent=silent
            )

            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i] + '_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            abs_accs.append(acc * 100)
            norm_accs.append((acc * 100) / fine_tuned_acc[dataset_names[i]] * 100)
    
    if return_abs_acc:
        return norm_accs, abs_accs
    return norm_accs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_config = get_config_from_name("vitB_r16_full_ties.py", device=device)
    config = prepare_experiment_config(raw_config)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']).to(device) for i in raw_config['dataset']]
    
    drop_state = np.zeros((12, 8))  # 12 layers, 8 models
    # drop_state[0,0]=1
    import time
    start_time = time.time()
    accs = validate(drop_state, device, raw_config, all_clip_encodings, config, silent=False, EVAL_SPLIT="val", merge_cache=False)
    avg_acc = sum(accs) / len(accs)
    
    print(f"Average Normalized Accuracy: {avg_acc}")
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time} seconds.")
    