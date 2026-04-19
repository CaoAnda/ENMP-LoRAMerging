import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*peft_config.*")

import transformers
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config
from task_merger import get_merge_handler
import torch
import numpy as np
import time
from copy import deepcopy
from huggingface_hub import utils

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
utils.logging.set_verbosity_error()

def validate(
    drop_state,
    device,
    raw_config,
    task_heads, 
    config,
    silent=True,
    EVAL_SPLIT="val",
    merge_cache=False,
    return_abs_acc=False
):
    config['task_merge_config'] = merge_args_into_task_merge_config(
        config['task_merge_config'], {})
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    # print(f"mask_class labels: {mask_class}")

    fine_tuned_acc = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }

    def merge_and_eval(Merge, EVAL_SPLIT, instance_params=None):
        all_results = deepcopy(instance_params)

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)
        merged_model = Merge.merge(config['task_merge_config'], drop_state=drop_state, merge_cache=merge_cache)

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1

        # print('Evaluate Merged Model on Each Dataset')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        norm_accs = []
        abs_accs = []
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i], silent=silent)

            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i] + '_norm_acc'] = (
                acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / \
                fine_tuned_acc[dataset_names[i]] * 100
            norm_accs.append(all_results[dataset_names[i] + '_norm_acc'])
            abs_accs.append(all_results[dataset_names[i]])
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)

        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        
        if return_abs_acc:
            return norm_accs, abs_accs
        return norm_accs

    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(
            config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(config['task_merge_config']['ingredients_path']):
            Merge.transform(config['task_merge_config'])

        return merge_and_eval(
            Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=config['task_merge_config'])

if __name__ == "__main__":
    device = 'cuda'
    raw_config = get_config_from_name("llama8B_r16_tv.py", device=device)
    config = prepare_experiment_config(raw_config)
    task_heads = torch.load("heads.pt")
    import time
    start_time = time.time()
    drop_state = torch.zeros((32, 6))
    # drop_state[[6, 3]] = 1
    print('Starting validation...')
    norm_accs = validate(
        drop_state=drop_state,
        device=device,
        raw_config=raw_config,
        task_heads=task_heads,
        config=config,
        merge_cache=False,
        silent=False,
        EVAL_SPLIT='val',
    )
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time} seconds")
    avg_norm_acc = sum(norm_accs) / len(norm_accs)
    print("Normalized Accuracies:", [f'{acc:<.4f}' for acc in norm_accs])
    print(f"Average Normalized Accuracy: {avg_norm_acc:.4f}")