import hashlib
import math
import os
import uuid
import numpy as np
import torch

from torch.utils.data import Sampler, DataLoader
from utils import evaluate_cliphead, get_clip_encodings, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config

class Env():

    def __init__(
        self,
        device,
        random_seed=None,
        config_name=None,
    ):
        if random_seed:
            set_seed(random_seed)

        self.random_seed = random_seed
        self.raw_config = get_config_from_name(config_name, device=device)
        if 'vit' in config_name:
            self.heads = [
                get_clip_encodings(i["clip_encodings"]).to(device)
                for i in self.raw_config["dataset"]
            ]
            self.num_layers = 12
            from eval_utils.vision_pertask import validate
            self.validate = validate
        elif 'llama' in config_name:
            self.heads = torch.load("heads.pt")
            self.num_layers = 32
            from eval_utils.nli_pertask import validate
            self.validate = validate
        else:
            raise NotImplementedError("Only ViT and LLaMA are supported currently.")
        
        self.config = prepare_experiment_config(self.raw_config)
        self.merge_config = self.config["task_merge_config"]
        self.dataset_names = [i["name"] for i in self.raw_config["dataset"]]
        self.device = device
        self.default_scaling_coeffs = self.config['task_merge_config']['scaling_coeffs']

        self.num_models = len(self.dataset_names)

        self.drop_state = torch.zeros(
            (self.num_layers * self.num_models), dtype=torch.float
        )

        norm_accs = self.validate_with_cache(cache=False)
        self.init_norm_acc = sum(norm_accs) / len(norm_accs)

    def validate_with_cache(self, cache):
        if not cache:
            return self.validate(
                self.drop_state.reshape(self.num_layers, self.num_models),
                self.device,
                self.raw_config,
                self.heads,
                self.config,
                EVAL_SPLIT="val",
                merge_cache=False
            )

    def eval(self, drop_state, scaling_coeffs=None):
        if scaling_coeffs:
            self.config['task_merge_config']['scaling_coeffs'] = scaling_coeffs
        self.drop_state = drop_state
        norm_accs = self.validate_with_cache(cache=False)
        avg_acc = sum(norm_accs) / len(norm_accs)
        reward = avg_acc - self.init_norm_acc
        self.reset()
        
        return reward
    
    def reset(self):
        # clear map/restore pos/clean blocks
        self.drop_state = torch.zeros(
            (self.num_layers * self.num_models), dtype=torch.float
        )
        self.config['task_merge_config']['scaling_coeffs'] = self.default_scaling_coeffs