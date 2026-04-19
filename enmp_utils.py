import numpy as np
import torch

def decode_params(x, topk, threshold, addscale, detail_print=False):
    if addscale:
        t = torch.as_tensor(x[:-1])
        scaling_coeffs = torch.sigmoid(torch.as_tensor(x[-1])).item() * 3.0  # constrain to [0, 3]
    else:
        t = torch.as_tensor(x)
        scaling_coeffs = None
    
    vals, idx = torch.topk(t, k=topk)

    if detail_print:
        length = (vals > threshold).sum().item()
        print(f"==>> mask: {idx.tolist()[:length]}")
        if addscale:
            print(f"==>> scaling_coeffs: {scaling_coeffs:.3f}")
    
    mask = torch.zeros_like(t, dtype=torch.long)
    valid_indices = idx[vals > threshold]
    mask[valid_indices] = 1
    
    return mask, scaling_coeffs

def objective(x, env, topk, threshold, addscale):
    drop_state, scaling_coeffs = decode_params(x, topk, threshold, addscale)

    score = env.eval(drop_state, scaling_coeffs=scaling_coeffs)

    # CMA-ES minimizes, so negate
    loss = -score
    return loss

import requests
from email.utils import parsedate_to_datetime
from datetime import timezone, timedelta

def get_beijing_time_str():
    url = "http://www.google.com"
    r = requests.get(url)

    dt_gmt = parsedate_to_datetime(r.headers['Date'])
    beijing_tz = timezone(timedelta(hours=8))
    dt_beijing = dt_gmt.astimezone(beijing_tz)

    return dt_beijing.strftime("%Y-%m-%d_%H-%M-%S")