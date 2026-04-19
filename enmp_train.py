import cma
import os
import sys
import time
import torch.multiprocessing as mp
from enmp_utils import decode_params, get_beijing_time_str
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=float, default=0.2)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--config", type=str, default="vitB_r16_full_ties.py")
    parser.add_argument("--addscale", action='store_true')
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="number of GPUs to use (default: all available)")
    return parser.parse_args()


def worker_fn(rank, num_gpus, task_queue, result_queue, args, init_queue=None):
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    import os
    import torch
    # Set visible GPU before any CUDA call to prevent context leaking to other cards
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    from enmp_utils import objective
    from enmp_env import Env
    device = torch.device("cuda:0")  # only one card visible to this process
    env = Env(device, args.seed, args.config)
    topk = int(args.topk * env.num_layers * env.num_models)

    # rank 0 sends metadata back to the main process to avoid allocating a tmp_env there
    if init_queue is not None:
        init_queue.put({
            'num_layers': env.num_layers,
            'num_models': env.num_models,
            'init_norm_acc': env.init_norm_acc,
            'default_scaling_coeffs': env.default_scaling_coeffs,
            'merge_config': env.merge_config,
        })

    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, x = item
        loss = objective(x, env, topk, args.threshold, args.addscale)
        result_queue.put((idx, loss))


def setup_print_log():
    log_dir = "EXP_LOG"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = get_beijing_time_str()
    return os.path.join(log_dir, f"enmp_log_{timestamp}.print")


def main():
    args = parse_args()

    # main process needs no GPU; hide all to prevent CUDA context leaks
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    num_gpus = len(result.stdout.strip().splitlines())
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found.")
    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, num_gpus)

    print(f"Using {num_gpus} GPU(s) for parallel evaluation.")

    # spawn workers; rank 0 sends metadata back via init_queue
    spawn_ctx = mp.get_context('spawn')
    task_queue = spawn_ctx.Queue()
    result_queue = spawn_ctx.Queue()
    init_queue = spawn_ctx.Queue()

    workers = []
    for rank in range(num_gpus):
        p = spawn_ctx.Process(
            target=worker_fn,
            args=(rank, num_gpus, task_queue, result_queue, args),
            kwargs={'init_queue': init_queue if rank == 0 else None},
        )
        p.start()
        workers.append(p)

    # wait for worker 0 to finish init and return metadata
    meta = init_queue.get()
    num_layers = meta['num_layers']
    num_models = meta['num_models']
    init_norm_acc = meta['init_norm_acc']
    default_scaling_coeffs = meta['default_scaling_coeffs']
    merge_config = meta['merge_config']

    if args.addscale:
        x0 = [-1.0] * (num_layers * num_models) + [1.0]
    else:
        x0 = [-1.0] * (num_layers * num_models)

    sigma0 = args.sigma0
    topk = int(args.topk * num_layers * num_models)
    options = {
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "verb_disp": 1,
        "seed": args.seed,
    }

    print_path = setup_print_log()
    sys.stdout = open(print_path, "w+")

    print("\n" + "=" * 80)
    print("Experiment Configuration Summary")
    print("=" * 80)
    print(f"GPUs          : {num_gpus}")
    print(f"Random Seed   : {args.seed}")
    print(f"config_name   : {args.config}")
    print(f"merge_config  : {merge_config}")
    print(f"init_acc      : {init_norm_acc}")
    print(f"default_scaling_coeffs: {default_scaling_coeffs}")
    print("\n[ CMA-ES Settings ]")
    print(f"  num_layers : {num_layers}")
    print(f"  num_models : {num_models}")
    print(f"  x0 length  : {len(x0)}")
    print(f"  sigma0     : {sigma0}")
    print(f"  topk       : {topk}")
    print(f"  threshold  : {args.threshold}")
    print(f"  addscale   : {args.addscale}")
    print("\n[ CMA Options ]")
    for k, v in options.items():
        print(f"  {k:12s}: {v}")
    print("=" * 80 + "\n")

    es = cma.CMAEvolutionStrategy(x0, sigma0, options)

    while not es.stop():
        solutions = es.ask()
        n = len(solutions)

        t0 = time.time()
        for idx, x in enumerate(solutions):
            task_queue.put((idx, x))

        results = [None] * n
        for _ in range(n):
            idx, loss = result_queue.get()
            results[idx] = loss
        elapsed = time.time() - t0
        print(f"[timing] gen {es.countiter}: {n} evals on {num_gpus} GPU(s) in {elapsed:.1f}s ({elapsed/n:.2f}s/eval)", flush=True)

        es.tell(solutions, results)
        es.disp()

        best_x = es.best.get()[0]
        decode_params(best_x, topk, args.threshold, args.addscale)

    for _ in workers:
        task_queue.put(None)
    for p in workers:
        p.join()

    res = es.result
    x_best = res.xbest
    f_best = res.fbest

    _ = decode_params(x_best, topk, args.threshold, args.addscale, detail_print=True)
    print("Improvement:", -f_best)

    sys.stdout.close()

if __name__ == '__main__':
    main()
