import multiprocessing
import torch
import argparse
from pathlib import Path
import time
import itertools
import os
import sys
script_dir_orchestrator = Path(__file__).parent.resolve()
project_root_orchestrator = script_dir_orchestrator.parent
if str(project_root_orchestrator) not in sys.path:
    sys.path.insert(0, str(project_root_orchestrator))
try:
    from scripts.train_model import run_training_instance
except ImportError as e:
    sys.exit(1)
except Exception as e:
    sys.exit(1)

def main_orchestrator(args):
    multiprocessing.set_start_method('spawn', force=True)
    start_time = time.time()
    model_types = ['csi', 'rssi']
    architectures = ['unet', 'fcn', 'deeplab']
    experiments = list(itertools.product(model_types, architectures))
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        max_concurrent_runs = args.max_cpu_processes
    else:
        max_concurrent_runs = min(num_gpus, args.max_gpu_processes) if args.max_gpu_processes > 0 else num_gpus
    print(f"[Orchestrator] Running up to {max_concurrent_runs} experiments concurrently.")
    run_arguments = []
    for i, (m_type, arch) in enumerate(experiments):
        gpu_id = i % num_gpus if num_gpus > 0 else -1
        run_args = {
            "model_type": m_type,
            "architecture": arch,
            "gpu_id": gpu_id,
            "base_output_dir": Path(args.base_output_dir),
            "csi_data_dir": Path(args.csi_data_dir),
            "rssi_data_dir": Path(args.rssi_data_dir),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "base_filters": args.base_filters,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "num_workers": args.num_workers,
            "weight_decay": args.weight_decay,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_factor": args.scheduler_factor
        }
        run_arguments.append(run_args)
    print(f"[Orchestrator] Starting multiprocessing Pool with {max_concurrent_runs} workers...")
    try:
        with multiprocessing.Pool(processes=max_concurrent_runs) as pool:
            results = pool.starmap(run_training_instance_wrapper, [(args_dict,) for args_dict in run_arguments])
        print("[Orchestrator] All training processes have completed.")
    except Exception as e:
        print(f"[Orchestrator] ERROR: An exception occurred during multiprocessing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass
    end_time = time.time()
    total_duration = end_time - start_time
    print(f"[Orchestrator] Finished all experiments in {total_duration:.2f} seconds.")

def run_training_instance_wrapper(args_dict):
    try:
        run_training_instance(**args_dict)
        return True
    except Exception as e:
        print(f"[Wrapper-GPU{args_dict.get('gpu_id','?')}] Error calling run_training_instance: {e}")
        return False

if __name__ == "__main__":
    default_processed_csi = project_root_orchestrator / "processed_data_csi"
    default_processed_rssi = project_root_orchestrator / "processed_data_rssi"
    default_output_dir = project_root_orchestrator / "training_output_parallel"
    parser = argparse.ArgumentParser(description="Run multiple training experiments in parallel.")
    parser.add_argument('--base_output_dir', type=str, default=str(default_output_dir),
                        help="Base directory where results for all runs will be stored (subdirs created).")
    parser.add_argument('--max_gpu_processes', type=int, default=0,
                        help="Maximum number of concurrent processes per GPU (0 = #GPUs).")
    parser.add_argument('--max_cpu_processes', type=int, default=2,
                        help="Maximum number of concurrent processes if running on CPU.")
    parser.add_argument('--csi_data_dir', type=str, default=str(default_processed_csi))
    parser.add_argument('--rssi_data_dir', type=str, default=str(default_processed_rssi))
    parser.add_argument('--epochs', type=int, default=50, help="Epochs for EACH training run.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=64, help="Base filters for all models (can be overridden if logic is added).")
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--num_workers', type=int, default=2, help="Workers PER process. Keep low if many processes.")
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting and training.")
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler_patience', type=int, default=7)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    args = parser.parse_args()
    main_orchestrator(args)