# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from utils.ddp_utils import init_processes
from torch.multiprocessing import Process

from tqdm import tqdm
from models import UNet
import torch.nn as nn
from cleanfid.fid import frechet_distance, get_reference_statistics
from cleanfid.features import build_feature_extractor
from utils.truncated_normal import TruncatedNormal
import glob
import shutil


def reflect(base, step):
    bottommost=-1
    edge=2.0
    sample = base + step
    reflectsample = (sample - bottommost) % (2 * edge)
    reflectsample[reflectsample>edge] = 2*edge - reflectsample[reflectsample>edge]
    return reflectsample + bottommost

def reflect_velocity(base, step, velocity):
    bottommost=-1
    edge=2.0
    sample = base + step
    cond = ((sample - bottommost) % edge) // 2   ## 1 if need reflection, 0 if not need reflection
    cond = 1 - 2 * cond
    return cond * velocity

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]

class NFECount(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))

    def __call__(self, t, x, *args, **kwargs):
        self.nfe += 1.0
        return self.model(t, x, *args, **kwargs)

def sample_and_test(rank, gpu, args):
    torch.set_grad_enabled(False)

    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:{}".format(gpu))
    model = UNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    state_dict = torch.load(
        f"./work_dir_TGaussian/{args.exp}/cifar10_weights_step_{args.iteration}.pt",
        map_location=device,
    )["ema_model"]

    print("Finish loading model")
    try:
        model.load_state_dict(state_dict,strict=True)
        del state_dict
    except RuntimeError:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict, strict=True)
        del new_state_dict
        del state_dict

    
    model.eval()

    save_dir = "./generated_samples/{}/exp{}_m{}_{}".format(args.dataset, args.exp, args.method, args.steps)
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tg_gaussian = TruncatedNormal(torch.Tensor([0]), torch.Tensor([1]), -1, 1)
    if args.reflect:
        import sys
        sys.path.append('../')
        from torchdiffeqReflect import odeint
    else:
        from torchdiffeq import odeint
    def run_sampling():
        with torch.no_grad():
            x = tg_gaussian.rsample(torch.Size([args.batch_size, 3, 32, 32])).squeeze().to(device)
            t_span = torch.linspace(0, 1, args.steps).to(device)
            if args.reflect:
                traj = odeint(
                model, x, t_span, rtol=args.rtol, atol=args.atol, method=args.method, reflect_fn=reflect, reflect_velocity_fn=reflect_velocity
            )
            else:
                traj = odeint(
                    model, x, t_span, rtol=args.rtol, atol=args.atol, method= args.method
                )
        traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
        img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
        return img


    if args.compute_fid:
        print("Compute fid")
        # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
        n = args.batch_size
        global_batch_size = args.world_size
        total_samples = int(math.ceil(args.n_sample / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % args.world_size == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // args.world_size)
        iters_needed = int(samples_needed_this_gpu // n)
        pbar = range(iters_needed)
        pbar = tqdm(pbar, desc="sampling: ") if rank == 0 else pbar
        total=0
        
        for i in pbar:
            with torch.no_grad():
                img_batch = run_sampling()
                #!save each batch
                index = rank + total
                np.save("{}/{}.npy".format(save_dir, index), img_batch.detach().cpu().numpy())
                torchvision.utils.save_image(img_batch/255., "{}/{}.jpg".format(save_dir, index)) #! torchvision save a 0-1 image: (img * 255 +0.5).clamp(0,255)
                total += global_batch_size
        model=model.cpu()
        del model

        dist.barrier()
        if rank == 0:
            feat_model = build_feature_extractor("legacy_tensorflow", device, use_dataparallel=False) #!InceptionV3
            all_imgs = glob.glob(f'{save_dir}/*npy')
            total=0
            np_feats=[]
            for index in tqdm(range(len(all_imgs)), desc="FID calculation: "):
                img_batch = torch.from_numpy(np.load(all_imgs[index])).to(device)
                with torch.no_grad():
                    np_feat = feat_model(img_batch.to(device)).detach().cpu().numpy()
                    np_feats.append(np_feat)
                total += img_batch.shape[0]
            np_feats = np.concatenate(np_feats)[:args.n_sample]
            mu = np.mean(np_feats, axis=0)
            sigma = np.cov(np_feats, rowvar=False)
            ref_mu, ref_sigma = get_reference_statistics(args.dataset, args.image_size,
                            mode="legacy_tensorflow", model_name="inception_v3",
                            seed=0, split="train")
            
            fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
            os.makedirs('./FID', exist_ok=True)
            print("FID = {}".format(fid))
            with open(args.output_log, "a") as f:
                f.write("FID = {}\n".format(fid))
        dist.barrier()
        dist.destroy_process_group()
        exit(0)
    
    if args.compute_nfe:
        average_nfe = 0.0
        num_trials = 300
        model = NFECount(model)
        for i in tqdm(range(num_trials)):
            with torch.no_grad():
                x = tg_gaussian.rsample(torch.Size([1, 3, 32, 32])).squeeze(-1).to(device)
                # print("Use method: ", args.method)
                t_span = torch.linspace(0, 1, args.steps).to(device)
                traj = odeint(
                    model, x, t_span, rtol=args.rtol, atol=args.atol, method= args.method
                )
            # traj = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
            # img = (traj * 127.5 + 128).clip(0, 255).to(torch.uint8)  # .permute(1, 2, 0)
            print(model.nfe)
        average_nfe = model.nfe / num_trials
        print(f"Average NFE over {num_trials} trials: {int(average_nfe)}")
        exit(0)
    
    if rank==0:
        shutil.rmtree(save_dir)
    dist.barrier()
    dist.destroy_process_group()
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--compute_nfe", action="store_true", default=False, help="whether or not compute NFE")
    parser.add_argument("--n_sample", type=int, default=50000, help="number of sampled images")
    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--output_log", type=str, default="")
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=200, help="sample generating batch size")

    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=[
            "dopri5",
            "dopri8",
            "adaptive_heun",
            "bosh3",
            "euler",
            "midpoint",
            "rk4",
            "heun3"
        ],
    )
    parser.add_argument("--steps", type=int, default=2, help="steps for solver")

    # ddp
    parser.add_argument("--num_proc_node", type=int, default=1, help="The number of nodes in multi node env.")
    parser.add_argument("--num_process_per_node", type=int, default=1, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="6000", help="port for master")

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    

    if size > 1 and args.compute_fid:
        torch.multiprocessing.set_start_method('spawn')
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print("Node rank %d, local proc %d, global proc %d" % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, sample_and_test, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print("starting in debug mode")

        init_processes(0, size, sample_and_test, args)
