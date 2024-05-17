import argparse
import math
import os
import numpy as np
import torch
import torchvision
import torch.distributed as dist
from utils.ddp_utils import init_processes
from torch.multiprocessing import Process
from cleanfid.fid import frechet_distance
from cleanfid.features import build_feature_extractor
import glob
import shutil

from tqdm import tqdm
from models.unet import DhariwalUNet



ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]

class NFECount(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))

    def __call__(self, t, x, *args, **kwargs):
        self.nfe += 1.0
        return self.model(t, x, *args, **kwargs)

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

def sample_and_test(rank, gpu, args):
    torch.set_grad_enabled(False)

    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:{}".format(gpu))
    
    model = DhariwalUNet(
            img_resolution=64,
            in_channels=3,
            out_channels=3,
            label_dim=1000,
            augment_dim=0,
            model_channels=args.num_channel,
            channel_mult=[1, 2, 3, 4],
            channel_mult_emb=4,
            num_blocks=3,
            attn_resolutions=[32,16,8],
            dropout=0.1,
            label_dropout=0.2,
            use_context=False,
    ).to(device)
    
    state_dict = torch.load(
        f"./work_dir/{args.exp}/pretrained_weight_imagenet64.pt",
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

    save_dir = "./work_dir/generated_samples_TGaussian_Reflected/{}/exp{}_m{}_cfg{}_step{}".format(args.dataset, args.exp, args.method, args.cfg_scale, args.num_steps)
    
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if args.reflect:
        from torchdiffeqReflect import odeint
    else:
        from torchdiffeq import odeint
    constraint_r=1.0
    def run_sampling(cls_index=None):
        with torch.no_grad():
            num_ = args.batch_size * 3* 64 * 64
            x = torch.randn(int(2.0 * num_))
            x = (x[(x.abs() <= constraint_r)][:num_].reshape(args.batch_size, 3, 64, 64)).to(device)
            if args.num_classes in [None, 1]:
                model_kwargs = {}
            else:
                if cls_index is None:
                    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
                else:
                    y = torch.ones(args.batch_size, device=device, dtype=torch.long) * cls_index
                    y = y.long()
                
                x = torch.cat([x, x], 0)
                y_null = torch.zeros_like(y, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y)
            
            def denoiser(t, x_0):
                half = x_0[: len(x_0) // 2]
                combined = torch.cat([half, half], dim=0)
                model_out = model(t, combined, y)
                model_out = model(t, combined, **model_kwargs)
                cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
                half_eps = uncond_eps + args.cfg_scale * (cond_eps - uncond_eps)
                vel = torch.cat([half_eps, half_eps], dim=0)
                return vel
            
            t_span = torch.linspace(0, 1, args.num_steps).to(device) 
            if args.reflect:
                traj = odeint(
                denoiser, x, t_span, rtol=args.rtol, atol=args.atol, method=args.method, reflect_fn=reflect, reflect_velocity_fn=reflect_velocity
            )
            else:
                traj = odeint(
                    model, x, t_span, rtol=args.rtol, atol=args.atol, method= args.method
                )
            
        traj = traj[-1, :]  
        traj,_ = traj.chunk(2, dim=0)  
        img = (traj * 127.5 + 128)
        return img.to(torch.uint8)
    
    if args.compute_fid:
        print("Compute fid")
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
                index = rank + total
                np.save("{}/{}.npy".format(save_dir, index), img_batch.detach().cpu().numpy())
                torchvision.utils.save_image(img_batch[0:64,...]/255., "{}/{}.jpg".format(save_dir, index)) #! torchvision save a 0-1 image: (img * 255 +0.5).clamp(0,255)
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
                #! load each batch for feat extraction.
                img_batch = torch.from_numpy(np.load(all_imgs[index])).to(device)
                with torch.no_grad():
                    np_feat = feat_model(img_batch.to(device)).detach().cpu().numpy()
                    np_feats.append(np_feat)
                total += img_batch.shape[0]
            np_feats = np.concatenate(np_feats)[:args.n_sample]
            mu = np.mean(np_feats, axis=0)
            sigma = np.cov(np_feats, rowvar=False)
                        
            ref_dict = dict(np.load('./inception/ImageNet64_train_fid_refs.npz'))
            ref_mu, ref_sigma = ref_dict['mu'],ref_dict['sigma']

            fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
            
            os.makedirs('./FID', exist_ok=True)
            print("FID = {}".format(fid ))
            with open(args.output_log, "a") as f:
                f.write("FID = {}\n".format(fid))
                
    if args.compute_nfe:
        if rank == 0:
            average_nfe = 0.0
            num_trials = 300
            model = NFECount(model)
            for i in tqdm(range(num_trials)):
                with torch.no_grad():
                    num_ = 1 * 3* 64* 64
                    x = torch.randn(int(2.0 * num_))
                    x = (x[(x.abs() <= constraint_r)][:num_].reshape(1, 3, 64, 64)).to(device)
                    if args.num_classes in [None, 1]:
                        model_kwargs = {}
                    else:
                        y = torch.randint(0, args.num_classes, (1,), device=device)
                        
                        x = torch.cat([x, x], 0)
                        y_null = torch.zeros_like(y, device=device)
                        y = torch.cat([y, y_null], 0)
                        model_kwargs = dict(y=y)
                    
                    def denoiser(t, x_0):
                        half = x_0[: len(x_0) // 2]
                        combined = torch.cat([half, half], dim=0)
                        model_out = model(t, combined, y)
                        model_out = model(t, combined, **model_kwargs)
                        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
                        half_eps = uncond_eps + args.cfg_scale * (cond_eps - uncond_eps)
                        vel = torch.cat([half_eps, half_eps], dim=0)
                        return vel
                    
                    t_span = torch.linspace(0, 1, args.num_steps).to(device) 
                    traj = odeint(
                        denoiser, x, t_span, rtol=args.rtol, atol=args.atol, method=args.method, reflect_fn=reflect, reflect_velocity_fn=reflect_velocity
                    )
                    
            average_nfe = model.nfe / num_trials
            print(f"Average NFE over {num_trials} trials: {int(average_nfe)}")
            with open(args.output_log, "a") as f:
                f.write(f"CFG: {args.cfg_scale}, average NFE = {average_nfe}")
        if rank==0:
            shutil.rmtree(save_dir)
        dist.barrier()
        dist.destroy_process_group()
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=True, help="whether or not compute FID")
    parser.add_argument("--compute_nfe", action="store_true", default=False, help="whether or not compute NFE")
    parser.add_argument("--n_sample", type=int, default=50000, help="number of sampled images")
    parser.add_argument("--num_classes", type=int, default=1000, help="num classes")
    parser.add_argument("--num_channel", type=int, default=192, help="num channels")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Scale for classifier-free guidance")
    parser.add_argument("--output_log", type=str, default="")
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="ImageNet64", help="name of dataset")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=50, help="sample generating batch size")

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
    parser.add_argument("--reflect", type=bool, default=True, help="using reflected FM")
    
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
