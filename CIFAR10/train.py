import copy
import os
import torch

from models import UNet
from torchvision import datasets, transforms
import torchvision

from accelerate import Accelerator
from time import time
from torchdiffeq import odeint_adjoint as odeint
import argparse
from utils.truncated_normal import TruncatedNormal
from utils.utils_cifar import ema, infiniteloop
from utils.flow import TargetConditionalFlowMatcher

def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

def sample_from_model(model, x_0):
    t = torch.tensor([0.0, 1.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5)
    return fake_image

def train(args):
    
    accelerator = Accelerator()
    accelerator.print("lr, total_steps, ema decay, save_step:",
          args.lr, args.total_steps, args.ema_decay, args.save_step)
    
    def warmup_lr(step):
        return min(step, args.warmup) /  args.warmup

    #### DATASETS/DATALOADER
    dataset = datasets.CIFAR10(
        root=args.root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)

    device = accelerator.device
    #### MODELS
    net_model = UNet(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
        )  # new dropout + bs of 128


    # 
    accelerator.print("FM size: {:.3f}MB".format(get_weight(net_model)))

    ema_model = copy.deepcopy(net_model)
    ema_model.eval()
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    dataloader, net_model, ema_model, optim, sched = accelerator.prepare(dataloader, net_model, ema_model, optim, sched)

    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    accelerator.print("Model params: %.2f M" % (model_size / 1024 / 1024))
  

    parent_dir = "./"

    exp_path = os.path.join(parent_dir, args.exp)
    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
    accelerator.print("Exp path:", exp_path)


    if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        # load G
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        init_step = checkpoint["step"]

        accelerator.print("=> resume checkpoint (iterations {})".format(checkpoint["step"]))
        del checkpoint

    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        init_step = 0

        accelerator.print("=> loaded checkpoint (iterations {})".format(step))
        del checkpoint
    else:
        init_step = 0
    
    sigma = 0.0
    FM = TargetConditionalFlowMatcher(sigma=sigma)
    start_time = time()
    log_steps = 0
    tg_gaussian = TruncatedNormal(torch.Tensor([0]), torch.Tensor([1]), -1, 1)
    for step in range(init_step, args.total_steps):
        optim.zero_grad()
        x1 = next(datalooper).to(device)
        x0 = tg_gaussian.rsample(x1.shape).squeeze().to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = net_model(t, xt)
        loss = torch.mean((vt - ut) ** 2)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(net_model.parameters(), max_norm=args.grad_clip)

        optim.step()
        sched.step()
        ema(net_model, ema_model, args.ema_decay) 

        if accelerator.is_main_process:
            if step % 1000 == 0:
                if accelerator.is_main_process:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    accelerator.print(
                        "iteration{}, lr{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                            step, sched.get_lr(), loss.item(), steps_per_sec
                        )
                    )
                    log_steps = 0
                    start_time = time()

        if accelerator.is_main_process:
            if args.save_step > 0 and step % args.save_step == 0:
                rand = tg_gaussian.rsample(torch.Size([64, 3, 32, 32])).squeeze().to(x1.device) 
                fake_image_ema = sample_from_model(ema_model, rand)[-1]
                torchvision.utils.save_image(
                    fake_image_ema,
                    os.path.join(exp_path, "image_iteration_ema_{}.png".format(step)),
                    normalize=True,
                    value_range=(-1, 1),
                )
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "step": step,
                    },
                    exp_path+"/cifar10_weights_step_{}.pt".format(step),
                )

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    exp_path+"/content.pt".format(step),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FM parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--root", type=str, default="/data/cifar10")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")
    
    parser.add_argument("--model", type=str, default='otcfm')
    parser.add_argument("--img_size", type=int, default=32, help="size of image")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=400001)
    parser.add_argument("--warmup", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_channel", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_step", type=float, default=5000)
    parser.add_argument("--exp", type=str, default='cifar10_CFM_bs128_c128')

    args = parser.parse_args()
    train(args)