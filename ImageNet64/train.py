import os
import torch
from utils.utils import ema, infiniteloop
from accelerate import Accelerator
from time import time
from torchvision import transforms
import argparse
import copy
from utils.flow import TargetConditionalFlowMatcher
from utils.dataset import ImageNet
from models.unet import DhariwalUNet

def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

def train(args):
    
    def warmup_lr(step):
        return min(step, args.warmup) /  args.warmup
    
    accelerator = Accelerator()
    accelerator.print("lr, total_steps, ema decay, save_step:",
          args.lr, args.total_steps, args.ema_decay, args.save_step)
    
    dataset = ImageNet(
        root=args.root,
        exp_size=64,
        split="train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
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

    net_model = DhariwalUNet(
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
            label_dropout=0.1,
            use_context=False,  
    ).to(device)

    accelerator.print("FM size: {:.3f}MB".format(get_weight(net_model)))

    ema_model = copy.deepcopy(net_model)
    ema_model.eval()
    optim = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    dataloader, net_model, ema_model, optim, sched = accelerator.prepare(dataloader, net_model, ema_model, optim, sched)

    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    accelerator.print("Model params: %.2f M" % (model_size / 1024 / 1024))
    
    parent_dir = "./work_dir/"

    exp_path = os.path.join(parent_dir, args.exp)
    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
    accelerator.print("Exp path:", exp_path)

    if args.resume or os.path.exists(os.path.join(exp_path, "content.pt")):
        checkpoint_file = os.path.join(exp_path, "content.pt")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optim.load_state_dict(checkpoint["optim"])
        sched.load_state_dict(checkpoint["sched"])
        init_step = checkpoint["step"]

        accelerator.print("=> resume checkpoint (iterations {})".format(checkpoint["step"]))
        del checkpoint

    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        net_model.load_state_dict(checkpoint["net_model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        init_step = checkpoint["step"]

        accelerator.print("=> resume checkpoint (iterations {})".format(step))
        del checkpoint
    else:
        init_step = 0
    
    sigma = 0.0
    FM = TargetConditionalFlowMatcher(sigma=sigma)
    start_time = time()
    constraint_r=1.0
    for step in range(init_step, args.total_steps*args.accm_grad):
        optim.zero_grad()
        x1,y = next(datalooper)
        x1 = x1.to(device)
        y = y.to(device)
        x0 = torch.randn(int(2.0 * x1.numel()))
        x0 = (x0[(x0.abs() <= constraint_r)][:x1.numel()].reshape(x1.shape)).to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        vt = net_model(t.squeeze(), xt, y=y)
        loss = torch.mean((vt - ut) ** 2)
        accelerator.backward(loss)
        optim.step()
        sched.step()
        ema(net_model, ema_model, args.ema_decay)  # new

        if accelerator.is_main_process:
            if step % 100 == 0 and step>0:
                if accelerator.is_main_process:
                    end_time = time()
                    steps_per_sec = (end_time - start_time)/step
                    accelerator.print(
                        "iteration{}, lr{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                            step, sched.get_lr(), loss.item(), steps_per_sec
                        )
                    )

        if accelerator.is_main_process:
            if args.save_step > 0 and step % args.save_step == 0:
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    exp_path+"/imagenet_weights_step_{}.pt".format(step),
                )
            if args.save_step > 0 and step % args.save_step//5 == 0:
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
    parser.add_argument("--root", type=str, default="/data/ImageNet")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")    
    parser.add_argument("--model", type=str, default='otcfm')
    parser.add_argument("--img_size", type=int, default=32, help="size of image")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=400001)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_channel", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--weight_decay",  type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_step", type=float, default=5000)
    parser.add_argument("--exp", type=str, default='ReflectedFM')
    parser.add_argument("--accm_grad", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='imagenet')

    args = parser.parse_args()
    train(args)