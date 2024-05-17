import torch
import os
from cleanfid.features import build_feature_extractor
from utils.dataset import ImageNet
import numpy as np
from tqdm import tqdm
from torchvision import transforms

EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}

def get_files_features(fdir, model=None, exp_size=32, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       custom_fn_resize=None,
                       description="", verbose=True,
                       custom_image_tranform=None):
    
    dataset = ImageNet(
        root=fdir,
        split="train",
        exp_size=exp_size,
        transform=custom_image_tranform
    )
    
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader
    
    for batch in pbar:
        with torch.no_grad():
            x, _ = batch
            feat = model(x.to(device)*255.)
        l_feats.append(feat.detach().cpu())
    np_feats = np.concatenate(l_feats)
    return np_feats

device = torch.device("cuda")

fdir = '/data/ImageNet'
feat_model = build_feature_extractor("legacy_tensorflow", device, use_dataparallel=True)

custom_image_tranform=transforms.Compose([transforms.ToTensor()])

# compute ImageNet64 statistics for FID
np_feats64 = get_files_features(fdir, feat_model, exp_size=64, num_workers=12,
                                batch_size=512, device=device, 
                                custom_fn_resize=None,
                                custom_image_tranform=custom_image_tranform,
                                description="Precomputed Statistics for FID", verbose=True)

mu = np.mean(np_feats64, axis=0)
sigma = np.cov(np_feats64, rowvar=False)
print(f"saving custom FID stats to {'ImageNet64'}")
os.makedirs('./inception', exist_ok=True)
outf = os.path.join('./inception', f"ImageNet64_train_fid_refs.npz")
np.savez_compressed(outf, mu=mu, sigma=sigma)

