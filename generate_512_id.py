"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from models import indentity

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--mask_path",
        type=str,
        default="test_data/512_masks/27007.png",
        help="path to the segmentation mask"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="text condition"
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: batch_size)",
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/512_id.yaml",
        help="path to config which constructs model",
    )
    
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/jijunhao/diffusion/outputs/512_id/2023-07-01T21-06-39_512_id/pretrained/last.ckpt",
        help="path to checkpoint of model",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="the seed (for reproducible sampling)",
    )
    
    # ========== set up model ==========
    print(f'Set up model')

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    # ========== set output directory ==========
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # ========== prepare seg mask for model ==========
    with open(opt.mask_path, 'rb') as f:
        img = Image.open(f)
        resized_img = img.resize((32,32), Image.NEAREST) # resize
        flattened_img = list(resized_img.getdata())
    flattened_img_tensor = torch.tensor(flattened_img) # flatten
    flattened_img_tensor_one_hot = F.one_hot(flattened_img_tensor, num_classes=19) # one hot
    flattened_img_tensor_one_hot_transpose = flattened_img_tensor_one_hot.transpose(0,1)
    flattened_img_tensor_one_hot_transpose = torch.unsqueeze(flattened_img_tensor_one_hot_transpose, 0).cuda() # add batch dimension

    # ========== prepare mask for visualization ==========
    mask = Image.open(opt.mask_path)
    mask = mask.convert('RGB')
    mask = np.array(mask).astype(np.uint8) # three channel integer
    input_mask = mask

    print(f'================================================================================')
    print(f'mask_path: {opt.mask_path} | text: {opt.prompt} | init_img: {opt.init_img}')

    batch_size = opt.batch_size
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # ========== sampling ==========
    with torch.no_grad():
        tic = time.time()
        all_samples = list()
        TestFace = indentity.TestFace()
        condition = TestFace.pred_id(init_image, 'ir152', TestFace.targe_models)
        #condition = flattened_img_tensor_one_hot_transpose
        """
         condition = {
            'seg_mask': flattened_img_tensor_one_hot_transpose
            'text': [args.input_text.lower()]
        }       
        """
        with model.ema_scope("Plotting"):
            # encode condition
            condition = model.get_learned_conditioning(condition)
            if isinstance(condition, dict):
                for key, value in condition.items():
                    condition[key] = condition[key].repeat(batch_size, 1, 1)
            else:
                condition = condition.repeat(batch_size, 1, 1)


            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
            # decode it
            samples = sampler.decode(z_enc, condition, t_enc)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            if not opt.skip_save:
                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
            all_samples.append(x_samples)

            if not opt.skip_grid:
            # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
