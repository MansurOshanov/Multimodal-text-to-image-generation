import math
from pathlib import Path
import sys
 
sys.path.append('./taming-transformers')
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
 
from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image
from imgtag import ImgTag    
from libxmp import *         
import libxmp                
from stegano import lsb
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
 
 
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply
 
 
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply
 
 
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
 
 
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
 
 
def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])
 
 
class Augmentions(nn.Module):
    def __init__(self, cut_size):
        super().__init__()
        self.cut_size = cut_size
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7)
        )
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY)
        cutouts = []
        size = int(torch.rand([]) * (max_size - min_size) + min_size)
        offsetx = torch.randint(0, sideX - size + 1, ())
        offsety = torch.randint(0, sideY - size + 1, ())
        cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
        cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        return batch
 
 
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    vqgan_model = vqgan.VQModel(**config.model.params)
    vqgan_model.eval().requires_grad_(False)
    vqgan_model.init_from_ckpt(checkpoint_path)
    del vqgan_model.loss
    return vqgan_model
 
 
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


#prompts = "A kid playing with a smartphone." 
#prompts = "A photo of a blue cat." 
#prompts = "A snail with the texture of a harp." 
prompts = "a small red block sitting on a large green block. " 
#prompts = 'A cat sitting on a table.'
#prompts = 'Two racing cars.'
#prompts = 'A photo of a train.'
#prompts = 'Kids are playing football.'
#prompts = 'A house of cards.'
#prompts = 'A stack of books.'
#prompts = 'A green clock in the shape of a pentagon.'
#prompts = 'Two racing cars.'
#prompts = 'A collection of glasses is sitting on a table.'
#prompts = 'a photo of a cat.'
#prompts = 'A kid playing with a smartphone.'
#prompts = 'A blue circle.'



width =  300
height = 300 
display_frequency =  50
max_iterations = 1000

prompts = [frase.strip() for frase in prompts.split("|")]
if prompts == ['']:
    prompts = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Text prompt:', prompts)
seed = torch.seed()
torch.manual_seed(seed)
print('Seed:', seed)

vqgan_model = load_vqgan_model('vqgan_imagenet_f16_16384.yaml', 'vqgan_imagenet_f16_16384.ckpt').to(device)
clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)

e_dim = vqgan_model.quantize.e_dim
f = 2**(vqgan_model.decoder.num_resolutions - 1)
augmentions = Augmentions(clip_model.visual.input_resolution)
n_toks = vqgan_model.quantize.n_e
toksX, toksY = width // f, height // f
sideX, sideY = toksX * f, toksY * f

one_hot_1 = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
z_1 = one_hot_1 @ vqgan_model.quantize.embedding.weight
z_1 = z_1.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

one_hot_2 = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
z_2 = one_hot_2 @ vqgan_model.quantize.embedding.weight
z_2 = z_2.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

z_1_orig = z_1.clone()
z_2_orig = z_2.clone()
z_1.requires_grad_(True)
z_2.requires_grad_(True)
#opt = optim.Adam([z_1, z_2], lr=0.1, weight_decay=1e-5)
opt = optim.Adam([z_1, z_2], lr=0.1)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

pMs = []

for prompt in prompts:
    txt, weight, stop = parse_prompt(prompt)
    embed = clip_model.encode_text(clip.tokenize(txt).to(device)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

def generate_vqgan_image(z_1, z_2):
    z_q_1 = vector_quantize(z_1.movedim(1, 3), vqgan_model.quantize.embedding.weight).movedim(3, 1)
    z_q_2 = vector_quantize(z_2.movedim(1, 3), vqgan_model.quantize.embedding.weight).movedim(3, 1)
    generated_image_1 = clamp_with_grad(vqgan_model.decode(z_q_1).add(1).div(2), 0, 1)
    generated_image_2 = clamp_with_grad(vqgan_model.decode(z_q_2).add(1).div(2), 0, 1)
    generated_image = torch.mean(torch.stack([generated_image_1, generated_image_2]), dim=0)
    return generated_image


@torch.no_grad()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = generate_vqgan_image(z_1, z_2)
    TF.to_pil_image(out[0].cpu()).save('progress.png')
    display.display(display.Image('progress.png'))

def compute_loss():
    global i
    out = generate_vqgan_image(z_1, z_2)
    iii = clip_model.encode_image(normalize(augmentions(out))).float()
    result = []
    for prompt in pMs:
        result.append(prompt(iii))
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    return result

def train(i):
    opt.zero_grad()
    lossAll = compute_loss()
    if i % display_frequency == 0:
        checkin(i, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()

i = 0
try:
    with tqdm() as pbar:
        while True:
            train(i)
            if i == max_iterations:
                break
            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass
