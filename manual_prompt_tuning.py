import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import open_clip
import torch
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Settings
device = 'cuda:0'
model_name = 'ViT-bigG-14-CLIPA'
pretrained = 'datacomp1b'
root = './'
dataset_name = 'Scene'

# Load pretrained CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# Load test dataset (Scene dataset)
ds = ImageFolder(os.path.join(root, dataset_name), transform=preprocess)
ds.samples = natsorted(ds.samples)
dl = DataLoader(ds, shuffle=False, batch_size=32, num_workers=2)

# Load class name list.
with open(os.path.join(root, f'{dataset_name}/classes.json'), 'r') as j:
     class_names = json.loads(j.read())

# TODO: manually tuning prompts
# Templates to use
templates = [
    "{}", # need
    # "itap of a {}.", # don't need
    "a bad photo of the {}.", # need
    "a origami {}.", # need
    "a photo of the large {}.", # need
    "a {} in a video game.", # need
    "art of the {}.", # need
    # "a photo of the small {}." # don't need
    # "a blurry photo of the {}." # don't need
    "a view of the {}.", # need
    # "a cropped photo of the {}." # don't need
    # "the embroidered {}." # don't need
    # "a dark photo of the {}.", # don't need
    # "a low resolution photo of the {}.", # don't need
    "a photo of many {}.", # need
    # "a photo of the hard to see {}." # don't need
    # "graffiti of a {}." # don't need
    # "the embroidered {}."  # don't need
    # "a rendering of a {}." # don't need
    # "a tattoo of a {}." # don't need
    # "a scene of the {}." # don't need
    # "a photo of a {}, a type of scene." # don't need
    "a bright photo of a {}."
]

# Perform zero-shot classification
print(f"Templates to use: {templates}")
print(f"model: {model_name}_{pretrained}")

submission = dict({'id_idx':list(range(8100)), 'label':[]})
with torch.no_grad(), torch.cuda.amp.autocast():
    embedding_dim = model.transformer.resblocks[-1].mlp.c_proj.out_features
    text_feat_ensemble = torch.zeros((len(class_names), embedding_dim))
    for template in templates:
        text = tokenizer([template.format(class_name) for class_name in class_names])
        text_features = model.encode_text(text)
        text_feat_ensemble += text_features
    text_feat_ensemble /= len(templates)
    text_feat_ensemble /= text_feat_ensemble.norm(dim=-1, keepdim=True)

    model = model.to(device)
    for x, y in tqdm(dl):
        x = x.cuda(device)
        image_features = model.encode_image(x).to('cpu').float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        zero_shot_probs = (100.0 * image_features @ text_feat_ensemble.T).softmax(dim=-1)
        zero_shot_pred = zero_shot_probs.max(dim=-1)[1].tolist()
        submission['label'] += zero_shot_pred

# Save prediction as csv file
pd.DataFrame(submission).to_csv(os.path.join(root, f'{model_name}_{pretrained}_manual_pt.csv'), index=False)