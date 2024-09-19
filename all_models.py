import os
import json
import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# Settings
device = 'cuda:0'
root = './'
dataset_name = 'Scene'


for (model_name, pretrained) in open_clip.list_pretrained():

  # Load CLIP model
  model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
  tokenizer = open_clip.get_tokenizer(model_name)

  # Load test dataset
  ds = ImageFolder(os.path.join(root, dataset_name), transform=preprocess)
  ds.samples = natsorted(ds.samples)
  dl = DataLoader(ds, shuffle=False, batch_size=32, num_workers=2) # shuffle=False for submission!!


  # Load class name list
  with open(os.path.join(root, 'classes.json'), 'r') as j:
      class_names = json.loads(j.read())

  # Perform zero-shot classification
  submission = dict({'id_idx':list(range(8100)), 'label':[]})
  with torch.no_grad(), torch.cuda.amp.autocast():
      model = model.to(device)
      text = tokenizer([f"{class_name}" for class_name in class_names]).to(device) # (num_classes, embedding) = (6, 77)
      text_features = model.encode_text(text) # (6, 512)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      for x, _ in tqdm(dl): # for each mini-batch x: (32, 3, 224, 224)
          x = x.to(device)
          image_features = model.encode_image(x).float() # (batch_size, embedding) = (32, 512)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          zero_shot_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # (batch_size, num_classes) = (32, 6)
          zero_shot_pred = zero_shot_probs.max(dim=-1)[1].tolist()
          submission['label'] += zero_shot_pred

  # Save prediction as .csv file
  pd.DataFrame(submission).to_csv(os.path.join(root, f'{model_name}_{pretrained}.csv'), index=False)
  print(f"{model_name}_{pretrained}.csv save complete!!!")



  # Perform zero-shot classification for "a photo of {class}" version
  submission = dict({'id_idx':list(range(8100)), 'label':[]})
  with torch.no_grad(), torch.cuda.amp.autocast():
      model = model.to(device)
      text = tokenizer([f"a photo of a {class_name}." for class_name in class_names]).to(device) # (num_classes, embedding) = (6, 77)
      text_features = model.encode_text(text) # (6, 512)
      text_features /= text_features.norm(dim=-1, keepdim=True) # normalize for classification
      for x, _ in tqdm(dl): # (32, 3, 224, 224)
          x = x.to(device)
          image_features = model.encode_image(x).float()
          image_features /= image_features.norm(dim=-1, keepdim=True) # (batch_size, embedding) = (32, 512)
          zero_shot_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1) # (batch_size, num_classes) = (32, 6)
          zero_shot_pred = zero_shot_probs.max(dim=-1)[1].tolist()
          submission['label'] += zero_shot_pred

  # 6-1. Save prediction as .csv file for "a photo of {class}" version
  pd.DataFrame(submission).to_csv(os.path.join(root, f'{model_name}_{pretrained}_pt.csv'), index=False)
  print(f"{model_name}_{pretrained}_pt.csv save complete!!!")