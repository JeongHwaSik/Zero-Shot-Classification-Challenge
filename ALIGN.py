import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from PIL import Image
import torch
import json
from tqdm import tqdm
from transformers import AlignProcessor, AlignModel
from PIL import Image

# Settings
root = './'

# Load ALIGN model
# Google ALIGN is not open-source model so use ALIGN model pretrained by Kakao-brain
processor = AlignProcessor.from_pretrained('kakaobrain/align-base')
model = AlignModel.from_pretrained('kakaobrain/align-base')

# Load class name list.
with open(os.path.join(root, 'Scene/classes.json'), 'r') as j:
     class_names = json.loads(j.read())

# Dictionary data structure to store results as .csv file
submission = dict({'id_idx':list(range(8100)), 'label':[]})

# 8100 images for testing: (3, 224, 224)
for i in tqdm(range(8100)):
    image = Image.open(f'./Scene/0/{i}.jpg')
    inputs = processor(
        images=image,
        text=class_names,
        images_kwargs={"crop_size": {"height": 224, "width": 224}},
        text_kwargs={"padding": "do_not_pad"},
        common_kwargs={"return_tensors": "pt"},
    )
    # print(inputs.pixel_values.shape) # (1, 32, 224, 224)

    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs.text_embeds.shape) # (num_classes, emb_dim) = (6, 640)
    # print(outputs.image_embeds.shape) # (batch_size, emb_dim) = (1, 640)

    # Image-text similarity score
    logits_per_image = outputs.logits_per_image # (1, 6)

    # Softmax to get the label probabilities
    probs = logits_per_image.softmax(dim=1) # (1, 6)
    zero_shot_pred = probs.max(dim=-1)[1].tolist()

    # add results in submission dictionary
    submission['label'] += zero_shot_pred

# store submission file as .csv file
pd.DataFrame(submission).to_csv(os.path.join(root, f'align.csv'), index=False)