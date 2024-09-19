import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tqdm
import hydra
import json
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from data import create_dataset
from adapters.clip_adapter import CLIP_Adapter
from adapters.linear_adapter import Linear_Adapter
from model import create_model_tokenizer


@hydra.main(config_path="config", config_name="base", version_base="1.3")
def main(cfg):

    # Model, Tokenizer and Transform methods
    model, tokenizer, transform = create_model_tokenizer(cfg.model_name, cfg.pretrained)

    if cfg.linear_probing:
        # Get dataset for linear probing
        train_loader, class_names = create_dataset(cfg.ft_dataset, cfg.root, transform, cfg.k_shot, cfg.batch_size, blurred=cfg.blurred)

        # Linear probing architecture
        if cfg.adapter.adapter_name == 'linear_adapter':
            adapter = Linear_Adapter(c_in=model.visual.proj.shape[1]).to(cfg.device)
        elif cfg.adapter.adapter_name == 'clip_adapter':
            adapter = CLIP_Adapter(c_in=model.visual.proj.shape[1], reduction=cfg.reduction).to(cfg.device)
        else:
            raise ValueError(f"Adapter {cfg.adapter.adapter_name} not supported")

        # Optimizer, Scheduler
        optimizer = optim.Adam(adapter.parameters(), lr=cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=4e-5)

        # Linear Probing(training) start
        train(model, tokenizer, train_loader, class_names, adapter, optimizer, scheduler, cfg)

        # Test with 'model + linear_probing layer' using challenge dataset (Scene Dataset)
        test(model, adapter, tokenizer, transform, cfg)

    else:
        # Test with 'model' using challenge dataset (Scene Dataset)
        test(model, None, tokenizer, transform, cfg)




def train(model, tokenizer, train_loader, class_names, adapter, optimizer, scheduler, cfg):

    # Linear Probing(training) start!!!
    print(f"{cfg.model_name}_{cfg.pretrained}: linear_probing with {cfg.ft_dataset} started!!!")

    for epoch in range(cfg.epochs):

        print(f"{cfg.model_name}_{cfg.pretrained}_{cfg.ft_dataset}_{cfg.k_shot}shot_ep{epochs}: {epoch}/{cfg.epochs} ongoing...")

        with torch.no_grad(), torch.cuda.amp.autocast():
            model = model.to(cfg.device)
            text = tokenizer([f"{class_name}" for class_name in class_names]).to(cfg.device)
            text_features = model.encode_text(text)
            norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (1000, 512)

        for x, y in tqdm(train_loader):
            with torch.no_grad(), torch.cuda.amp.autocast():
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                image_features = model.encode_image(x).float()

            img_out = adapter(image_features)  # (32, 512)
            image_features = cfg.alpha * img_out + (1 - cfg.alpha) * image_features

            norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (128, 512)

            logits = norm_image_features @ norm_text_features.T  # (batch_size, num_classes) # (128, 1000)
            loss = nn.CrossEntropyLoss()(logits, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        if cfg.save_arch:
            # Save parameters of linear probing architecture
            torch.save(adapter,
                       os.path.join(cfg.root, f'model/{cfg.model_name}_{cfg.pretrained}_{cfg.ft_dataset}_{cfg.k_shot}shot_ep{cfg.epochs}_visual_adapter.pt'))
            torch.save(adapter.state_dict(), os.path.join(cfg.root,
                                                                 f'model/{cfg.model_name}_{cfg.pretrained}_{cfg.ft_dataset}_{cfg.k_shot}shot_ep{cfg.epochs}_visual_adapter_state_dict.pt'))



def test(model, adapter, tokenizer, transform, cfg):
    # Get test dataset (Scene Dataset)
    ds = ImageFolder(os.path.join(cfg.root, cfg.test_dataset), transform=transform)
    ds.samples = natsorted(ds.samples)
    dl = DataLoader(ds, shuffle=False, batch_size=32, num_workers=2)

    # Load class name list
    with open(os.path.join(cfg.root, f'{cfg.test_dataset}/classes.json'), 'r') as j:
        class_names = json.loads(j.read())

    # Perform zero-shot classification
    submission = dict({'id_idx': list(range(8100)), 'label': []})
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = tokenizer([f"{class_name}" for class_name in class_names]).to(cfg.device)
        model = model.to(cfg.device)
        text_features = model.encode_text(text)
        norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for x, _ in tqdm(dl):
            x = x.to(cfg.device)
            image_features = model.encode_image(x).float()

            if adapter:
                img_out = adapter(image_features)
                image_features = cfg.alpha * img_out + (1 - cfg.alpha) * image_features

            norm_image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            zero_shot_probs = (100.0 * norm_image_features @ norm_text_features.T).softmax(dim=-1)
            zero_shot_pred = zero_shot_probs.max(dim=-1)[1].tolist()
            submission['label'] += zero_shot_pred

    # Save prediction as .csv file.
    pd.DataFrame(submission).to_csv(
        os.path.join(cfg.root, f'results/{cfg.model_name}_{cfg.pretrained}_{cfg.ft_dataset}_{cfg.k_shot}shot_ep{cfg.epochs}.csv'), index=False)


if __name__ == '__main__':
    main()
