# Zero-shot Classification Challenge

#### 🎉🎉 Achieved 5th place with a private score of 90.617 and a public score of 90.760 🎉🎉
</br>

<img width="1046" alt="Screenshot 2024-09-18 at 8 39 24 PM" src="https://github.com/user-attachments/assets/4a62068b-73af-4a7f-8c02-73d422da3041">

</br>
</br>

## 1. Challenge Overview
The purpose of this challenge is to predict the labels of a given test image dataset using a Vision Language Model (VLM). **In this case, the VLM model must not be trained on the provided test data.** The goal of this challenge is to enhance the performance of the model in recognizing image datasets it has never seen before (zero-shot classification) using the VLM model.

- Data Composition: The challenge dataset is for zero-shot classification and consists of 8,100 images and 6 classes. I uploaded some images in the Scene folder.
<div align="center">

|Category|Statistics|
|------|---|
|number of classes|6|
|number of samples|8100|
|image size|(224, 224, 3)|

</div>

- Zero-shot Classification: Zero-shot classification is a task where the model predicts classes it has not seen during training. A typical example is CLIP, a huge pre-trained model. Therefore, using the challenge-provided data or the original data for model training (including supervised, unsupervised, and self-supervised) is prohibited.

- Predicting Test Data: The labels of the test data should be predicted using the trained model. One label should be predicted for each test image. The predicted test labels should be submitted to Kaggle in CSV format, where they will be automatically scored. The test labels must be predicted by the trained model and manual labeling by a human is not allowed.

</br>

## 2. Prerequisites
### 2-1. Contrastive Language-Image Pre-training (CLIP)
[CLIP](https://arxiv.org/pdf/2103.00020), which stands for Contrastive Language-Image Pre-training, learns from natural language supervision rather than relying on traditional 1-of-N majority voting (gold-label) used in earlier image classification tasks. By leveraging vast amounts of language-image data from the Internet, CLIP not only learns image representations but also links them to natural language, providing exceptional flexibility and significantly outperforming previous zero-shot classification models. The key to this natural language supervision is the large-scale data, with CLIP having been trained on over 400 million image-text pairs collected online. 

During pre-training, each text is paired with its corresponding image, and together they are used to jointly train both an image and text encoder. The objective is to maximize the cosine similarity between the embeddings of correct image-text pairs $(B)$, while minimizing it for incorrect pairs $(B^2−B)$. At test time, candidate labels with prompts are fed into the text encoder, while the image is processed by the image encoder. The dot product of the image and text embeddings, followed by a softmax operation, produces a final score for each image. Final output will have a shape of $(B, num classes)$.

<img width="906" alt="Screenshot 2024-09-18 at 9 14 30 PM" src="https://github.com/user-attachments/assets/aa6c5a89-7329-4d28-9918-434207bb7b23">

### 2-2. Various approaches to improve CLIP
There are several ways to enhance the CLIP model for downstream tasks. Here are three approaches.

- **Fine-tuning whole CLIP model**: [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf) aims to combine well-fine-tuned CLIP models with the original zero-shot CLIP models by sharing parameters through a weighted averaging method.
  
- **Linear probing with Adapters**: [CLIP-Adapter](https://arxiv.org/pdf/2110.04544) adds two linear layers to the final text and image encoders, using a few-shot linear probing approach. On the other hand [Tip-Adapter](https://arxiv.org/pdf/2111.03930) introduces a training-free CLIP adapter by directly setting the adapter's weights using a cache model, thereby bypassing the traditional SGD linear probing. This cached weight can then further enhance the performance when combined with a few-shot linear probing.
  
- **Prompt tuning**: Prompt tuning is one of the most effective way to improve the performance of the model. [CoOp](https://arxiv.org/pdf/2109.01134) replaces text prompts with learnable vectors so that prompt vectors can be optimized in a few-shot manner. [CoCoOp](https://arxiv.org/pdf/2203.05557) builds on this by using instance-specific prompts that are conditioned on the input image. Additionally, there is a research exploring the use of Large Language Model (LLM) as a prompt.

But the challenge does not allow for the use of few-shot methods to improve the performance, meaning that the aforementioned approaches cannot be applied for enhancing zero-shot classification. 😭

</br>

## 3. Experiments

### 3-0. ALIGN model
[ALIGN](https://arxiv.org/pdf/2102.05918) is another model for zero-shot classification proposed by Google. While the original ALIGN model is not publicly available, Kakao Brain has provided an open-sourced pre-trained ALIGN model that offers performance similar to Google's ALIGN model. The ALIGN model's performance on the test dataset is 61.530%, which is bad compared to various CLIP models.

Here's the script for ALIGN model.
```
python3 ALIGN.py
```

### 3-1. Various pretrained CLIP models
According to OpenClip library, there are over 100 pre-trained CLIP models [(See here)](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv). The table below shows the performance of different CLIP models with two different prompts: "{class}" and "a photo of {class}". The results showed that the "ViT-bigG-14-CLIPA" model, pre-trained with the "Datacomp1b" dataset and using the prompt "{class}", achieved the highest score on the test dataset. The results also indicated that the effectiveness of prompts is model-dependent.

|Top-1 Acc.|# Params. (M)|CLIP Model|Pretrained|Prompt tuning|
|---|---|---|---|---|
|0.67382|351.77|convnext_large_d_320|laion2b_s29b_b131k_ft|“{class}”|
|0.75604|351.77|convnext_large_d_320|laion2b_s29b_b131k_ft|“a photo of {class}”|
|0.73358|351.77|convnext_large_d_320|laion2b_s29b_b131k_ft_soup|“{class}”|
|0.79160|351.77|convnext_large_d_320|laion2b_s29b_b131k_ft_soup|“a photo of {class}”|
|0.72592|149.62|ViT-B-16|openai|“{class}”|
|0.67333|149.62|ViT-B-16|openai|“a photo of {class}”|
|0.80814|149.62|ViT-B-16|Dfn2b|“{class}”|
|0.81654|149.62|ViT-B-16|Dfn2b|“a photo of {class}”|
|0.84740|427.62|ViT-L-14|datacomp_xl_s13b_b90k|“{class}”|
|0.82074|427.62|ViT-L-14|datacomp_xl_s13b_b90k|“a photo of {class}”|
|0.84691|427.62|ViT-L-14|commonpool_xl_clip_s13b_b90k|“{class}”|
|0.84740|427.62|ViT-L-14|commonpool_xl_clip_s13b_b90k|“a photo of {class}”|
|0.83234|427.62|ViT-L-14|commonpool_xl_laion_s13b_b90k|“{class}”|
|0.84765|427.62|ViT-L-14|commonpool_xl_laion_s13b_b90k|“a photo of {class}”|
|0.86987|427.62|ViT-L-14-quickgelu|metaclip_fullcc|“{class}”|
|0.85012|427.62|ViT-L-14-quickgelu|metaclip_fullcc|“a photo of {class}”|
|0.86098|427.62|ViT-L-14-quickgelu|Dfn2b|“{class}”|
|0.86419|427.62|ViT-L-14-quickgelu|Dfn2b|“a photo of {class}”|
|0.84987|986.11|ViT-H-14-quickgelu|Dfn5b|“{class}”|
|0.86814|986.11|ViT-H-14-quickgelu|Dfn5b|“a photo of {class}”|
|0.86913|986.11|ViT-H-14-quickgelu|metaclip_fullcc|“{class}”|
|0.86074|986.11|ViT-H-14-quickgelu|metaclip_fullcc|“a photo of {class}”|
|0.84197|986.71|ViT-H-14-378-quickgelu|Dfn5b|“{class}”|
|0.85975|986.71|ViT-H-14-378-quickgelu|Dfn5b|“a photo of {class}”|
|0.85259|968.64|ViT-H-14-CLIPA-336|Datacomp1b|“{class}”|
|0.87703|968.64|ViT-H-14-CLIPA-336|Datacomp1b|“a photo of {class}”|
|0.78246|877.36|ViT-SO400M-14-SigLIP-384|Webli|“{class}”|
|0.77728|877.36|ViT-SO400M-14-SigLIP-384|Webli|“a photo of {class}”|
|0.79506|2539.57|ViT-bigG-14|laion2b_s39b_b160k|“{class}”|
|0.78932|2539.57|ViT-bigG-14|laion2b_s39b_b160k|“a photo of {class}”|
|**0.88617**|**2517.22**|**ViT-bigG-14-CLIPA**|**Datacomp1b**|**“{class}”**|
|0.87259|2517.22|ViT-bigG-14-CLIPA|Datacomp1b|“a photo of {class}”|
|0.88172|2517.76|ViT-bigG-14-CLIPA-336|Datacomp1b|“{class}”|
|0.87061|2517.76|ViT-bigG-14-CLIPA-336|Datacomp1b|“a photo of {class}”|

Here are the other experimental results. The first two rows show the results of weighted averaging the final layers from two different image encoders. (This is inspired by WiSE-FT paper.) Unsurprisingly, the results demonstrated that weighted averaging two different linear layers had nearly the same effect as averaging the output performance of two models. The third and fourth rows present the results of ensembling all prompts provided [here](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb), which outperformed using only a single prompt. So in section 3.2, I attempted to add some useful manual prompts through trial and error.

|Top-1 Acc.|# Params. (M)|CLIP Model|Pretrained|Prompt tuning|
|---|---|---|---|---|
|0.81481|149.62|ViT-B-16|0.9*dfn2b + 0.1*openai|“{class}”|
|0.86839|427.62|ViT-L-14-quickgelu|0.95*metaclip_fullcc + 0.05*Dfn2b|“{class}”|
|**0.89827**|**2517.22**|**ViT-bigG-14-CLIPA**|**Datacomp1b**|**Prompt ensemble**|
|0.86938|2517.22|ViT-bigG-14-CLIPA-quickgelu|Datacomp1b|Prompt ensemble|

Here's the script for all given CLIP models.
```
python3 all_models.py
```

### 3-2. Manual prompt tuning

From section 3-1, I learned that ensembling some useful prompts can enhance the performance of the CLIP model. Therefore, I selected the "ViT-bigG-14-CLIPA" model, pre-trained with the "Datacomp1b" dataset, and manually experimented with various prompts. Through trial and error, I found that while ensembling prompts does improve performance, the effectiveness is really slight.

|Top-1 Acc.|Models|Pretrained|prompt|
|---|---|---|---|
|0.88617|ViT-bigG-14-CLIPA|Datacomp1b|[{}]|
|0.89086|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.89037|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.89135|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.88913|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.88913|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.88938|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.89061|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.89358|ViT-bigG-14-CLIPA|Datacomp1b|['itap of a {}.', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a photo of the small {}.']|
|0.89160|ViT-bigG-14-CLIPA|Datacomp1b|['a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a blurry photo of the {}.']|
|0.89432|ViT-bigG-14-CLIPA|Datacomp1b|['a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.']|
|0.89111|ViT-bigG-14-CLIPA|Datacomp1b|['a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a cropped photo of the {}.']|
|0.89506|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.']|
|0.89382|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'the embroidered {}.']|
|0.89432|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a dark photo of the {}.']|
|0.89234|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a low resolution photo of the {}.']|
|**0.89506**|**ViT-bigG-14-CLIPA**|**Datacomp1b**|**['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.']**|
|0.89209|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'a photo of the hard to see {}.']|
|0.89456|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'graffiti of a {}.']|
|0.89234|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'the embroidered {}.']|
|0.89308|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'a rendering of a {}.']|
|0.89382|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'a tattoo of a {}.']|
|0.89333|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'a scene of the {}.']|
|0.89259|ViT-bigG-14-CLIPA|Datacomp1b|['{}', 'a bad photo of the {}.', 'a origami {}.', 'a photo of the large {}.', 'a {} in a video game.', 'art of the {}.', 'a view of the {}.', 'a photo of many {}.', 'a photo of a {}, a type of scene.']|


Here's the script for manually tuning prompts.
```
python3 manual_prompt_tuning.py
```

### 3-3. Linear probing with ImageNet and SUN397
The CLIP model is pre-trained on a vast dataset from the Internet, which helps enhance its generalizability. However, this large-scale training data can also introduce some noise into the model. To address this issue, I attempted to train additional adapters using the ImageNet and SUN397 datasets. (In this experiment, I use CLIP-Adapter.) Contrary to what I expected, the results were lower than those of the original CLIP model and I discovered that there is something unique about the test dataset.

|Top-1 Acc.|Models|Dataset|k-shot|Epochs|prompt|
|---|---|---|---|---|---|
|0.88049|ViT-bigG-14-CLIPA|ImageNet|Full|16|"a photo of a {class}."|
|0.88421|ViT-bigG-14-CLIPA|ImageNet|Full|32|"a photo of a {class}."|
|0.84197|ViT-bigG-14-CLIPA|ImageNet|16-shot|200|"a photo of a {class}."|
|0.84197|ViT-bigG-14-CLIPA|ImageNet|32-shot|200|"a photo of a {class}."|
|0.83555|ViT-bigG-14-CLIPA|SUN397|Full|16|"a photo of a {class}."|
|0.82666|ViT-bigG-14-CLIPA|SUN397|Full|32|"a photo of a {class}."|
|0.84987|ViT-bigG-14-CLIPA|SUN397|16-shot|200|"a photo of a {class}."|
|0.84197|ViT-bigG-14-CLIPA|SUN397|32-shot|200|"a photo of a {class}."|

Here's the scripts for linear probing adapters.
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=ImageNet k_shot=16 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=ImageNet k_shot=32
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=ImageNet k_shot=full 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=SUN397 k_shot=16 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=SUN397 k_shot=32
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=linear_adapter ft_dataset=SUN397 k_shot=full 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=ImageNet k_shot=16 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=ImageNet k_shot=32
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=ImageNet k_shot=full 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=SUN397 k_shot=16 
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=SUN397 k_shot=32
```
```
python3 main.py model_name=ViT-bigG-14-CLIPA pretrained=datacomp1b linear_probing=True adapter=clip_adapter ft_dataset=SUN397 k_shot=full 
```

### 3-4. Linear probing with "blurred" ImageNet and "blurred" SUN397

|ImageNet Image with shape (3, 224, 224)|Challenge Image with shape (3, 224, 224)|
|---|---|
|<img src="https://github.com/user-attachments/assets/39c9f1e7-83fd-4cea-9e31-be2f2ad482cb" width="400" height="400"/>|<img src="https://github.com/user-attachments/assets/0fa75233-c893-4580-be9a-8549bc052dbc" width="400" height="400"/>|

In section 3-3, I discovered that the test dataset images are quite blurry. This led me to consider using a "blurry" version of the ImageNet or SUN397 datasets to improve the model's performance on the test dataset. However, processing Gaussian blur for all images in the dataset proved to be a significant bottleneck, taking nearly a week for linear probing with the blurry ImageNet dataset. Due to the time constraints of this challenge, I decided to test only on the "blurry" SUN397 dataset, which yielded better results than those from section 3-3. When combining the results from section 3-3 and 3-4, I estimate that linear probing with a "blurry" ImageNet would have achieved results above 90%(🌟). 

|Top-1 Acc.|Models|Pretrained|prompt|
|---|---|---|---|
|0.86049|ViT-bigG-14-CLIPA|Blurred SUN397|Full|16|“{class}”|
|🌟|ViT-bigG-14-CLIPA|Blurred ImageNet|Full|16|“{class}”|


### 3-5. Ensemble ALL
