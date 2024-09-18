# Zero-shot Classification Challenge

#### 🎉🎉 Achieved 5th place in the challenge 🎉🎉 
</br>

<img width="1046" alt="Screenshot 2024-09-18 at 8 39 24 PM" src="https://github.com/user-attachments/assets/4a62068b-73af-4a7f-8c02-73d422da3041">

</br>
</br>

## 1. Challenge Overview
The purpose of this challenge is to predict the labels of a given test image dataset using a Vision Language Model (VLM). **In this case, the VLM model must not be trained on the provided test data.** The goal of this challenge is to enhance the performance of the model in recognizing image datasets it has never seen before (zero-shot classification) using the VLM model.

- Data Composition: The challenge dataset is for zero-shot classification and consists of 8,100 images and 6 classes.

|Category|Statistics|
|------|---|
|number of classes|6|
|number of samples|8100|
|image size|(224, 224, 3)|

- Zero-shot Classification: Zero-shot classification is a task where the model predicts classes it has not seen during training. A typical example is CLIP, a huge pre-trained model. Therefore, using the challenge-provided data or the original data for model training (including supervised, unsupervised, and self-supervised) is prohibited.

- Predicting Test Data: The labels of the test data should be predicted using the trained model. One label should be predicted for each test image. The predicted test labels should be submitted to Kaggle in CSV format, where they will be automatically scored. The test labels must be predicted by the trained model, and manual labeling by a human is not allowed.

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

### 3-0. ALIGN model by Kakao brain
[ALIGN](https://arxiv.org/pdf/2102.05918) is another model for zero-shot classification proposed by Google. While the original ALIGN model is not publicly available, Kakao Brain has provided an open-sourced pre-trained ALIGN model that offers performance similar to Google's ALIGN model. By using pre-trained ALIGN model 

```
python3 ALIGN.py
```

### 3-1. Various pretrained CLIP models
According to OpenClip library, there are over 100 pretrained CLIP models 
```
python3 all_models.py
```

### 3-2. Manual prompt tuning
```
python3 manual_prompt_tuning.py
```

### 3-3. Linear probing with ImageNet (or blurred ImageNet) and SUN397 (or blurred SUN397)

### 3-4. Ensemble ALL
