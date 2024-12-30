# DHR-CLIP
>
> by [Jiyul Ham](), [Jun-Geal Baek]().

### [paper]()

## Abstract
Zero-shot anomaly segmentation (ZSAS) is crucial for detecting and localizing defects in target datasets without need for training samples. This approach is particularly valuable in industrial quality control, where there are distributional shifts between training and operational environments or when data access is restricted. Recent vision-language models have demonstrated strong zero-shot performance across various visual tasks. However, the variations in the granularity of local anomaly regions due to resolution changes and their focus on class semantics make it challenging to directly apply them to ZSAS. To address these issues, we propose DHR-CLIP, a novel approach that incorporates dynamic high-resolution processing to enhance ZSAS in industrial inspection tasks. Additionally, we adapt object-agnostic prompt design to detect normal and anomalous patterns without relying on specific object semantics. Finally, we implement deep-text prompt tuning in the text encoder for refined textual representations and employ V-V attention layers in the vision encoder to capture detailed local features. Our integrated framework enables effective identification of fine-grained anomalies through refinement of image and text prompt design, providing precise localization of defects. The effectiveness of DHR-CLIP has been demonstrated through comprehensive experiments on real-world industrial datasets, MVTecAD and VisA, achieving strong performance and generalization capabilities across diverse industrial scenarios.  

## Overview of DHR-CLIP
![overview](https://github.com/YUL-git/DHR-CLIP/blob/main/asset/figure_2.png)

## Qualitative results
![overview](https://github.com/YUL-git/DHR-CLIP/blob/main/asset/figure_3.png)
