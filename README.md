# RALF reasoning: Multi-Layer LLM Prompting for Open-Vocabulary Object Detection
> Chia-Yu Wu, Wei Huang
> 
> National Yang Ming Chiao Tung University
> 
Our project is based on the paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)". Here is its [GitHub](https://github.com/mlvlab/RALF/tree/main?tab=readme-ov-file)

## Introduction
![RAF flow chart](Figures/RAF%20flow%20chart.png)
The paper proposed a method called RALF to inhance the model's accuracy on open-vocabulary object detection. The method includes RAL and RAF, and our project is to modify RAF to make the model generate better concepts.

We add one more LLM in RAF pipeline. This two-LLM structure can do reasoning and therefore generate better concepts. In the first LLM, we ask it to list some nouns that may appear similar to {vocabulary}. For example, for vocabulary "truck", the LLM will return "train" and "bus". Then we pass these nouns to the second LLM and ask it to point out the visual differences between {vocabulary} and {synonym}. For example, with vocabulary "truck" and synonmy "train", "bus", the LLM will output "Multiple carriages, No large tires" for train and "Rectangular shape" for bus. Save these outputs as the concepts and use them to augment the visual fearures.

## Installation
Following the [RALF documentation](https://github.com/mlvlab/RALF/tree/OADP?tab=readme-ov-file) to setup the environment.

## Preparation
Following the [OADP documentation](https://github.com/LutingWang/OADP/blob/main/README.md), prepare the data for baseline training as shown below.
```
~/OADP
    ├── pretrained
    └── data
        ├── coco
        ├── lvis_v1
        └── prompts
```

## Run code
### RAL training
```
torchrun --nproc_per_node=4 -m oadp.dp.train coco_ral ./configs/dp/ralf/ral/coco_ral.py
```

### RALF inference
```
torchrun --nproc_per_node=4 -m oadp.dp.test ./configs/dp/ralf/raf/coco_raf.py work_dirs/coco_ral/iter_32000.pth
```

## Results
### OADP on COCO
|Model|mAP|mAP score gain|
|---|---|---|
|Baseline| 48.10 | - |
|RALF| 48.96 | +0.86 |
|RALF + our structure| 49.09 | +0.13 |