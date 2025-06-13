# RALF reasoning: Multi-Layer LLM Prompting for Open-Vocabulary Object Detection
> Chia-Yu Wu, Wei Huang, Yen-Yu Lin, Chien-Yao Wang
> 
> National Yang Ming Chiao Tung University
> 

Our project is based on the paper "[Retrieval-Augmented Open-Vocabulary Object Detection](https://arxiv.org/abs/2404.05687)", which introduces the RALF method to boost model accuracy in open-vocabulary object detection. You can find the original paper's implementation on [GitHub](https://github.com/mlvlab/RALF/tree/main?tab=readme-ov-file)

## Introduction
![RAF flow chart](Figures/RAF%20flow%20chart.png)
RALF comprises two key components: RAL (Retrieval-Augmented Losses) and RAF ( Retrieval-Augmented visual Features). Our work focuses on modifying RAF to enable the model to generate more precise and visually relevant concepts.

We add an additional LLM in RAF pipeline. This two-LLM structure can do reasoning and therefore generate better concepts. In the first LLM, we ask it to list some nouns that may appear similar to {vocabulary}. For example, for vocabulary "truck", the LLM will return "train" and "bus". Then we pass these nouns to the second LLM and ask it to point out the visual differences between {vocabulary} and {synonym}. For example, with vocabulary "truck" and synonmy "train", "bus", the LLM will output "Multiple carriages, No large tires" for train and "Rectangular shape" for bus. Save these outputs as the concepts and use them to augment the visual fearures.

## Preparation
Please follow the README in [OADP](OADP/README.md) and [RAF](RAF/README.md) folder to prepare the files needed.

## Run code
### RAF training
In RAF folder
```
cd RAF
python txt_to_pkl.py
```

### RAL training
In OADP folder
```
torchrun --nproc_per_node=4 -m oadp.dp.train coco_ral ./configs/dp/ralf/ral/coco_ral.py
```

### RALF inference
In OADP folder
```
torchrun --nproc_per_node=4 -m oadp.dp.test ./configs/dp/ralf/raf/coco_raf.py work_dirs/coco_ral/iter_32000.pth
```

---
The checkpoints for RAL are available [here](https://drive.google.com/drive/folders/1ptNaoSlbvP4CXFXrI2gySCwtaiH3mOwA). And the checkpoints for RAF are available [here](https://drive.google.com/drive/folders/1VdJMaqtvNDnUz4jS7xURUVzVJ_anFrve?usp=sharing).

## Results
### OADP on COCO
|Model|mAP|mAP score gain|
|---|---|---|
|Baseline| 48.10 | - |
|RALF| 48.96 | +0.86 |
|RALF + our structure| 49.09 | +0.13 |