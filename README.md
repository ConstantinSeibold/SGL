# Self-guiding Loss for Multiple Instance Learning
![Title Image](./imgs/supervision_types.png)

The Self-Guiding Loss is a novel multiple-instance learning loss which integrates artificial supervision based on the networks predictions into its formulation in an online step. The SGL can be seen as an extension to the standard MIL-setting. This repository contains the loss comparison studies on the MNIST-Bags dataset of the ACCV paper [**Self-Guided Multiple Instance Learning for Weakly Supervised Thoracic Disease Classification and Localization in Chest Radiographs**](https://arxiv.org).

> [**Self-Guided Multiple Instance Learning for Weakly Supervised Thoracic Disease Classification and Localization in Chest Radiographs**](https://arxiv.org)<br>
> Constantin Seibold, Jens Kleesiek, Heinz-Peter Schlemmer, Rainer Stiefelhagen<br>
> 
>
> **Abstract:** *Due to the high complexity of medical images and the scarcityof  trained  personnel,  most  large-scale  radiological  datasets  are  lackingfine-grained  annotations  and  are  often  only  described  on  image-level.These shortcomings hinder the deployment of automated diagnosis sys-tems, which require human-interpretable justification for their decisionprocess.  In  this  paper,  we  address  the  problem  of  weakly  supervisedidentification  and  localization  of  abnormalities  in  chest  radiographs  ina multiple-instance learning setting. To that end, we introduce a novelloss function for training convolutional neural networks increasing thelo-calization confidenceand assisting the overalldisease identification. Theloss leverages both image- and patch-level predictions to generate auxil-iary supervision and enables specific training at patch-level. Rather thanforming strictly binary from the predictions as done in previous loss for-mulations, we create targets in a more customized manner. This way, theloss accounts for possible misclassification of less certain instances. Weshow that the supervision provided within the proposed learning schemeleads to better performance and more precise predictions on prevalentdatasets  for  multiple-instance  learning  as  well  as  on  the  NIH  ChestX-Ray14 benchmark for disease recognition than previously used losses.*


## Contents

Available material to our paper can be found here:

| Path | Description
| :--- | :----------
| [SPL](https://github.com/ConstantinSeibold/SGL) | Main folder.
| &boxvr;&nbsp;[MNIST-Bags_Experiments](https://github.com/ConstantinSeibold/SPL/tree/master/MNIST-Bags_Experiments) | Contains the proposed loss formulation for both Tensorflow and Pytorch


## Citation
If you use this work or dataset, please cite:
```latex
@inproceedings{sgl,
  title={Self-Guided Multiple Instance Learning for Weakly Supervised Thoracic Disease Classification and Localization in Chest Radiographs},
  author={Seibold, Constantin and Kleesiek, Jens and Schlemmer, Heinz-Peter and Stiefelhagen, Rainer},
  booktitle={Asian Conference on Computer Vision},
  year={2020},
  organization={Springer}
}
