## Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types
[wp](https://research.google/pubs/factors-of-influence-for-transfer-learning-across-diverse-appearance-domains-and-task-types/)

### Transfer learning protocl for experiments
    1) we train a model on ILSVRC’12 classification.
    2) we copy the weights of the backbone of the ILSVRC’12 classification model to the source model. We randomly initialize the head of the source model, which is specific
    to the task type.
    3) we train on the source task.
    4) we copy the weights of the backbone of the source
    model to the target model. Again, we randomly ini-
    tialize its head.
    5) we train on the target training set.
    6) we evaluate on the target validation set.
    This protocol essentially defines a transfer chain: ILSVRC’12 → source task → target task. We compare these transfer chains to the default practice: ILSVRC’12 → target task.

### Data normalization and augmentation

In this paper we do not want data normalization and augmentation to be confounding factors in transfer learning experiments. Hence we aim to keep data normalization and augmentation as simple as possible without compromising on accuracy, and apply the same protocol to all experiments.

We found the following augmentations to al- ways have a positive (or neutral) effect and thus use them in all experiments: (1) random horizontal flipping of the image; (2) random rescaling of the image; and (3) taking a random crop. For object detection and keypoint detection, we consider only random crops fully containing at least one object. As in [65], we found that the image scales that lead to the best performance are intrinsic to the dataset, and hence dataset-dependent

### Network Arch

Transfer learning through pre-training is only possible when the models for all task types share the same backbone architecture