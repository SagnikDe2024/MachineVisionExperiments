This repository contains some experiments and designs for machine vision, image generation experiments.

Currently there is an image classification and one image encode - decoder (WIP) related model being designed.

The classification model went through various changes until we finally go

The classification is done on CIFAR-10.
The classification model consists of two parts, a set a CNN layers and then a FCN layer which narrows down to 10 items.


The tuning is done with the help of Ray Tune (ASHAScheduler and Optuna space search).


2025-04-08 14:34:13,696 - INFO - [classifier.py:__init__] Classifier channel upscale ratio: 1.3548533596918833
2025-04-08 14:34:13,696 - INFO - [classifier.py:__init__] Layers = 6, downsampled_sizes = [20, 13, 8, 5, 3, 2], channels = [3, 46, 62, 84, 114, 155, 210]
2025-04-08 14:34:13,705 - INFO - [classifier.py:__init__] FCN layers: [840, 277, 92, 30, 10]
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
Classifier                                    [100, 10]                 --
├─Encoder: 1-1                                --                        --
│    └─Sequential: 2-1                        --                        --
│    │    └─Conv2d: 3-1                       [100, 46, 32, 32]         1,288
│    │    └─BatchNorm2d: 3-2                  [100, 46, 32, 32]         92
│    │    └─Mish: 3-3                         [100, 46, 32, 32]         --
│    │    └─FractionalMaxPool2d: 3-4          [100, 46, 20, 20]         --
│    │    └─Conv2d: 3-5                       [100, 62, 20, 20]         25,730
│    │    └─BatchNorm2d: 3-6                  [100, 62, 20, 20]         124
│    │    └─Mish: 3-7                         [100, 62, 20, 20]         --
│    │    └─FractionalMaxPool2d: 3-8          [100, 62, 13, 13]         --
│    │    └─Conv2d: 3-9                       [100, 84, 13, 13]         46,956
│    │    └─BatchNorm2d: 3-10                 [100, 84, 13, 13]         168
│    │    └─Mish: 3-11                        [100, 84, 13, 13]         --
│    │    └─FractionalMaxPool2d: 3-12         [100, 84, 8, 8]           --
│    │    └─Conv2d: 3-13                      [100, 114, 8, 8]          86,298
│    │    └─BatchNorm2d: 3-14                 [100, 114, 8, 8]          228
│    │    └─Mish: 3-15                        [100, 114, 8, 8]          --
│    │    └─FractionalMaxPool2d: 3-16         [100, 114, 5, 5]          --
│    │    └─Conv2d: 3-17                      [100, 155, 5, 5]          159,185
│    │    └─BatchNorm2d: 3-18                 [100, 155, 5, 5]          310
│    │    └─Mish: 3-19                        [100, 155, 5, 5]          --
│    │    └─FractionalMaxPool2d: 3-20         [100, 155, 3, 3]          --
│    │    └─Conv2d: 3-21                      [100, 210, 3, 3]          293,160
│    │    └─BatchNorm2d: 3-22                 [100, 210, 3, 3]          420
│    │    └─Mish: 3-23                        [100, 210, 3, 3]          --
│    │    └─FractionalMaxPool2d: 3-24         [100, 210, 2, 2]          --
├─Sequential: 1-2                             [100, 10]                 --
│    └─Mish: 2-2                              [100, 210, 2, 2]          --
│    └─Flatten: 2-3                           [100, 840]                --
│    └─Linear: 2-4                            [100, 277]                232,957
│    └─Mish: 2-5                              [100, 277]                --
│    └─Linear: 2-6                            [100, 92]                 25,576
│    └─Mish: 2-7                              [100, 92]                 --
│    └─Linear: 2-8                            [100, 30]                 2,790
│    └─Mish: 2-9                              [100, 30]                 --
│    └─Linear: 2-10                           [100, 10]                 310
├─Softmax: 1-3                                [100, 10]                 --
===============================================================================================
Total params: 875,592
Trainable params: 875,592
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 3.20
===============================================================================================
Input size (MB): 1.23
Forward/backward pass size (MB): 158.98
Params size (MB): 3.50
Estimated Total Size (MB): 163.72
===============================================================================================
2025-04-08 14:34:25,611 - INFO - [tune_classifier.py:load_cifar_dataset] 50000 training samples and 10000 validation samples
2025-04-08 14:34:25,611 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Showing perf of C:\mywork\python\MachineVisionExperiments\checkpoints\tune_classifier\4_20250408T035811_-8491173734139600649_1\checkpoint_000009\model_checkpoint.pth
2025-04-08 14:34:52,281 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: plane is 86.3 %
2025-04-08 14:34:52,281 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: car   is 89.4 %
2025-04-08 14:34:52,281 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: bird  is 74.9 %
2025-04-08 14:34:52,281 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: cat   is 74.3 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: deer  is 84.7 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: dog   is 71.9 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: frog  is 84.4 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: horse is 81.8 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: ship  is 93.5 %
2025-04-08 14:34:52,282 - INFO - [visualize_classifier_results.py:get_state_and_show_accuracy] Accuracy for class: truck is 89.2 %
