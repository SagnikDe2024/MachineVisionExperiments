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
MachineVisionExperiments\checkpoints\tune_classifier\4_20250408T035811_-8491173734139600649_1\checkpoint_000009\model_checkpoint.pth






Using ray tune for hyperparameter optimization, the optuna searcher finally narrowed down to the following parameters.

| Parameters        | Value |
|-------------------|-------|
| batch_size        | 125   |
| cnn_layers        | 6     |
| fcn_layers        | 4     |
| final_channels    | 210   |
| learning_rate     | 0.001 |
| starting_channels | 46    |

The final model has 875,592 parameters with an estimated parameter size of 3.5 MB. 
The accuracy achieved for each of the CIFAR-10 classes is shown below.

| Class | Accuracy (%) |
|-------|--------------|
| plane | 86.3         |
| car   | 89.4         |
| bird  | 74.9         |
| cat   | 74.3         |
| deer  | 84.7         |
| dog   | 71.9         |
| frog  | 84.4         |
| horse | 81.8         |
| ship  | 93.5         |
| truck | 89.2         |

The validation loss (vloss) is 0.541896045207977.  

CIFAR-10 is extremely challenging to improve accuracy on. The classes 'cat' and 'dog' seemed to cause the most problems.
All the models achieving >90% accuracy seemed to use significantly much larger number of parameters and featured skip connections. 

A few highlights from my experimentation.
1) Batchnorm works ! There is some disagreement in the ML community whether to put them before the activations or after the activations. I mean it works really well, the validation error just got sliced to half when I removed the bias of the CNN filter and replaced with a batchnorm on the output channels.
2) Dividing a $3 \times 3$ kernel into $1 \times 3$ + $3 \times 1$ kernel is useful if the number of parameters is too high. In my case it decreased model performance as CIFAR-10 really likes more parameters.
3) FractionalMaxPool2D is significantly more flexible compared to plain old MaxPool2D. So in spite of the CNN layers downsampling the image by 16, we don't have $\log_2(16) = 4$ layers, but 6 layers. Not sure how popular this is. 
4) My intuition was that the number of features extracted will be small in the first layer like 24-32 and last CNN layer will have a lot of them like ~300.  Interestingly when experimenting with hyperparameter tuning one has the first layer extracting 46 features and the last layer extracting 210 features. 
5) Properly configuring ray tune to tune hyperparameters is an experiment in itself. If one is not careful the number of parameters will blow up quickly and training will slow down significantly. Instead of guessing the learning rate and batch size, one can use them as hyperparameters.    
6) To prevent a parameter explosion, I changed from purely minimizing vloss to acceptable amount of vloss with the minimum number of parameters. So I roughly used $\text{efficiency} =\frac{1}{\text{vloss} \times \text{parameters}}$ as working principle and redid the model tuning tighter hyperparameter bounds multiple times.

Annoyances:
1) Ray tune has a bug in `tune.quniform` where it completely ignores the quantization if the quantization value is 1. Wasted too much valuable man-hours hunting down why some of the training is erroring out.
2) The model must be compiled before training or the training will be very slow. However, compiled pytorch model not saveable (or picklable) ! Nowhere in the `torch.compile` documentation tells us that. Anyway I found a workaround. 
3) If ray tune is run on a single computer with multiple child processes launched, then the `torch.compile` might have a conflict when generating compiled code. So expect an error the first time.
4) Optuna seems to have issues resuming from an existing tuning experiment which got terminated, preferring to start a new study instead.
5) The ray tune directory naming can cause issues in Windows and it has a tendency to use illegal characters.



(Probably F1 scores will show better results).

Inspired by image generation systems like Stable Diffusion and Flux, an attempt was made to create a VAE
so that one can generate multiple samples as long as the $\sigma$ and $\mu$ 

To fully realize the project one needs 
This is very hard to explain but let us assume that there are 2 classes $\rightarrow$ [cat, tree]


1) One can use the image of the class 'cat' to generate $\sigma$ and $\mu$ to generate pictures of a cat. However, during evaluation one needs a classifier to test if the generated sample is that of a cat. This also means that a classifier needs to be designed first.
2) Down the line I realized that using images are not useful to generate $\sigma$ and $\mu$ as then it won't be able to generate picture of cat and tree at the same time. A better option use to use 
   1) a one hot encoded vector to represent the classes. Then use linear combination of the vectors to generate $\sigma$ and $\mu$.
   2) a text encoder to generate the $\sigma$ and $\mu$.
3) Looking through existing literature it is obvious that the generated images from VAEs are blurry which somehow escaped my attention during my first reading. This can confuse the classifier. There are VQ-VAEs and Normalizing Flows (NF) but the design of NF is very different from VAEs in general.
4) 

This started out as wa



