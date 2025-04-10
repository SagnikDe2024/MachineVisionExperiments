This repository contains some experiments I am doing on images, different designs for machine vision, image generation
experiments.

Currently, there is an image classification and one image encode - decoder (WIP) related model being designed.

The classification model went through various changes until we finally go

The classification is done on CIFAR-10.
The classification model consists of two parts, a set a CNN layers and then a FCN layer which narrows down to 10 items.

The tuning is done with the help of Ray Tune (ASHAScheduler and Optuna space search).

Classifier channel upscale ratio: 1.3548533596918833
Layers = 6, downsampled_sizes = [20, 13, 8, 5, 3, 2], channels = [3, 46, 62, 84, 114, 155, 210]
FCN layers: [840, 277, 92, 30, 10]

MachineVisionExperiments\checkpoints\tune_classifier\4_20250408T035811_
-8491173734139600649_1\checkpoint_000009\model_checkpoint.pth

Using ray tune for hyperparameter optimization, the optuna searcher finally narrowed down to the following parameters.

| Parameters        | Value |
|-------------------|-------|
| batch_size        | 125   |
| cnn_layers        | 6     |
| fcn_layers        | 4     |
| final_channels    | 210   |
| learning_rate     | 0.001 |
| starting_channels | 46    |

The final model has 875,592 parameters with an estimated parameter size of 3.1 MB. A large number of parameters ~60% are
taken up by the last CNN (293160) and the first FCN (232957) layer.
The accuracy achieved for each of the CIFAR-10 classes is shown
below.^[Probably F1 is a better indication. Just using accuracy as everyone else seems to be using it]

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
All the models achieving >90% accuracy seemed to use significantly much larger number of parameters and featured skip
connections.
Since I am just experimenting, I tried to have acceptable vloss while minimizing the number of parameters as much as
possible.

A few highlights from my experimentation.

1) Batchnorm works ! There is some disagreement in the ML community whether to put them before the activations or after
   the activations. I mean it works really well, the validation error just got sliced to half when I removed the bias of
   the CNN filter and replaced with a batchnorm on the output channels.
2) Dividing a $3 \times 3$ kernel into $1 \times 3$ + $3 \times 1$ kernel is useful if the number of parameters is too
   high. In my case it decreased model performance as CIFAR-10 really likes more parameters.
3) FractionalMaxPool2D is significantly more flexible compared to plain old MaxPool2D. So in spite of the CNN layers
   downsampling the image by 16, we don't have $\log_2(16) = 4$ layers, but 6 layers. Not sure how popular this is.
4) My intuition was that the number of features extracted will be small in the first layer like 24-32 and last CNN layer
   will have a lot of them like ~300. Interestingly when experimenting with hyperparameter tuning one has the first
   layer extracting 46 features and the last layer extracting 210 features.
5) Properly configuring ray tune to tune hyperparameters is an experiment in itself. If one is not careful the number of
   parameters will blow up quickly and training will slow down significantly. Instead of guessing the learning rate and
   batch size, one can use them as hyperparameters.
6) To prevent a parameter explosion, I changed from purely minimizing vloss to acceptable amount of vloss with the
   minimum number of parameters. So I roughly used $\text{efficiency} =\frac{1}{\text{vloss} \times \text{parameters}}$
   as working principle and redid the model tuning tighter hyperparameter bounds multiple times.

Annoyances:

1) Ray tune has a bug in `tune.quniform` where it completely ignores the quantization if the quantization value is 1.
   Wasted too much valuable man-hours hunting down why some of the training is erroring out.
2) The model must be compiled before training or the training will be very slow. However, compiled pytorch model not
   saveable (or picklable) ! Nowhere in the `torch.compile` documentation tells us that. Anyway I found a workaround.
3) If ray tune is run on a single computer with multiple child processes launched, then the `torch.compile` might have a
   conflict when generating compiled code. So expect an error the first time.
4) Optuna seems to have issues resuming from an existing tuning experiment which got terminated, preferring to start a
   new study instead.
5) The ray tune directory naming can cause issues in Windows, and it has a tendency to use illegal characters. So a new
   directory naming scheme needed to be created.

This started out as a personal project image generation and has evolved (devolved ?) to something else.
Inspired by image generation systems like Stable Diffusion and Flux, an attempt was made to create a VAE
so that one can generate multiple samples as long as the $\sigma$ and $\mu$

To fully realize the project one needs
This is very hard to explain but let us assume that there are 2 classes $\rightarrow$ [cat, tree]

1) One can use the image of the class 'cat' to generate $\sigma$ and $\mu$ to generate pictures of a cat. However,
   during evaluation one needs a classifier to test if the generated sample is that of a cat. This also means that a
   classifier needs to be designed first.
2) Down the line I realized that using images are not useful to generate $\sigma$ and $\mu$ as then it won't be able to
   generate picture of cat and tree at the same time. A better option use to use
    1) a one hot encoded vector to represent the classes. Then use linear combination of the vectors to
       generate $\sigma$ and $\mu$.
    2) a text encoder to generate the $\sigma$ and $\mu$.
3) Looking through existing literature it is obvious that the generated images from VAEs are blurry which somehow
   escaped my attention during my first reading. This can confuse the classifier. There are VQ-VAEs and Normalizing
   Flows (NF) but the design of NF is very different from VAEs in general.
4) Maybe using $N(0,1)$ as a prior is not a good idea, perhaps using a power law distribution is better. Not sure how to
   do KL-Divergence of power law though and then minimizing it.

If I am using normalized flows maybe we can have some model like the one below where Jacobian is generated using the
text encoder and instead of operating directly on the image, operates on smaller dimensional $z$ \
$$ x \rightarrow ImageEncoder \rightarrow z$$ \
$$ t \rightarrow TextEncoder \rightarrow J$$ \
$$ z \rightarrow J \rightarrow z_g$$ \
$$ z_g \rightarrow ImageDecoder \rightarrow x_g$$ \

Anyway I think I need a classifier model and an image encoder and decoder model. 



