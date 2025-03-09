from torchvision.datasets import CIFAR10

from src.Model import Model

if __name__ == '__main__':

    model = Model(64,16,16,8,256)

    training = CIFAR10.train_list

    print(model)



