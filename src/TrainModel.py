from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.Model import Model
from src.codec import Encoder, Decoder


def show_image(img_data):
    transposed = torch.permute(img_data, (1, 2, 0)).numpy()
    plt.imshow(transposed)
    plt.show()


def train_one_epoch(encoder, decoder, optimizer, training_loader, epoch, tb_writer):
    loss_fn = nn.MSELoss()
    running_loss_rms = 0.
    running_loss_kl = 0.

    last_loss_kl = 0.
    last_loss_mse = 0.
    for i, data in enumerate(training_loader):
        images, labels = data
        images = images.cuda()
        optimizer.zero_grad()
        mu, log_var = encoder.forward(images)
        latent = Model.reparameterization(mu, log_var)
        decoded_output = decoder.forward(latent)

        kl_loss_sum = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
        kl_loss = kl_loss_sum.mean()
        mse_loss = loss_fn(decoded_output, images)
        vae_loss = kl_loss + mse_loss
        vae_loss.backward()
        optimizer.step()

        running_loss_kl += kl_loss.item()
        running_loss_rms += mse_loss.item()
        if i % 1000 == 999:
            last_loss_kl = running_loss_kl / 1000
            last_loss_mse = running_loss_rms / 1000
            last_loss = last_loss_kl + last_loss_mse
            # loss per batch
            print('  batch {} losskl: {} lossmse'.format(i + 1, last_loss_kl, last_loss_mse))
            tb_x = epoch * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss_kl = 0.
            running_loss_rms = 0.
    return last_loss_kl, last_loss_mse


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    training_set = CIFAR10(root='../data/CIFAR/train', train=True, download=True, transform=transform)
    validation_set = CIFAR10(root='../data/CIFAR/test', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    encoder = Encoder(32, 4, [3, 5, 5], [3, 8, 14, 24]).cuda()
    decoder = Decoder(4, 32, [5, 5], [12, 8, 3]).cuda()
    enc_param = encoder.parameters()
    dec_param = decoder.parameters()

    # encoder_params = sum(p.numel() for p in enc_param if p.requires_grad)
    # decoder_params = sum(p.numel() for p in dec_param if p.requires_grad)

    enc_p = set(p for p in enc_param)
    dec_p = set(p for p in dec_param)
    params = enc_p.union(dec_p)
    # params.update(enc_param)
    # params.update(dec_param)

    # total_params = encoder_params + decoder_params

    print(f'Encoder parameters: {encoder}')
    print(f'Decoder parameters: {decoder}')
    print(
        f'Encoder parameters : {len(enc_p)} , Decoder parameters {len(dec_p)}, Total number of parameters: {len(params)}')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params, lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/vae_traimer_{}'.format(timestamp))
    EPOCHS = 5

    # time.sleep(2)

    best_vloss = 1_000_000
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        encoder.train(True)
        decoder.train(True)
        last_loss_kl, last_loss_mse = train_one_epoch(encoder, decoder, optimizer, train_loader, epoch, writer)
        avg_loss = last_loss_kl + last_loss_mse

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        encoder.eval()
        decoder.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, _ = vdata
                vinputs = vinputs.cuda()
                mean, log_var = encoder.forward(vinputs)
                latent = Model.reparameterization(mean, log_var)
                outputs = decoder.forward(latent)

                vloss = loss_fn(outputs, vinputs)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            encoder_path = 'encoder_{}_{}'.format(timestamp, epoch)
            decoder_path = 'decoder_{}_{}'.format(timestamp, epoch)
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
