import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time


def positional_encoding(x, num_frequencies=6, incl_input=True):
    """
    Apply positional encoding to the input.
    
    Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. 
    """

    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    # encode input tensor and append the encoded tensor to the list of results.
    [N, D] = x.shape

    L_2 = 2 * num_frequencies

    encoding = np.zeros([N, L_2, D])

    n, d = np.meshgrid(np.arange(N), np.arange(D))

    for L in range(num_frequencies):
        encoding[n, 2 * L, d] = torch.sin(2 ** L * np.pi * x[n, d])
        encoding[n, 2 * L + 1, d] = torch.cos(2 ** L * np.pi * x[n, d])

    encoding = torch.from_numpy(encoding)

    torch_encode = torch.reshape(encoding, (N, D * L_2))

    results.append(torch_encode)
    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)


class model_2d(nn.Module):
    """
    Define a 2D model comprising of three fully connected layers,
    two relu activations and one sigmoid activation.
    """

    def __init__(self, filter_size=128, num_frequencies=6):
        super().__init__()
        #############################  TODO 1(b) BEGIN  ############################
        input_dimension = 2
        encoded_dimension = 2 * input_dimension * num_frequencies

        self.input = nn.Linear(input_dimension + encoded_dimension, filter_size)
        self.hidden = nn.Linear(filter_size, filter_size)
        self.output = nn.Linear(filter_size, 3)
        #############################  TODO 1(b) END  ##############################

    def forward(self, x):
        #############################  TODO 1(b) BEGIN  ############################
        layer0 = F.relu(self.input(x))
        layer1 = F.relu(self.hidden(layer0))
        x = torch.sigmoid(self.output(layer1))
        #############################  TODO 1(b) END  ##############################
        return x


def train_2d_model(test_img, num_frequencies, device, model=model_2d, positional_encoding=positional_encoding,
                   show=True):
    # Optimizer parameters
    lr = 5e-4
    iterations = 10000
    height, width = test_img.shape[:2]

    # Number of iters after which stats are displayed
    display = 2000

    # Define the model and initialize its weights.
    model2d = model(num_frequencies=num_frequencies)
    model2d.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model2d.apply(weights_init)

    #############################  TODO 1(c) BEGIN  ############################
    # Define the optimizer
    optimizer = torch.optim.Adam(model2d.parameters(), lr=lr)
    #############################  TODO 1(c) END  ############################

    # Seed RNG, for repeatability
    seed = 5670
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    t = time.time()
    t0 = time.time()

    #############################  TODO 1(c) BEGIN  ############################
    # Create the 2D normalized coordinates, and apply positional encoding to them
    raw_coords = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    raw_coords = torch.stack([raw_coords[0], raw_coords[1]], dim=-1)

    raw_coords = raw_coords.reshape(height * width, 2)

    position_encode = positional_encoding(raw_coords, num_frequencies).float()
    #############################  TODO 1(c) END  ############################

    for i in range(iterations + 1):
        optimizer.zero_grad()
        #############################  TODO 1(c) BEGIN  ############################
        # Run one iteration
        pred = model2d(position_encode).reshape(height, width, 3)
        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = F.mse_loss(pred, test_img)
        loss.backward()
        optimizer.step()
        #############################  TODO 1(c) END  ############################

        # Display images/plots/stats
        if i % display == 0 and show:
            #############################  TODO 1(c) BEGIN  ############################
            # Calculate psnr
            psnr = 10 * torch.log10(1 ** 2 / loss)
            #############################  TODO 1(c) END  ############################

            print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
                  "Time: %.2f secs per iter" % ((time.time() - t) / display), "%.2f secs in total" % (time.time() - t0))
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(13, 4))
            plt.subplot(131)
            plt.imshow(pred.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(132)
            plt.imshow(test_img.cpu().numpy())
            plt.title("Target image")
            plt.subplot(133)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            # if i==iterations:
            #  np.save('result_'+str(num_frequencies)+'.npz',pred.detach().cpu().numpy())

    print('Done!')
    return pred.detach().cpu()
