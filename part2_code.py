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


def get_rays(height, width, intrinsics, Rcw, Tcw):
    
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    Rcw: Rotation matrix of shape (3,3) from camera to world coordinates.
    Tcw: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that desipte that all ray share the same origin, here we ask you to return 
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder
    
    #############################  TODO 2.1 BEGIN  ##########################  
    # h
    # create tensor torch.Tensor
    index_h = torch.arange(height, dtype=torch.float32)
    index_h = index_h

    # horizontal focal length
    centered_index_h = index_h - height / 2
    fx = intrinsics[0][0]
    h = centered_index_h / fx

    # w
    # create tensor
    index_w = torch.arange(width, dtype=torch.float32)
    index_w = index_w

    # vertical focal length
    centered_index_w = index_w - height / 2
    fy = intrinsics[1][1]
    w = centered_index_w / fy

    (h, w) = torch.meshgrid(h, w)

    # find dirs
    ones = torch.ones_like(w)
    stacked_dirs = torch.stack([w, h, ones], -1)
    dirs = stacked_dirs

    # rays
    # Transpose and reshape dirs tensor
    dirs = (dirs.permute(2, 0, 1)).reshape(3, -1)

    # Multiply Rcw matrix by the dirs tensor then sawp and reshape
    ray_directions = ((torch.matmul(Rcw, dirs)).permute(1, 0)).reshape(width, height, 3)

    # find ray origins
    ray_origins = (torch.broadcast_to(Tcw, ray_directions.shape))

    #############################  TODO 2.1 END  ############################
    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.
  
    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    height, width, _ = ray_origins.shape

    # Compute the depth values
    depth_range = far - near
    sample_indices = torch.arange(samples, dtype=torch.float32) - 1
    sample_ratios = sample_indices / samples
    depth_values = depth_range * sample_ratios + near

    # find ray origins, ray directions
    ray_origins = ray_origins.reshape(height, width, 1, 3)
    ray_directions = ray_directions.reshape(height, width, 1, 3)

    # find depth_points, ray points
    depth_points = torch.broadcast_to(depth_values, (height, width, samples))
    ray_points = ray_origins + depth_points.reshape((height, width, samples, 1)) * ray_directions

    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points
    
class nerf_model(nn.Module):
    
    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper. 
    """

    def __init__(self, filter_size=256, num_x_frequencies=6, num_d_frequencies=3):
        super().__init__()

        #############################  TODO 2.3 BEGIN  ############################
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        xlen = 3 * num_x_frequencies * 2 + 3

        self.l1 = nn.Linear(xlen, filter_size)
        self.l2 = nn.Linear(filter_size, filter_size)
        self.l3 = nn.Linear(filter_size, filter_size)
        self.l4 = nn.Linear(filter_size, filter_size)
        self.l5 = nn.Linear(filter_size, filter_size)

        dlen = 3 * num_d_frequencies * 2 + 3

        self.l6 = nn.Linear(filter_size + xlen, filter_size)
        self.l7 = nn.Linear(filter_size, filter_size)
        self.l8 = nn.Linear(filter_size, filter_size)
        self.l9 = nn.Linear(filter_size, 1)
        self.l10 = nn.Linear(filter_size, filter_size)
        self.l11 = nn.Linear(filter_size + dlen, 128)
        self.l12 = nn.Linear(128, 3)
    #############################  TODO 2.3 END  ############################


    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        h1 = self.relu(self.l1(x))
        h2 = self.relu(self.l2(h1))
        h3 = self.relu(self.l3(h2))
        h4 = self.relu(self.l4(h3))
        h5 = self.relu(self.l5(h4))
        h6 = self.relu(self.l6(torch.cat((h5, x), dim=1)))
        h7 = self.relu(self.l7(h6))
        h8 = self.relu(self.l8(h7))
        sigma = self.l9(h8)
        h10 = self.l10(h8)
        h11 = self.relu(self.l11(torch.cat((h10, d), dim=1)))
        rgb = self.sigmoid(self.l12(h11))
        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):
    
    def get_chunks(inputs, chunksize = 2**15):
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]
    
    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before 
    dividing them into chunks, as well as normalizing and populating the directions.
    """
    #############################  TODO 2.3 BEGIN  ############################
    H, W, S, C = ray_points.shape

    rp = torch.reshape(torch.tensor(ray_points), (H * W * S, C))

    # normalize and divide
    ray_dir = F.normalize(ray_directions, dim=-1)

    # reshape
    ray_dir = torch.reshape(ray_dir.unsqueeze(2).expand(-1, -1, S, -1), (H * W * S, C))

    # positional encoding
    en_ray_points = positional_encoding(rp, num_x_frequencies)
    en_ray_directions = positional_encoding(ray_dir, num_d_frequencies)

    # get chunks
    ray_points_batches = get_chunks(en_ray_points)
    ray_directions_batches = get_chunks(en_ray_directions)
    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches

def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    sigma: Volume density at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
  
    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """
    
    #############################  TODO 2.4 BEGIN  ############################
    relu = torch.nn.ReLU()
    H, W, _, _ = rgb.shape
    rec_image = torch.zeros([H, W, 3])
    dp = torch.tensor(depth_points, dtype=torch.float32)
    delta = torch.cat(((dp[:, :, 1:] - dp[:, :, :-1]), torch.tensor([10e9]).expand([H, W, 1])), dim=-1)

    alpha = 1 - torch.exp(-relu(s) * delta)
    alpha_ = torch.roll(torch.cumprod(1 - alpha, dim=2), 1, dims=2)

    alpha = alpha.unsqueeze(-1)
    alpha_ = alpha_.unsqueeze(-1)

    rec_image = alpha_ * alpha * rgb
    rec_image = rec_image.sum(2)
    #############################  TODO 2.4 END  ############################

    return rec_image

def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):
    
    #############################  TODO 2.5 BEGIN  ############################

    # compute all the rays from the image
    ray_o, ray_dir = get_rays(height, width, intrinsics, pose[:3, :3], pose[:3, 3])
    # sample the points from the rays
    ray_p, depth_p = stratified_sampling(ray_o, ray_dir, near, far, samples)

    # divide data into batches to avoid memory errors
    ray_batches, ray_dir_batches = get_batches(ray_p, ray_dir, num_x_frequencies, num_d_frequencies)
    ray_batches = [batch.to(torch.float32) for batch in ray_batches]
    ray_dir_batches = [dir_batch.to(torch.float32) for dir_batch in ray_dir_batches]

    # forward pass the batches and concatenate the outputs at the end
    # 1. Create empty lists to hold the outputs
    rgb_batch = []
    s_batch = []

    # 2. Forward pass the batches and append the outputs to the respective lists
    for ray_points, ray_directions in zip(ray_batches, ray_dir_batches):
        rgb, sigma = model(ray_points, ray_directions)
        rgb_batch.append(rgb)
        s_batch.append(sigma)

    # 3. Concatenate the outputs
    rgb = torch.cat(rgb_batch, dim=0)
    s = torch.cat(s_batch, dim=0)

    rgb = rgb.reshape(height, width, samples, 3)
    s = s.reshape(height, width, samples)
    # Apply volumetric rendering to obtain the reconstructed image
    rec_image = volumetric_rendering(rgb, s, depth_p)

    #############################  TODO 2.5 END  ############################

    return rec_image