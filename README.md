# Multilayer-Perceptron-Networks
# Machine Perception Project: Fitting 2D Images and 3D Scenes with MLP Networks (Python)

## Project Overview

This project delves into machine perception, specifically targeting the task of fitting both 2D images and 3D scenes utilizing Multilayer Perceptron (MLP) networks. It comprises two distinct parts, each with its set of objectives and methodologies.

## Part 1: Fitting 2D Images

### Positional Encoding and Neural Network Design

- Implemented Positional Encoding, a technique to map continuous input coordinates into a higher-dimensional space, enhancing the neural network's ability to capture intricate color and texture variations effectively.
- Designed a Multilayer Perceptron (MLP) with three linear layers, incorporating ReLu activation for the initial two layers and a Sigmoid activation function for the final layer.

### MLP Training and Image Reconstruction

- Trained the MLP to match the provided 2D image, employing the Adam optimizer and Mean Square Error as the loss function.
- Utilized normalized pixel coordinates and transformed the network's output back into image format.
- Evaluated the MLP's performance by computing the Peak Signal-to-Noise Ratio (PSNR) between the original and reconstructed images.

## Part 2: Fitting 3D Scenes

### Ray Calculation and Sampling

- Calculated the rays of the images based on the transformation between camera and world coordinates, combined with the camera's intrinsic parameters.
- Sampled points along each ray, adopting a uniform distribution from the nearest to the farthest points.

### Neural Radiance Fields (NeRF) and Volumetric Rendering

- Developed a Neural Radiance Fields (NeRF) MLP that ingested input from the position and direction of sampled points along each ray, with positional encoding applied to both.
- Implemented a volumetric rendering formula to compute the color of each pixel, involving the numerical approximation of a continuous integral for ray color.
- Rendered an image by computing all the rays, sampling points along these rays, forwarding them through the neural network, and then applying the volumetric equation to generate the reconstructed image.

### Model Training and Iterative Improvement

- Integrated all the aforementioned steps to train the NeRF model using the Adam optimizer and Mean Square Error as the loss function.
- Iteratively improved the model, experimenting with different positional encoding frequencies and evaluating their impact on the image-fitting process.

## Key Outcomes

This project served as a profound exploration of machine learning applications in computer vision, providing a comprehensive understanding of advanced concepts like NeRF and volumetric rendering. The final model achieved a remarkable PSNR exceeding 24.2 after 3000 iterations, demonstrating effective learning and the successful approximation of a 3D scene from 2D perspectives.
<img width="1026" alt="Screen Shot" src="https://github.com/norahty/Multilayer-Perceptron-Networks/assets/94091909/9d95bdf6-4b6d-4d15-830f-9eaca812d3f5">

