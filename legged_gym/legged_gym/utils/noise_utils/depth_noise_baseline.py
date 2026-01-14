# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthNoiseBaseline(nn.Module):
    def __init__(
        self,
        focal_length,
        baseline,
        gaussian_noise_std=0.1,
        missing_data_prob=0.001,
        salt_pepper_noise_prob=0.02,
        gaussian_shift_std=0.5,
        gaussian_blur_kernel_size=3,
        gaussian_blur_sigma=0.5,
        min_depth=0.0,
        max_depth=10.0,
    ):
        super(DepthNoiseBaseline, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.missing_data_prob = missing_data_prob
        self.salt_pepper_noise_prob = salt_pepper_noise_prob
        self.gaussian_shift_std = gaussian_shift_std
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.depth_min = min_depth
        self.depth_max = max_depth
        self.focal_length = focal_length
        self.baseline = baseline

    def forward(self, depth):
        # correct input shape
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1) # add channel dimension
        
        # Clamp to valid depth range
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        
        # Add Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(depth) * self.gaussian_noise_std
            depth = depth + noise

        # Apply contiguous missing data mask
        if self.missing_data_prob > 0:
            depth = self.apply_missing_data(depth, self.missing_data_prob)

        # Apply salt-and-pepper noise
        if self.salt_pepper_noise_prob > 0:
            rand_vals = torch.rand_like(depth)
            salt_mask = rand_vals < (self.salt_pepper_noise_prob / 2)
            pepper_mask = (rand_vals >= (self.salt_pepper_noise_prob / 2)) & (rand_vals < self.salt_pepper_noise_prob)
            depth = depth.masked_fill(salt_mask, self.depth_max)
            depth = depth.masked_fill(pepper_mask, self.depth_min)

        # Apply Gaussian shift (spatial shift)
        if self.gaussian_shift_std > 0:
            shift_x = torch.randn(depth.size(0)) * self.gaussian_shift_std
            shift_y = torch.randn(depth.size(0)) * self.gaussian_shift_std
            grid = self._get_gaussian_shift_grid(depth, shift_x, shift_y)
            depth = F.grid_sample(depth, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # Optionally, apply Gaussian blur to simulate optical smoothing
        if self.gaussian_blur_kernel_size > 0:
            depth = self.gaussian_blur(depth, self.gaussian_blur_kernel_size, self.gaussian_blur_sigma)

        # Final clamping to valid range
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        return depth

    def _get_gaussian_shift_grid(self, depth, shift_x: torch.Tensor, shift_y: torch.Tensor):
        B, C, H, W = depth.size()
        device = depth.device
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = -2 * shift_x / (W - 1)
        theta[:, 1, 2] = -2 * shift_y / (H - 1)
        grid = F.affine_grid(theta, size=depth.size(), align_corners=False)
        return grid

    def gaussian_blur(self, depth, kernel_size: int, sigma: float):
        # Create 1D Gaussian kernel
        device = depth.device
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        # Outer product to get 2D kernel
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d.expand(depth.size(1), 1, kernel_size, kernel_size)
        # Convolve using groups to apply the same kernel to each channel
        depth = F.conv2d(depth, gauss_2d, padding=kernel_size//2, groups=depth.size(1))
        return depth

    def apply_missing_data(self, depth, missing_prob: float):
        B, C, H, W = depth.size()
        # Create a full-resolution binary mask where each pixel has a chance to be missing.
        mask = (torch.rand(B, 1, H, W, device=depth.device) < missing_prob).float()
        
        # Use max pooling as a form of dilation to grow the missing regions.
        dilation_kernel_size = (3, 5)
        dilation_padding_size = (dilation_kernel_size[0] // 2, dilation_kernel_size[1] // 2)
        missing_mask = F.max_pool2d(mask, kernel_size=dilation_kernel_size, stride=1, padding=dilation_padding_size)
        missing_mask = missing_mask > 0.5  # Convert to a boolean mask.
        
        # Apply the missing data mask.
        depth = depth.masked_fill(missing_mask, 0.0)
        return depth

# Example usage
if __name__ == "__main__":
    import cv2
    import yaml
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from dataloader import TartanAirDataLoader
    
    from network.image_utils import RandomCropDownsample
    
    def normalize_depth(depth, min_depth, max_depth, is_log):
        depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
        depth = torch.clamp(depth, min_depth, max_depth) # Clamp the depth values
        depth = torch.log(depth + 1.0) if is_log else depth
        return depth
    
    
    config_path = './config/pretrain.yaml'
    
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    config = load_config(config_path)
    dataloader_config = config.get('dataloader', {})
    training_config = config.get('training', {})
    
    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    focal_length = dataloader_config.get('focal_length', 50.0)
    baseline = dataloader_config.get('baseline', 0.12)
    
    dataloader = TartanAirDataLoader(dataloader_config)
    batch = dataloader.load_batch()
    dataloader.stop()
    
    depth = batch['depth_lcam_front']
    # reshape the depth tensor to (B, 1, H, W)
    depth = depth.unsqueeze(2)
    batch_size, seq_len = depth.shape[0], depth.shape[1]
    depth = depth.view(batch_size * seq_len, 1, depth.shape[3], depth.shape[4])
    
    normalize_depth_fn = lambda x: normalize_depth(x, min_depth, max_depth, is_log=training_config['log_space'])
    
    crop_min = (dataloader_config['crop_min'], dataloader_config['crop_min'])
    crop_max = (dataloader_config['new_image_shape_hw'][0], dataloader_config['new_image_shape_hw'][1])
    final_size = (dataloader_config['resize_shape_hw'][0], dataloader_config['resize_shape_hw'][1])
    
    random_cropper = RandomCropDownsample(crop_min, crop_max, final_size)
    
    depth_noise = DepthNoiseBaseline(focal_length=focal_length,
                                     baseline=baseline, 
                                     min_depth=min_depth,
                                     max_depth=max_depth)
    # depth_noise = torch.jit.script(depth_noise)
    
    # Normalize the depth values
    depth = normalize_depth_fn(depth)
    
    # Random crop and downsample
    depth = random_cropper(depth)
    
    # Add noise to the depth images
    noisy_depth = depth_noise(depth)
    
    for i in range(batch_size * seq_len):
        clip_depth = depth[i].clip(0.1, 10.0)
        disp_depth = clip_depth.squeeze().detach().cpu().numpy()
        disp_nosie_depth = noisy_depth[i].squeeze().detach().cpu().numpy()
        # enlarge the resolution for better visualization
        disp_depth = cv2.resize(disp_depth, (256, 160))
        disp_nosie_depth = cv2.resize(disp_nosie_depth, (256, 160))
        # normalize the depth values to 0-1
        disp_depth = (disp_depth - disp_depth.min()) / (disp_depth.max() - disp_depth.min())
        disp_nosie_depth = (disp_nosie_depth - disp_nosie_depth.min()) / (disp_nosie_depth.max() - disp_nosie_depth.min())
        
        # Merge the images horizontally
        merged_image = cv2.hconcat([disp_depth, disp_nosie_depth])
        
        # Display the merged image
        cv2.imshow("Depth and Noisy Depth Image", merged_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break
        cv2.destroyAllWindows()
    
    print("Noisy depth shape:", noisy_depth.shape)
