from nerfstudio.field_components.encodings import Encoding
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType



class CodeBookEncodingCPU(Encoding):  # we are trying a CPU-GPU version now

    # volume_coef: TensorType[3, “num_components”, “resolution”, “resolution”,“resolution”]
    def __init__(
        self,
        num_levels: int = 4,
        resolutions: int = [16, 32, 128, 256],
        init_scale: float = 0.1,
        features_per_entry: int = 4,  # codebook width
        entries_per_level: int = 8
    ) -> None:
        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_entry = features_per_entry
        self.entries_per_level = entries_per_level
        
        # Store resolutions as a torch.IntTensor and validate its length
        self.resolutions = resolutions
        
        if not isinstance(resolutions, torch.Tensor):
            self.resolutions = torch.tensor(self.resolutions)

        self.resolutions = self.resolutions.flatten()
        assert len(self.resolutions) == self.num_levels
        
        # For a level of resolution n, store n + 1 grid points along each dimension (n cubes)
        self.n_total_grid_points = ((self.resolutions + 1) ** 3).sum()
        
        self.volume_raw_prob = nn.Parameter(init_scale * torch.randn((self.n_total_grid_points, self.entries_per_level)).to('cpu'))
        self.codebook = nn.Parameter(init_scale * torch.randn(size=(self.num_levels, self.entries_per_level, self.features_per_entry)).to('cpu'))
        
        # Old (for reference)
        # self.volume_raw_prob = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolution, self.resolution, self.resolution))).to('cpu')
        # self.codebook=nn.Parameter(init_scale * torch.randn(size=(self.n_codebook_level,self.n_codebook_entry, self.codebook_width))).to('cpu')
    
    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_entry
    
    def volume_lookup(self, scaled_in_tensor):
        # Takes in scaled in tensor of shape [..., L, 3], which contains Int values
        # Returns the relevant raw probabilities, on the same device as scaled_in_tensor
        assert (scaled_in_tensor.shape[-2], scaled_in_tensor.shape[-1]) == (self.num_levels, 3)
        
        per_level_offsets = torch.cat((torch.tensor([0]), torch.cumsum((self.resolutions[:-1] + 1) ** 3, dim=0)))
        
        coord_to_idx_multipliers = torch.hstack((
            torch.transpose(self.resolutions[None, ...] + 1, 0, 1) ** 2,
            torch.transpose(self.resolutions[None, ...] + 1, 0, 1) ** 1,
            torch.transpose(self.resolutions[None, ...] + 1, 0, 1) ** 0,
        ))
        
        access_indices = (scaled_in_tensor.to('cpu') * coord_to_idx_multipliers).sum(dim=-1, dtype=torch.int32) + per_level_offsets
        
        ret = self.volume_raw_prob[access_indices].to(scaled_in_tensor.device)
        
        return ret
    
        
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        in_tensor = torch.zeros(in_tensor.shape).to(in_tensor.device)
        scaled = in_tensor * self.resolutions.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled)
        scaled_f = torch.floor(scaled)
        
        offset = scaled - scaled_f
        
        
        # Perform trilinear interpolation over probabilities of each codebook entry being selected
        f_0 = torch.softmax(self.volume_lookup(scaled_c), dim=-1)  # [..., num_levels, entries_per_level]
        f_1 = torch.softmax(self.volume_lookup(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1)), dim=-1)
        f_2 = torch.softmax(self.volume_lookup(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1)), dim=-1)
        f_3 = torch.softmax(self.volume_lookup(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1)), dim=-1)
        f_4 = torch.softmax(self.volume_lookup(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1)), dim=-1)
        f_5 = torch.softmax(self.volume_lookup(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1)), dim=-1)
        f_6 = torch.softmax(self.volume_lookup(scaled_f), dim=-1)
        f_7 = torch.softmax(self.volume_lookup(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1)), dim=-1)
        
        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])
        
        encoded_codebook_weights = f0312 * offset[..., 2:3] + f4756 * (1 - offset[..., 2:3])
              
        features = encoded_codebook_weights[..., None, :] @ self.codebook.to('cuda')
        
        features = features.flatten(start_dim=-3, end_dim=-1)
        
        codebook_index = torch.moveaxis(encoded_codebook_weights.max(dim=-1, keepdim=True)[1], -2, 0)
        
        return features, codebook_index


# asdf = CodeBookEncodingCPU()
# print('running')
# breakpoint()
# in_tensor = torch.rand((4096, 48, 3)).to('cuda')
# asdf(in_tensor)
# breakpoint()
# in_tensor2 = torch.rand((4096, 48, 3)).to('cuda')
# asdf(in_tensor2)

# breakpoint()
# in_tensor3 = torch.rand((4096, 48, 3)).to('cuda')
# asdf(in_tensor3)
# breakpoint()