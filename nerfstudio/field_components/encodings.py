# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Encoding functions
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.math import components_from_spherical_harmonics, expected_sin
from nerfstudio.utils.printing import print_tcnn_speed_warning

try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offet to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self, in_dim: int, num_frequencies: int, min_freq_exp: float, max_freq_exp: float, include_input: bool = False
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: TensorType["bs":..., "input_dim"],
        covs: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)
        scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoding frequencies
        scale: Std of Gaussian to sample frequencies. Must be greater than zero
        include_input: Append the input coordinate to the encoding
    """

    def __init__(self, in_dim: int, num_frequencies: int, scale: float, include_input: bool = False) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        self.register_buffer(name="b_matrix", tensor=b_matrix)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def forward(
        self,
        in_tensor: TensorType["bs":..., "input_dim"],
        covs: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates RFF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = in_tensor @ self.b_matrix  # [..., "num_frequencies"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:

        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        self.tcnn_encoding = None
        if not TCNN_EXISTS and implementation == "tcnn":
            print_tcnn_speed_warning("HashEncoding")
        elif implementation == "tcnn":
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": min_res,
                    "per_level_scale": growth_factor,
                },
            )

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: TensorType["bs":..., "num_levels", 3]) -> TensorType["bs":..., "num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class TensorCPEncoding(Encoding):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: TensorType[3, "num_components", "resolution", "resolution"]
    line_coef: TensorType[3, "num_components", "resolution", 1]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        line_coef = F.interpolate(self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True)

        # TODO(ethan): are these torch.nn.Parameters needed?
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.resolution = resolution


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical hamonic levels to encode.
    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only suports 1 to 4 levels, requested {levels}")

        self.levels = levels

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)


class HashtorchEncoding(Encoding):
    """VQAD encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.(codebook width?)
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,  #determine the level
        max_res: int = 1024, #max grid res
        log2_hashmap_size: int = 19, #codebookwidth
        features_per_level: int = 2, #feature dim
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "torch",#tcnn        
    ) -> None:

        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)
        
        #build hash table
        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)
        
        
        # build codebook
        #self.codebook=torch.rand
    

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: TensorType["bs":..., "num_levels", 3]) -> TensorType["bs":..., "num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0
        #now we can the input to be 
        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        #reakpoint()
        in_tensor = in_tensor[..., None, :]  # [...,48,3]->[..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3] L:level of the hash table
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        #get the idx of the corner 
        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        
        #use the index to get the corresponding feature
        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        #interpolation here
        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])
  
        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]
        #breakpoint()
        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        fw= self.pytorch_fwd(in_tensor)
        breakpoint()
        #res2=self.grid.interpolate(in_tensor, lod_idx)
        return fw
    

class CodeBookEncoding(Encoding):

    # volume_coef: TensorType[3, “num_components”, “resolution”, “resolution”,“resolution”]
    def __init__(
        self,
        resolutions: int = [256], #[32,64,128,256],
        init_scale: float = 0.1,
        codebook_width: int = 32,
        n_codebook_entry: int = 8,
        n_codebook_level: int = 1,
        
    ) -> None:
        super().__init__(in_dim=3)
        self.codebook_width = codebook_width
        self.n_codebook_entry = n_codebook_entry
        self.n_codebook_level = n_codebook_level
        self.codebook = torch.randn(size=(self.n_codebook_level,self.n_codebook_entry, self.codebook_width))
        
        self.resolutions = resolutions
        self.volume_raw_prob0 =  torch.randn((1, self.n_codebook_entry, self.resolutions[0], self.resolutions[0], self.resolutions[0]))
        # self.volume_raw_prob1 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[1], self.resolutions[1], self.resolutions[1])))
        # self.volume_raw_prob2 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[2], self.resolutions[2], self.resolutions[2])))
        # self.volume_raw_prob3 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[3], self.resolutions[3], self.resolutions[3])))
        # self.volume_raw_prob4 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[4], self.resolutions[4], self.resolutions[4])))
        # self.volume_raw_prob5 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[5], self.resolutions[5], self.resolutions[5])))
        # self.volume_raw_prob6 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[6], self.resolutions[6], self.resolutions[6])))
        # self.volume_raw_prob7 = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolutions[7], self.resolutions[7], self.resolutions[7])))
        self.volume_raw_prob0 = nn.Parameter(init_scale * self.volume_raw_prob0)
        self.codebook=nn.Parameter(init_scale * self.codebook)
    
    def get_out_dim(self) -> int:
        return self.codebook_width*self.n_codebook_level
    
    def load_checkpoint(self, checkpoint_file):
        self.codebook.data = checkpoint_file['codebook']
        self.volume_raw_prob0.data = checkpoint_file['weights']
    
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        volume_coord = (in_tensor.view(1, -1, 1, 1, 3) * 2 - 1).detach()#([1, 196608, 1, 1, 3])
        #now try to keep the initial codebook fixed
        #self.codebook=self.codebook.detach() #set the codebook to be fixed
        
        #resolution
        # volume_selection0 = torch.softmax(self.volume_raw_prob0, dim=1).to(in_tensor.device)
        # max_index=torch.argmax(volume_selection0,dim=1).view(self.resolutions[0],self.resolutions[0],self.resolutions[0],1)
        codebook=self.codebook.view(self.codebook.shape[1],self.codebook.shape[2])

        #resolution-16
        volume_selection0 = torch.softmax(self.volume_raw_prob0, dim=1).to(in_tensor.device) #[1,4,128,128,128]
        '''
        ##now we try to use gumbel softmax
        temperature=0.5
        gumbel_noise = -torch.empty_like(self.volume_raw_prob0).exponential_().log()
        gumbel_noise = (gumbel_noise + self.volume_raw_prob0) / temperature
        volume_gumbel_selection0 = torch.softmax(gumbel_noise, dim=1).to(in_tensor.device) #[1,4,128,128,128]
        
        # breakpoint()
        volume_selection0 =volume_gumbel_selection0
        '''
        volume_coef0 = F.grid_sample(volume_selection0, volume_coord)
        volume_coef0= torch.moveaxis(volume_coef0.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # codebook_max0,codebook_index0=torch.max(volume_coef0,dim=-1,keepdim=True)
        # breakpoint()
        
        
       
        # origianl_shape=volume_selection0.transpose(1, 4).shape
        
        # # kmeans 
        # from kmeans_pytorch import kmeans  
        # num_clusters=2
        
        # X= volume_coef0.reshape(-1,volume_coef0.shape[-1])
        # cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=num_clusters, distance='euclidean', device=in_tensor.device)
        ''''''

        
        # X = volume_selection0.transpose(1, 4).reshape(-1,self.codebook_width)
        # cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=num_clusters, distance='euclidean', device=in_tensor.device)
       
        # cluster_selection0=cluster_ids_x.unsqueeze(0)#.reshape(origianl_shape).transpose(3,1)
        # cluster_selection0=cluster_centers[ cluster_selection0]
        # cluster_selection0=cluster_selection0.reshape(origianl_shape[0],origianl_shape[1],origianl_shape[2],origianl_shape[3],cluster_selection0.shape[-1]).transpose(4,1).to(in_tensor.device).type(torch.float32)
       
        # cluster_feature=F.grid_sample(cluster_selection0, volume_coord)
        # cluster_feature=torch.moveaxis(cluster_feature.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        ''' ''' 
        
        # breakpoint()
        # cluster_index0= torch.moveaxis(cluster_index0.view(1, *in_tensor.shape[:-1]), 0, -1).type(torch.long)
        
        
        
         # breakpoint()
        codebook=self.codebook.view(self.codebook.shape[1],self.codebook.shape[2])
        max_index=torch.argmax(volume_coef0,dim=2)
        max_features0=codebook[max_index]
        max_codebook_index0=max_index.unsqueeze(-1)
        # breakpoint()
        features0=torch.matmul(volume_coef0, self.codebook[0].to(in_tensor.device))
        codebook_max0,codebook_index0=torch.max(volume_coef0,dim=-1,keepdim=True)
        
        
        # #resolution-32
        # volume_selection1 = torch.softmax(self.volume_raw_prob1, dim=1).to(in_tensor.device)
        # volume_coef1 = F.grid_sample(volume_selection1, volume_coord)
        # volume_coef1= torch.moveaxis(volume_coef1.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features1=torch.matmul(volume_coef1, self.codebook[1].to(in_tensor.device))
        
        # codebook_max1,codebook_index1=torch.max(volume_coef1,dim=-1,keepdim=True)

        # #resolution-128
        # volume_selection2 = torch.softmax(self.volume_raw_prob2, dim=1).to(in_tensor.device)
        # volume_coef2 = F.grid_sample(volume_selection2, volume_coord)
        # volume_coef2= torch.moveaxis(volume_coef2.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        
        # codebook_max2,codebook_index2=torch.max(volume_coef2,dim=-1,keepdim=True)
        # #codebook_index2=torch.cat((codebook_index,codebook_index,codebook_index),dim=-1)
        
        # features2=torch.matmul(volume_coef2, self.codebook[2].to(in_tensor.device))
        
        
        # #resoluiton-256
        # volume_selection3 = torch.softmax(self.volume_raw_prob3, dim=1).to(in_tensor.device)
        # volume_coef3 = F.grid_sample(volume_selection3, volume_coord)
        # volume_coef3 = torch.moveaxis(volume_coef3.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features3=torch.matmul(volume_coef3, self.codebook[3].to(in_tensor.device))
        
        # codebook_max3,codebook_index3=torch.max(volume_coef3,dim=-1,keepdim=True)
        
        # #resolution-256
        # volume_selection4 = torch.softmax(self.volume_raw_prob4, dim=1).to(in_tensor.device)
        # volume_coef4 = F.grid_sample(volume_selection4, volume_coord)
        # volume_coef4= torch.moveaxis(volume_coef4.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features4=torch.matmul(volume_coef4, self.codebook[4].to(in_tensor.device))
        
        # #resolution-512
        # volume_selection5 = torch.softmax(self.volume_raw_prob5, dim=1).to(in_tensor.device)
        # volume_coef5 = F.grid_sample(volume_selection5, volume_coord)
        # volume_coef5= torch.moveaxis(volume_coef5.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features5=torch.matmul(volume_coef3, self.codebook[5].to(in_tensor.device))

        # #resolution-768
        # volume_selection6 = torch.softmax(self.volume_raw_prob6, dim=1).to(in_tensor.device)
        # volume_coef6 = F.grid_sample(volume_selection6, volume_coord)
        # volume_coef6= torch.moveaxis(volume_coef6.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features6=torch.matmul(volume_coef6, self.codebook[6].to(in_tensor.device))
         
        # #resolution-1024
        # volume_selection7 = torch.softmax(self.volume_raw_prob7, dim=1).to(in_tensor.device)
        # volume_coef7 = F.grid_sample(volume_selection7, volume_coord)
        # volume_coef7= torch.moveaxis(volume_coef7.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        # features7=torch.matmul(volume_coef1, self.codebook[7].to(in_tensor.device))
        
        
        
        
        #features=torch.cat((features0, features1,features2,features3), dim=-1)
        # features=features0
        # codebook_index=torch.cat((codebook_index0[None, ...],codebook_index1[None, ...],codebook_index2[None, ...],codebook_index3[None, ...]),dim=0)
        codebook_index=codebook_index0.unsqueeze(0)
        # cluster_index0=cluster_index0.unsqueeze(0)
        volume_coef =volume_coef0.unsqueeze(0)
        volume_coef = volume_coef.contiguous()
        volume_coef=volume_coef.view(self.n_codebook_entry,*in_tensor.shape[:-1],1)
        
        # max_features=max_features0
        # breakpoint()
        # breakpoint()
        # features=features0
        # features_max=codebook[codebook_index0].view(features.shape)
        features=features0
        # breakpoint()
        features_max=max_features0#cluster_centers[cluster_ids_x].view(features.shape).to(in_tensor.device)
        # features_max=cluster_feature
        index =codebook_index#cluster_ids_x.unsqueeze(0).reshape(1,features.shape[0],features.shape[1],1).to(in_tensor.device).type(torch.long)
        # breakpoint()
        return features,features_max,index,volume_coef#features=[4096,48,16],codebook_index=[4,4096,48,1],volume_coef=[4096,48,5]
    
    
    
