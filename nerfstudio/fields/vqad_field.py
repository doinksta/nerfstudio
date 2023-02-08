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
Field for compound vqad model, adds scene contraction and image embeddings to instant-ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import *
from nerfstudio.field_components.codebook_encoding import *
from nerfstudio.field_components.codebook_cpu import CodeBookEncodingCPU
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
<<<<<<< HEAD
    RGBMAXFieldHead,
=======
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
    CodeBookIndexFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
<<<<<<< HEAD
    CoefficientFieldHead,
=======
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass



def get_normalized_directions(directions: TensorType["bs":..., 3]):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

#now VQAD is the pytorch-nerfstudio code
class VQADField(Field):
    """
    PyTorch implementation of the compound field.
    """

    def __init__(
        self,
        aabb,
        num_images: int,
        position_encoding:  Encoding = CodeBookEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        base_mlp_num_layers: int = 3,
        base_mlp_layer_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        appearance_embedding_dim: int = 40,
        skip_connections: Tuple = (4,),
<<<<<<< HEAD
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),CodeBookIndexFieldHead(),CoefficientFieldHead(),RGBMAXFieldHead()),
=======
        field_heads: Tuple[FieldHead] = (RGBFieldHead(),CodeBookIndexFieldHead(),),
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
        spatial_distortion: SpatialDistortion = SceneContraction(),
    ) -> None:
        super().__init__()
        
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)

        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim() + self.appearance_embedding_dim,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore
    
    '''         
    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
        else:
            positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        #breakpoint()
        return density, base_mlp_out
    ''' 
    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0 
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        # Positions are normalized to be in the range [0, 1]
<<<<<<< HEAD
        encoded_xyz,encoded_xyz2,self.codebook_index,self.coef = self.position_encoding(positions)       
        base_mlp_out = self.mlp_base(encoded_xyz)
        base_mlp_out2 = self.mlp_base(encoded_xyz2)
        
        density = self.field_output_density(base_mlp_out)
        # breakpoint()
        #base_mlp_out=density_embedding
        return density, base_mlp_out,base_mlp_out2
    #'''
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None ,density_embedding2: Optional[TensorType] = None
=======
        encoded_xyz, self.codebook_index = self.position_encoding(positions)       
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        
        return density, base_mlp_out
    #'''
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
    ) -> Dict[FieldHeadNames, TensorType]:

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            embedded_appearance = torch.zeros(
                (*outputs_shape, self.appearance_embedding_dim),
                device=ray_samples.frustums.directions.device,
            )

        outputs = {}
        #breakpoint()
        field_head =self.field_heads[0]
        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
<<<<<<< HEAD
       
=======
        #breakpoint()
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
        mlp_out = self.mlp_head(
            torch.cat(
                [
                    encoded_dir,
<<<<<<< HEAD
                    density_embedding,  
=======
                    density_embedding,  # type:ignore
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
                    embedded_appearance,
                ],
                dim=-1,  # type:ignore
            )
<<<<<<< HEAD
        ) 
        # breakpoint()
        outputs[field_head.field_head_name] = field_head(mlp_out)#[4096,48,3]

        ''''''
        field_head =self.field_heads[3]
        mlp_out = self.mlp_head(
            torch.cat(
                [
                    encoded_dir,
                    density_embedding2,  
                    embedded_appearance,
                ],
                dim=-1,  # type:ignore
            )
        ) 
        # breakpoint()
        outputs[field_head.field_head_name] = field_head(mlp_out)#[4096,48,3]

        field_head =self.field_heads[1] 
        outputs[field_head.field_head_name]= self.codebook_index#[4,4096,48,1]
        
        field_head =self.field_heads[2] 
        outputs[field_head.field_head_name]= self.coef # we use codebook_entry 5 here[5,4096,48,1]
        
        
=======
        )
        outputs[field_head.field_head_name] = field_head(mlp_out)#[4096,48,3]
        field_head =self.field_heads[1] 
        outputs[field_head.field_head_name]= self.codebook_index#[4,4096,48,1]
>>>>>>> fe0c2d56495d345d46cb88d188e5d6c456f88853
        # breakpoint()
        return outputs

