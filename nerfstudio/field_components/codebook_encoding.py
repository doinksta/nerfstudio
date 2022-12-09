from nerfstudio.field_components.encodings import Encoding
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

# Version 1
# class CodeBookEncoding_level(Encoding):
#     def __init__(self,codebookwidth:int=4,codebookentry:int=100,codebooklevel:int=8,resolution=32)-> None:
#         super().__init__(in_dim=3)
#         #build codebook 
#         self.codebookwidth=codebookwidth
#         self.codebookentry=codebookentry
#         self.codebook=torch.randn(size=(codebookentry,codebookwidth))
        
#         #build input
#         #the input among[0,1] and [:,i,j].sum=1
#         self.resolution=resolution
#         tensor=torch.rand(size=(1,self.codebookentry,self.resolution,self.resolution,self.resolution))
#         sum = tensor.sum(dim=0)
#         probability=  tensor/sum
#         # apply the softmax function along the first dimension
#         softmax = torch.softmax(probability, dim=0)
#         # calculate the maximum value along the first dimension
#         max_values = torch.max(softmax, dim=0)[0]
#         # create a new tensor where the maximum values are 1.0 and the rest are 0.0
#         self.input= torch.where(softmax == max_values, torch.ones_like(softmax), torch.zeros_like(softmax)).detach()
#         self.input=nn.Parameter(self.input)
#         self.codebook=nn.Parameter(self.codebook)
        
        
    
#     def get_out_dim(self) -> int:
#         return self.codebookwidth

#     def forward(self, in_tensor: TensorType["bs": ..., "input_dim"]) -> TensorType["bs": ..., "output_dim"]:
#         #the grid 
#         grid3d=in_tensor.view((1, 1, 1, -1, 3)) .to(in_tensor.device)
#         #output is the coefficient
#         coefficent=F.grid_sample(self.input.to(in_tensor.device), grid3d.to(in_tensor.device), align_corners=True).view(-1,self.codebookentry).to(in_tensor.device)
#         features=torch.matmul(coefficent,self.codebook.to(in_tensor.device)).view(in_tensor.shape[0],in_tensor.shape[1],self.codebookwidth).to(in_tensor.device)
#         #print(output)
#         #o=coefficent.view(in_tensor.shape[0],in_tensor.shape[1],self.codebookentry)
#         return features

    
'''
in_tensor=torch.rand(size=(4096,48,3))
asdf = CodeBookEncoding(codebookwidth=4,codebookentry=16,resolution=64)
print(asdf.forward(in_tensor).shape)
'''




# class CodeBookEncoding(Encoding):
#     def __init__(self,codebookwidth:int=4,codebookentry:int=100,codebooklevel:int=8,resolutions:int =[4,8,16,32,64,128,256,512])-> None:
#         super().__init__(in_dim=3)
#         #build codebook
#         self.codebookwidth=codebookwidth
#         self.codebookentry=codebookentry
#         self.codebooklevel=codebooklevel
#         self.codebook=torch.randn(size=(codebooklevel,codebookentry,codebookwidth))
        
#         #build input
#         #the input among[0,1] and [:,i,j].sum=1
#         self.resoluitons=resolutions
#         self.inputs=[]
#         for resolution in self.resoluitons:
#             tensor=torch.rand(size=(1,self.codebookentry,resolution,resolution,resolution))
#             sum = tensor.sum(dim=0)
#             probability=  tensor/sum
#             # apply the softmax function along the first dimension
#             softmax = torch.softmax(probability, dim=0)
#             # calculate the maximum value along the first dimension
#             max_values = torch.max(softmax, dim=0)[0]
#             # create a new tensor where the maximum values are 1.0 and the rest are 0.0
#             self.inputs.append(torch.where(softmax == max_values, torch.ones_like(softmax), torch.zeros_like(softmax)).detach()) 

#         self.inputs=torch.Tensor(self.inputs)
#         #build the paramenter
#         self.inputs=nn.Parameter(self.inputs)
#         self.codebook=nn.Parameter(self.codebook)
        
        
    
#     def get_out_dim(self) -> int:
#         return self.codebookwidth*self.codebooklevel

#     def forward(self, in_tensor: TensorType["bs": ..., "input_dim"]) -> TensorType["bs": ..., "output_dim"]:
#         features=[]
#         for input in self.inputs:
#             #the grid 
#             grid3d=in_tensor.view((1, 1, 1, -1, 3)) .to(in_tensor.device)
#             #output is the coefficient
#             coefficent=F.grid_sample(input.to(in_tensor.device), grid3d.to(in_tensor.device), align_corners=True).view(-1,self.codebookentry).to(in_tensor.device)
#             features.append(torch.matmul(coefficent,self.codebook.to(in_tensor.device)).view(in_tensor.shape[0],in_tensor.shape[1],self.codebookwidth).to(in_tensor.device))
#             #print(output)
#             #o=coefficent.view(in_tensor.shape[0],in_tensor.shape[1],self.codebookentry)
#         features=torch.Tensor(features)
#         return features


#########



# class CodeBookEncoding(Encoding):
#     """Learned vector-matrix encoding proposed by TensoRF

#     Args:
#         resolution: Resolution of grid.
#         num_components: Number of components per dimension.
#         init_scale: Initialization scale.
#     """

#     # volume_coef: TensorType[3, "num_components", "resolution", "resolution","resolution"]


#     def __init__(
#         self,
#         resolution: int = 128,
#         init_scale: float = 0.1,
#         codebook_width: int = 4,
#         n_codebook_entry: int = 24,

#     ) -> None:
#         super().__init__(in_dim=3)

#         self.codebook_width = codebook_width
#         self.n_codebook_entry = n_codebook_entry
        
#         self.codebook = torch.randn(size=(self.n_codebook_entry, self.codebook_width))

#         self.resolution = resolution
        
#         self.volume_raw_prob = nn.Parameter(init_scale * torch.randn((1, self.n_codebook_entry, self.resolution, self.resolution, self.resolution)))
#         self.codebook=nn.Parameter(init_scale * self.codebook)
        
#     def get_out_dim(self) -> int:
#         return self.codebook_width


#     def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
#         """Compute encoding for each position in in_positions
#         in_tensor: expected range is [0, 1]

#         Args:
#             in_tensor: position inside bounds in range [-1,1],

#         Returns: Encoded position
#         """
        
#         # Convert in_tensor range to -1 to 1 for sampling
#         # volume_coord=torch.stack([in_tensor[..., [0, 1 ,2]]])
#         volume_coord = (in_tensor.view(1, -1, 1, 1, 3) * 2 - 1).detach()
        
#         volume_selection = torch.softmax(self.volume_raw_prob, dim=1).to(in_tensor.device)
    
#         volume_coef = F.grid_sample(volume_selection, volume_coord)

#         volume_coef = torch.moveaxis(volume_coef.view(self.n_codebook_entry, *in_tensor.shape[:-1]), 0, -1)
        
#         features=torch.matmul(volume_coef, self.codebook.to(in_tensor.device))
        
#         return features  # [..., 1 * Components]


    