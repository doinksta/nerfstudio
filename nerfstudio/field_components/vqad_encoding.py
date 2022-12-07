from nerfstudio.field_components.encodings import Encoding
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

class VQADEncoding(Encoding):
    def __init__(self,codebookwidth,resolution):
        self.codebookwidth=codebookwidth
        self.codebookentry=2**codebookwidth
        self.codebook=torc
        print(codebookwidth)

asdf = CodeBookEncoding(33)