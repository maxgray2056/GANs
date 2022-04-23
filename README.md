# Deep learning project 3 (GAN)
DCGAN, WGAN, ACGAN, WACGAN

## Table of Contents
- [Projects](#projects)
- [Background](#background)
- [Install](#install)
- [Packages](#packages)
- [Contributing](#contributing)


## Projects
- [DCGAN](DCGAN.py)
- [WGAN](WGAN.py)
- [ACGAN](ACGAN.py)
- [WACGAN](WACGAN.py)


## Background
These projects are for CPSC8430, Clemson University



## Install
This project uses Python 3 based on Jupyter Notebook

This project is based on Pytorch 1.7.0 & cudnn 1.10
Official website: <a href="https://pytorch.org/get-started/previous-versions/">Previous PyTorch Versions</a>
```
pip install -f https://download.pytorch.org/whl/cu110/torch_stable.html torch==1.7.0+cu110 torchvision==0.8.0 --user
```



## Packages
This list gives all recommended packages (may not necessary)
```sh
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"
CUDA_LAUNCH_BLOCKING=1

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import torchvision.transforms as transformtransforms

from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
```



## Contributing
This project is contributed by: 
<a href="hao9@g.clemson.edu">hao9@clemson.edu</a>
