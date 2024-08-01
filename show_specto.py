import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

img = torch.load('..\\spectrograms\\S00_2_0.pt')

for index, i in enumerate(img):
    arr = i.detach().cpu().numpy()*255.   
    im = Image.fromarray(arr).convert('L')
    im.save(f"tmp\\{index}.jpg")