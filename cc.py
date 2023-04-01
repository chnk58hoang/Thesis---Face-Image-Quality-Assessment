import torch.nn as nn
import torch

st1 = torch.load('weights/backbone.pth',map_location='cpu')
st2 = torch.load('weights/pose1.pth',map_location='cpu')

res = []
for name in st2:
    sname = name.replace('backbone.','')
    if sname in st1 and not torch.equal(st1[sname],st2[name]):
        res.append((sname,name)

print(res)

