import torch


index = torch.tensor([[ 0,  1],
                      [ 1,  2],
                      [ 1,  4]])

a = torch.zeros(3,6,3)

# a[torch.arange(a.size(0)).unsqueeze(1), index] = 1.
a[index[:,0],index[:,1],:] = 1.

print(a)

