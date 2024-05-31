import torch
torch.manual_seed(137)
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
t = torch.zeros(8);
t.normal_()
for i in range(len(t)) :
    print(t[i].item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
t = torch.zeros(16);
t.normal_()
for i in range(len(t)) :
    print(t[i].item())
print(torch.randint(0, 0xFFFFFFFF, [1]).item())
