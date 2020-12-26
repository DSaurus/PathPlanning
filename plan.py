import torch
import numpy as np
import matplotlib.pyplot as plt

pts = np.zeros((15, 2), dtype=float)

f = open('pts.txt', 'r')
for i in range(15):
    line = f.readline().split(' ')
    pts[i][0] = float(line[0])
    pts[i][1] = float(line[1])
pts = torch.FloatTensor(pts)
plan = []
for i in range(15):
    plan.append(int(f.readline()))

# load(pts_plan)
pts_plan = []
for i in range(7*14):
    pts_plan.append(torch.tensor([0.0, 0.0], requires_grad=True))
# pts_plan = torch.cat(pts_plan, dim=0)
optim = torch.optim.Adam(pts_plan, lr=1e-1)

for _ in range(1000):
    loss = 0
    for i in range(14):
        loss += torch.sum((pts_plan[i*7] - pts[plan[i]])**2)
        # print(pts_plan[i*7], pts[plan[i]])
    loss += torch.sum((pts_plan[-1] - pts[plan[-1]])**2)

    for i in range(7*14-1):
        # loss += torch.mean( torch.abs((pts_plan[i+2] - pts_plan[i+1])**2 - (pts_plan[i+1] - pts_plan[i])**2))
        loss += torch.sum( (pts_plan[i+1] - pts_plan[i])**2)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(_)

# save(pts_plan)
x = []
y = []
for i in range(7*14):
    x.append(pts_plan[i][0])
    y.append(pts_plan[i][1])
plt.plot(x, y, '-')
plt.plot(x, y, '.')
plt.savefig('test.jpg')




