import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--init", action='store_true')
    parser.add_argument("--workdir", type=str, default='test')
    parser.add_argument("--checkpoint", type=str, default='test')
    return parser

def load_data(filename):
    pts = np.zeros((15, 2), dtype=float)
    f = open(filename, 'r')
    for i in range(15):
        line = f.readline().split(' ')
        pts[i][0] = float(line[0])
        pts[i][1] = float(line[1])
    pts = torch.FloatTensor(pts)
    plan = []
    for i in range(15):
        plan.append(int(f.readline()))
    return pts, plan

def load_checkpoint(filename):
    f = open(filename, 'r')
    pts_plan = []
    for i in range(7*14):
        x, y = f.readline().split(' ')
        pts_plan.append(torch.tensor([float(x), float(y)], requires_grad=True))
    return pts_plan

def save_checkpoint(workdir, pts_plan):
    filename = os.path.join(workdir, 'checkpoint.txt')
    f = open(filename, 'w')
    for i in range(7*14):
        f.write("%.18f %.18f\n" % (pts_plan[i][0], pts_plan[i][1]))


parser = config_parser()
config = parser.parse_args()
workdir = config.workdir
os.makedirs(workdir, exist_ok=True)

pts, plan = load_data('pts.txt')

if config.init:
    pts_plan = []
    for i in range(7*14):
        pts_plan.append(torch.tensor([0.0, 0.0], requires_grad=True))
else:
    pts_plan = load_checkpoint(config.checkpoint)

optim = torch.optim.Adam(pts_plan, lr=1e-2)

for _ in range(100):
    loss = 0
    # data term
    for i in range(14):
        loss += torch.sum((pts_plan[i*7] - pts[plan[i]])**2)
    loss += torch.sum((pts_plan[-1] - pts[plan[-1]])**2)

    # angle term
    angle_t = math.cos(0.2*math.acos(-1))
    for i in range(7*14-2):
        a = pts_plan[i+2] - pts_plan[i+1]
        b = pts_plan[i+1] - pts_plan[i]
        cos = torch.sum(a*b) / torch.sqrt(torch.sum(a*a) * torch.sum(b*b))
        # linear
        if cos > angle_t:
            loss += -cos * 0.1
        else:
            loss += -cos * 10
        # log
        # if cos - angle_t > 1e-6:
        #     loss += -torch.log( (cos - angle_t))
        # else:
        #     loss += -1e2 * (cos - angle_t)

    # distance term
    for i in range(7*14-1):
        # loss += torch.mean( torch.abs((pts_plan[i+2] - pts_plan[i+1])**2 - (pts_plan[i+1] - pts_plan[i])**2))
        loss += torch.sum( (pts_plan[i+1] - pts_plan[i])**2)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(_)

save_checkpoint(workdir, pts_plan)
x = []
y = []
for i in range(7*14):
    x.append(pts_plan[i][0])
    y.append(pts_plan[i][1])
plt.plot(x, y, '-')
plt.plot(x, y, '.')
plt.savefig(os.path.join(workdir, 'test.jpg'))




