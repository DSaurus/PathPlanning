import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from check_collision import calc_collision, check_in

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--init", action='store_true')
    parser.add_argument("--workdir", type=str, default='test')
    parser.add_argument("--checkpoint", type=str, default='test')
    parser.add_argument("--input", type=str, default='pts.txt')
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--its", type=int, default=1000)
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
    tri = np.zeros((3+3+2+1, 3, 2))
    ind = 0
    for i in range(4):
        line = f.readline().split(' ')
        n = int(line[0])
        tri_pts = np.zeros((n, 2), dtype=float)
        for j in range(n):
            tri_pts[j][0] = float(line[1+j*2])
            tri_pts[j][1] = float(line[2+j*2])
        for j in range(1, n-1):
            tri[ind][0] = tri_pts[0]
            tri[ind][1] = tri_pts[j]
            tri[ind][2] = tri_pts[j+1]
            ind += 1
        
    return pts, plan, tri

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


def draw(workdir, pts_plan, pts, tri):

    pt2 = pts_plan[0]
    pt3 = pts_plan[1]
    vx = (pt3 - pt2) / torch.sqrt(torch.sum( (pt3-pt2)**2))
    vy = torch.zeros_like(vx)
    vy[0] = -vx[1]
    vy[1] = vx[0]
    Y = 0.285
    X = 0.35
    p1 = torch.zeros_like(vx)
    p2 = torch.zeros_like(vx)
    p3 = torch.zeros_like(vx)
    p4 = torch.zeros_like(vx)

    p1[0] = pt2[0] + X*vx[0] + Y*vy[0]
    p1[1] = pt2[1] + X*vx[1] + Y*vy[1]
    p2[0] = pt2[0] - X*vx[0] + Y*vy[0]
    p2[1] = pt2[1] - X*vx[1] + Y*vy[1]
    p3[0] = pt2[0] - X*vx[0] - Y*vy[0]
    p3[1] = pt2[1] - X*vx[1] - Y*vy[1]
    p4[0] = pt2[0] + X*vx[0] - Y*vy[0]
    p4[1] = pt2[1] + X*vx[1] - Y*vy[1]
    xx = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    yy = [p1[1], p2[1], p3[1], p4[1], p1[1]]
    plt.plot(xx, yy, 'b-')

    for i in range(7*14-2):
        pt1 = pts_plan[i]
        pt2 = pts_plan[i+1]
        pt3 = pts_plan[i+2]
        v1 = (pt2 - pt1) / torch.sqrt(torch.sum( (pt2-pt1)**2))
        v2 = (pt3 - pt2) / torch.sqrt(torch.sum( (pt3-pt2)**2))
        vx = (v1 + v2) / torch.sqrt(torch.sum( (v1 + v2)**2))
        vy = torch.zeros_like(vx)
        vy[0] = -vx[1]
        vy[1] = vx[0]
        Y = 0.285
        X = 0.35
        p1 = torch.zeros_like(vx)
        p2 = torch.zeros_like(vx)
        p3 = torch.zeros_like(vx)
        p4 = torch.zeros_like(vx)

        p1[0] = pt2[0] + X*vx[0] + Y*vy[0]
        p1[1] = pt2[1] + X*vx[1] + Y*vy[1]
        p2[0] = pt2[0] - X*vx[0] + Y*vy[0]
        p2[1] = pt2[1] - X*vx[1] + Y*vy[1]
        p3[0] = pt2[0] - X*vx[0] - Y*vy[0]
        p3[1] = pt2[1] - X*vx[1] - Y*vy[1]
        p4[0] = pt2[0] + X*vx[0] - Y*vy[0]
        p4[1] = pt2[1] + X*vx[1] - Y*vy[1]
        xx = [p1[0], p2[0], p3[0], p4[0], p1[0]]
        yy = [p1[1], p2[1], p3[1], p4[1], p1[1]]
        plt.plot(xx, yy, 'b-')
    
    x = []
    y = []
    for i in range(7*14):
        x.append(pts_plan[i][0])
        y.append(pts_plan[i][1])
    plt.plot(x, y, '-')
    plt.plot(x, y, '.')

    for i in range(tri.shape[0]):
        x, y = [], []
        for j in range(3):
            x.append(tri[i][j][0])
            y.append(tri[i][j][1])
        x.append(tri[i][0][0])
        y.append(tri[i][0][1])
        print(x, y)
        plt.plot(x, y, 'r-')
    
    x, y = [], []
    for i in range(14):
        x.append(pts[i][0])
        y.append(pts[i][1])
    plt.plot(x, y, 'g.', ms=10)
    plt.savefig(os.path.join(workdir, 'test.jpg'))


parser = config_parser()
config = parser.parse_args()
workdir = config.workdir
os.makedirs(workdir, exist_ok=True)

pts, plan, tri = load_data(config.input)

if config.init:
    pts_plan = []
    for i in range(7*14):
        pts_plan.append(torch.tensor([0.0, 0.0], requires_grad=True))
else:
    pts_plan = load_checkpoint(config.checkpoint)

optim = torch.optim.Adam(pts_plan, lr=config.lr)
its = config.its

import time
t = time.time()
for _ in range(its):
    loss = 0
    # data term
    if config.init:
        for i in range(14):
            loss += torch.sum((pts_plan[i*7] - pts[plan[i]])**2) * 1e2
        loss += torch.sum((pts_plan[-1] - pts[plan[-1]])**2) * 1e2
    

    # angle term
    if not config.init:
        for i in range(14):
            loss += torch.sum(check_in(pts_plan[i*7-1], pts_plan[i*7], pts_plan[i*7+1], pts[plan[i]]))
        loss += torch.sum(check_in(pts_plan[-3], pts_plan[-2], pts_plan[-1], pts[plan[-1]]))
        angle_t = math.cos(0.2*math.acos(-1))
        for i in range(7*14-2):
            a = pts_plan[i+2] - pts_plan[i+1]
            b = pts_plan[i+1] - pts_plan[i]
            cos = torch.sum(a*b) / torch.sqrt(torch.sum(a*a) * torch.sum(b*b))
            # linear
            if cos > angle_t:
                loss += 0
            else:
                loss += -cos * 10
            # log
            # if cos - angle_t > 1e-6:
            #     loss += -torch.log( (cos - angle_t))
            # else:
            #     loss += -1e2 * (cos - angle_t)

        # collision term
        collision = calc_collision(torch.cat(pts_plan, dim=0).reshape(-1, 2), torch.FloatTensor(tri))
        loss += torch.sum(collision)
    
    # distance term
    ans = 0
    for i in range(7*14-1):
        loss += torch.sum( (pts_plan[i+1] - pts_plan[i])**2)
        ans += torch.sqrt(torch.sum( (pts_plan[i+1] - pts_plan[i])**2))
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(_, ans)

save_checkpoint(workdir, pts_plan)
draw(workdir, pts_plan, pts, tri)
print(time.time() - t)



