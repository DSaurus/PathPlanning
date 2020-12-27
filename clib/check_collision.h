#ifndef CHECK_COLLISION
#define CHECK_COLLISION

#include <torch/torch.h>
#include <iostream>

at::Tensor calc_collision(torch::Tensor pts_plan, torch::Tensor tri);

#endif