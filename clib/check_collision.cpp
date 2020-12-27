#include "check_collision.h"

torch::Tensor mynorm(torch::Tensor x){
    return x / torch::sqrt(torch::sum(x*x));
}

double cross(torch::Tensor p1, torch::Tensor p2, torch::Tensor p0){
    return ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])).item().to<double>();
}

bool zero(double x){
    return (abs(x) < 1e-6);
}

bool dot_online_in(torch::Tensor p, torch::Tensor l1, torch::Tensor l2){
    return (zero(cross(p, l1, l2)) && (((l1[0]-p[0])*(l2[0]-p[0])).item().to<double>() < 1e-6) && (((l1[1]-p[1])*(l2[1]-p[1])).item().to<double>() < 1e-6));
}

bool same_side(torch::Tensor p1, torch::Tensor p2, torch::Tensor l1, torch::Tensor l2){
    return (cross(l1, p1, l2)*cross(l1, p2, l2) > 1e-6);
}

bool parallel(torch::Tensor u1, torch::Tensor u2, torch::Tensor v1, torch::Tensor v2){
    return zero(((u1[0]-u2[0])*(v1[1]-v2[1])-(v1[0]-v2[0])*(u1[1]-u2[1])).item().to<double>());
}

bool dots_inline(torch::Tensor p1, torch::Tensor p2, torch::Tensor p3){
    return zero(cross(p1, p2, p3));
}

bool intersect(torch::Tensor u1, torch::Tensor u2, torch::Tensor v1, torch::Tensor v2){
    if (parallel(u1, u2, v1, v2))
        return 0;
    return cross(u2, v1, u1)*cross(u2, v2, u1) < 0 && cross(v2, u1, v1)*cross(v2, u2, v1) < 0;
}

int check_seg(torch::Tensor pt1, torch::Tensor pt2, torch::Tensor& tri){
    for(int i = 0; i < tri.size(0); i++){
        for(int j = 0; j < 2; j++){
            if(intersect(pt1, pt2, tri[i][j], tri[i][j+1]))
                return i;
        }
        if(intersect(pt1, pt2, tri[i][2], tri[i][0]))
            return i;
    }
    return -1;
}

int calc_collision2(torch::Tensor pts1, torch::Tensor pts2, torch::Tensor pts3, torch::Tensor& tri){
    torch::Tensor v1 = (pts2 - pts1);
    torch::Tensor v2 = (pts3 - pts2);
    torch::Tensor vx = mynorm(mynorm(v1) + mynorm(v2));
    torch::Tensor vy = torch::zeros_like(vx);
    vy[0] = -vx[1];
    vy[1] = vx[0];
    double Y = 0.285;
    double X = 0.35;
    torch::Tensor pt1 = torch::zeros_like(vx), pt2 = torch::zeros_like(vy);

    int flag = -1;
    pt1[0] = pts3[0] + X*vx[0] + Y*vy[0]; pt1[1] = pts3[1] + X*vx[1] + Y*vy[1];
    pt2[0] = pts2[0] - X*vx[0] + Y*vy[0]; pt2[1] = pts2[1] - X*vx[1] + Y*vy[1];
    flag = std::max(flag, check_seg(pt1, pt2, tri));
    
    pt1[0] = pts2[0] - X*vx[0] + Y*vy[0]; pt1[1] = pts2[1] - X*vx[1] + Y*vy[1];
    pt2[0] = pts2[0] - X*vx[0] - Y*vy[0]; pt2[1] = pts2[1] - X*vx[1] - Y*vy[1];
    flag = std::max(flag, check_seg(pt1, pt2, tri));

    pt1[0] = pts2[0] - X*vx[0] - Y*vy[0]; pt1[1] = pts2[1] - X*vx[1] - Y*vy[1];
    pt2[0] = pts3[0] + X*vx[0] - Y*vy[0]; pt2[1] = pts3[1] + X*vx[1] - Y*vy[1];
    flag = std::max(flag, check_seg(pt1, pt2, tri));

    pt1[0] = pts3[0] + X*vx[0] - Y*vy[0]; pt1[1] = pts3[1] + X*vx[1] - Y*vy[1];
    pt2[0] = pts3[0] + X*vx[0] + Y*vy[0]; pt2[1] = pts3[1] + X*vx[1] + Y*vy[1];
    flag = std::max(flag, check_seg(pt1, pt2, tri));

    return flag;
}

at::Tensor calc_collision(torch::Tensor pts_plan, torch::Tensor tri){
    at::Tensor cost = at::zeros({7*14-2});
#pragma omp parallel for num_threads(16) 
    for(int i = 0; i < 7*14-2; i++){
        int flag = calc_collision2(pts_plan[i], pts_plan[i+1], pts_plan[i+2], tri);
        if(flag >= 0){
            at::Tensor center = (tri[flag][0] + tri[flag][1] + tri[flag][2]) / 3;
            cost[i] = -1e2*at::sum( (pts_plan[i+1] - center)*(pts_plan[i+1] - center));
        }
    }
    return cost;
}

at::Tensor check_in(torch::Tensor pts1, torch::Tensor pts2, torch::Tensor pts3, torch::Tensor data){
    torch::Tensor v1 = -(pts2 - pts1);
    torch::Tensor v2 = -(pts3 - pts2);
    torch::Tensor vx = mynorm(mynorm(v1) + mynorm(v2));
    torch::Tensor vy = torch::zeros_like(vx);
    vy[0] = -vx[1];
    vy[1] = vx[0];
    double Y = 0.285;
    double X = 0.35;
    torch::Tensor pt1 = torch::zeros_like(vx), pt2 = torch::zeros_like(vy), pt3 = torch::zeros_like(vy), pt4 = torch::zeros_like(vy);

    pt1[0] = pts2[0] + X*vx[0] + Y*vy[0]; pt1[1] = pts2[1] + X*vx[1] + Y*vy[1];
    pt2[0] = pts2[0] - X*vx[0] + Y*vy[0]; pt2[1] = pts2[1] - X*vx[1] + Y*vy[1];
    pt3[0] = pts2[0] - X*vx[0] - Y*vy[0]; pt3[1] = pts2[1] - X*vx[1] - Y*vy[1];
    pt4[0] = pts2[0] + X*vx[0] - Y*vy[0]; pt4[1] = pts2[1] + X*vx[1] - Y*vy[1];
    bool flag = cross(pt1, pt2, data) * cross(pt3, pt4, data) >= 0 && cross(pt2, pt3, data) * cross(pt4, pt1, data) >= 0;
    if(flag){
        return torch::zeros({1});
    } else{
        return 1e2*torch::sum((pts2 - data)*(pts2 - data));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calc_collision", &calc_collision, "calc_collision (CPU)");
    m.def("check_in", &check_in, "check_in (CPU)");
}