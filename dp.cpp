#include <iostream>
#include <cmath>
using namespace std;
const int n = 15;
double dp[n][(1<<n)];
int pre[n][(1<<n)];

struct Point{
    double x, y;
}p[15];

double dis(int i, int j){
    return sqrt( (p[i].x - p[j].x)*(p[i].x - p[j].x) + (p[i].y - p[j].y)*(p[i].y - p[j].y));
}

int main()
{
    freopen("points.txt", "r", stdin);
    for(int i = 0; i < n; i++){
        cin>>p[i].x>>p[i].y;
        cout<<p[i].x<<" "<<p[i].y<<endl;
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < (1<<n); j++){
            dp[i][j] = 1e9;
        }
    }
    dp[0][1] = 0;
    for(int step = 1; step <= n; step++){
        for(int i = 0; i < n; i++){
            for(int s = 0; s < (1<<n); s++){
                if(dp[i][s] > 1e8) continue;
                for(int j = 0; j < n; j++){
                    if((1<<j) & s) continue;
                    if(dp[j][s | (1<<j)] > dp[i][s] + dis(i, j)){
                        dp[j][s | (1<<j)] = dp[i][s] + dis(i, j);
                        pre[j][s | (1<<j)] = i;
                    }
                }
            }
        }
    }
    double dis = 1e9;
    for(int i = 1; i < n; i++){
        cout<<i<<" "<<dp[i][(1<<n)-1]<<endl;
        if(i == 12  ) {
            int s = i;
            int ss = (1<<n)-1;
            for(int step=1; step <= n; step++){
                cout<<pre[s][ss]<<endl;
                int ns = pre[s][ss];
                ss = ss & (~(1<<s));
                s = ns;
            }
        }
    }
    return 0;
}
