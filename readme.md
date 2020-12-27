# Path Planning

![](https://raw.githubusercontent.com/DSaurus/PathPlanning/master/vehicle.png)
![](https://raw.githubusercontent.com/DSaurus/PathPlanning/master/result/route3/test.jpg)

The planning requirements are as follows:

- minimize the length of the planned path as much as possible;
- vehicle must avoid all obstacles;
- vehicle must clean all "rubbishes"
- the turning angle of the vehicle should not be larger than 36 degree.

## Installation

```
pip install -r requirements.txt
cd clib
python setup.py build install
```

## How to Run?

### Dynamic Programming

Use `g++` to compile `dp.cpp` and it is directly executable.

### Optimization

To run the optimization for path planning:

```
sh run.sh
```


