# Path Planning

The planning requirements are as follows:

- 1)	minimize the length of the planned path as much as possible;
- 2)	vehicle must avoid all obstacles;
- 3)	vehicle must clean all "rubbishes"
- 4)	the turning angle of the vehicle should not be larger than 36 degree.

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


