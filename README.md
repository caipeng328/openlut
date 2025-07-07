# OPENLUT

Here are two implementations: one uses Eigen to quickly read LUT files with an average speed of about 0.5 seconds; the other is a GPU-accelerated version based on PyTorch, achieving an average speed of around 0.1 seconds.

这里有两种实现方案：一种是基于 Eigen 实现的快速读取 LUT 文件，平均速度约为 0.5 秒；另一种是基于 PyTorch 的 GPU 加速版本，平均速度约为 0.1 秒。

## CPU Version Build and Install

### requirements:
- pybind11
- eigen
- the C++ compiler support C++14 and openmp

### install: 

```
mkdir build
cmake ..
add the generated .so file to the path where python can find it.
```

Or you can directly use the Python 3.8 version of the code we compiled. **ApplyLUT.cpython-38-x86_64-linux-gnu.so**

### How to use
```
python CPU_inference.py
```

## GPU Version Build and Install

### requirements:
- torch
- opencv-python
- ### How to use
```
python GPU_inference.py
```