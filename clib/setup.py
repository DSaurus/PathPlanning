from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME
from torch.utils.cpp_extension import CppExtension, CUDAExtension
import torch

# In any case, include the CPU version
modules = [
    CppExtension('check_collision',
                 ['check_collision.cpp'],
                 extra_compile_args=['-fopenmp']),
]

include_dirs = torch.utils.cpp_extension.include_paths()
print(include_dirs)
include_dirs.append('.')

# Now proceed to setup
setup(
    name='check_collision',
    version='0.1',
    author='DSaurus',
    author_email='2238454358@qq.com',
    packages=find_packages(where='.'),
    package_dir={"": "."},
    ext_modules=modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
