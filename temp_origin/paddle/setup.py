import paddle

paddle.enable_compat()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="repro_ext",
    ext_modules=[
        CUDAExtension(
            name="repro_ext",
            sources=["repro_ext.cu"],
            extra_compile_args={
                "cxx": ["-O0", "-g", "-pthread"],
                "nvcc": ["-O0", "-g"],
            },
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
)
