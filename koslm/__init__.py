import os
import torch

# 手动把 torch 的 lib 路径加入 DLL 搜索路径
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if hasattr(os, "add_dll_directory"):  # Python 3.8+
    os.add_dll_directory(torch_lib_path)
else:
    os.environ["PATH"] += os.pathsep + torch_lib_path

# # 再导入 CUDA 扩展
# from . import koslm_cuda
