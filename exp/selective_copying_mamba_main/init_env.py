# init_env.py

import os
import subprocess


def load_visual_studio_env(vcvars_path=None):
    """
    从 vcvars64.bat 加载 Visual Studio 编译器环境，并设置 CC=cl。
    """
    if vcvars_path is None:
        # 默认路径（你可以根据实际 VS 安装路径修改）
        vcvars_path = r"D:\vs2022\vs IDE\Community\VC\Auxiliary\Build\vcvars64.bat"

    # 构建命令并获取环境变量
    command = f'"{vcvars_path}" && set'
    try:
        output = subprocess.check_output(command, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to load vcvars64.bat. Please make sure the path is correct.") from e

    # 解析并更新环境变量
    for line in output.splitlines():
        if '=' in line:
            key, val = line.strip().split('=', 1)
            os.environ[key] = val

    # 明确设置 CC=cl
    os.environ["CC"] = "cl"

    print("[INFO] Visual Studio compiler environment loaded (CC=cl)")


