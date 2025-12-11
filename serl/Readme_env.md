 # 常见的环境问题解决:
 

 1. 降级相关库

        pip install "dm-robotics-transformations<0.10.0"

        pip install "numpy<2.0"

 2. 修改JAX版本

        卸载所有相关库，清理环境
        pip uninstall -y jax jaxlib jax-cuda12-plugin optax orbax-checkpoint chex distrax flax

        安装 JAX 0.4.35 (最稳定的近期版本)
        pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

        安装兼容 JAX 0.4.35 的生态库版本
        pip install "flax<=0.8.5" "optax<=0.2.2" "chex<=0.1.86" "distrax<=0.1.5" "orbax-checkpoint<=0.5.23"


 3. 安装缺失的 NVIDIA 运行时组件
 
        pip install --force-reinstall --no-cache-dir nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

 4. 给 nvidia.cuda_nvcc 补一个真实文件路径

        mkdir -p "$SITE_PKGS/nvidia/cuda_nvcc"

        cat <<'EOF' > "$SITE_PKGS/nvidia/cuda_nvcc/__init__.py"
        """Shim module so JAX sees a real CUDA NVCC package."""
        import pathlib

        NVCC_PATH = pathlib.Path("/usr/local/cuda/bin/nvcc")
        if not NVCC_PATH.exists():
            raise RuntimeError(f"Expected NVCC at {NVCC_PATH}, please update __init__.py")
        __file__ = str(NVCC_PATH)
        __path__ = [str(NVCC_PATH.parent)]
        EOF

