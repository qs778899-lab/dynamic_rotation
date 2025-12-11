 # 常见的环境问题解决:
 

 1. pip install "dm-robotics-transformations<0.10.0"

 2. pip install "numpy<2.0"


    # 1. 卸载所有相关库，清理环境
    pip uninstall -y jax jaxlib jax-cuda12-plugin optax orbax-checkpoint chex distrax flax

    # 2. 安装 JAX 0.4.35 (最稳定的近期版本)
    pip install "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # 3. 安装兼容 JAX 0.4.35 的生态库版本
    pip install "flax<=0.8.5" "optax<=0.2.2" "chex<=0.1.86" "distrax<=0.1.5" "orbax-checkpoint<=0.5.23"

        pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12

        pip install --force-reinstall --no-cache-dir nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

