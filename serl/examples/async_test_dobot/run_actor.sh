export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_test_dobot.py "$@" \
    --actor \
    --env DobotEnv-v0 \
    --exp_name=serl_dobot_drq \
    --seed 0 \
    --random_steps 0 \
    --training_starts 500 \
    --encoder_type resnet-pretrained

