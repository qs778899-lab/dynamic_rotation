export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_test_dobot.py "$@" \
    --learner \
    --env DobotEnv-v0 \
    --exp_name=serl_dobot_drq \
    --seed 0 \
    --random_steps 500 \
    --training_starts 500 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 0 \
    --encoder_type resnet-pretrained \
    --checkpoint_period 0

