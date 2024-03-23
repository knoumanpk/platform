TITLE="PADQN"
MULTIPASS=""
SEED="1"
EVALUATION_EPISODES="1000"
EPISODES="80000"
BATCH_SIZE="128"
GAMMA="0.9"
REPLAY_MEMORY_SIZE="10000"
INITIAL_MEMORY_THRESHOLD="500"
EPSILON_INITIAL="1"
EPSILON_STEPS="1000"
EPSILON_FINAL="0.01"
ALPHA_ACTOR="0.001"
ALPHA_PARAM_ACTOR="0.0001"
TAU_ACTOR="0.1"
TAU_PARAM_ACTOR="0.001"
LOSS="l1_smooth"
USE_ORNSTEIN_NOISE="True"
SCALE_ACTIONS="True"
CLIP_GRAD="10"
RESULTS_DIR="results"
SAVE_FREQ="1000"
RENDER_FREQ="100"
CHECK_POINT="None"

source env_ubuntu/bin/activate
cd ./../

python3 run_platform_padqn.py --title $TITLE --multipass $MULTIPASS --seed $SEED --evaluation_episodes $EVALUATION_EPISODES --episodes $EPISODES --batch_size $BATCH_SIZE --gamma $GAMMA --replay_memory_size $REPLAY_MEMORY_SIZE --initial_memory_threshold $INITIAL_MEMORY_THRESHOLD --epsilon_initial $EPSILON_INITIAL --epsilon_steps $EPSILON_STEPS --epsilon_final $EPSILON_FINAL --alpha_actor $ALPHA_ACTOR --alpha_param_actor $ALPHA_PARAM_ACTOR --tau_actor $TAU_ACTOR --tau_param_actor $TAU_PARAM_ACTOR --loss $LOSS --use_ornstein_noise $USE_ORNSTEIN_NOISE --scale_actions $SCALE_ACTIONS --clip_grad $CLIP_GRAD --results_dir $RESULTS_DIR --save_freq $SAVE_FREQ --render_freq $RENDER_FREQ

#PAUSE