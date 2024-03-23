@echo off
set TITLE="PADQN"
set MULTIPASS=""
set SEED="1"
set EVALUATION_EPISODES="1000"
set EPISODES="80000"
set BATCH_SIZE="128"
set GAMMA="0.9"
set REPLAY_MEMORY_SIZE="10000"
set INITIAL_MEMORY_THRESHOLD="500"
set EPSILON_INITIAL="1"
set EPSILON_STEPS="1000"
set EPSILON_FINAL="0.01"
set ALPHA_ACTOR="0.001"
set ALPHA_PARAM_ACTOR="0.0001"
set TAU_ACTOR="0.1"
set TAU_PARAM_ACTOR="0.001"
set LOSS="l1_smooth"
set USE_ORNSTEIN_NOISE="True"
set SCALE_ACTIONS="True"
set CLIP_GRAD="10"
set RESULTS_DIR="results"
set SAVE_FREQ="1000"
set RENDER_FREQ="100"
set CHECK_POINT="None"

cd ..\
call venv\Scripts\activate.bat

python run_platform_padqn.py --title %TITLE% --multipass %MULTIPASS% --seed %SEED% --evaluation_episodes %EVALUATION_EPISODES% --episodes %EPISODES% --batch_size %BATCH_SIZE% --gamma %GAMMA% --replay_memory_size %REPLAY_MEMORY_SIZE% --initial_memory_threshold %INITIAL_MEMORY_THRESHOLD% --epsilon_initial %EPSILON_INITIAL% --epsilon_steps %EPSILON_STEPS% --epsilon_final %EPSILON_FINAL% --alpha_actor %ALPHA_ACTOR% --alpha_param_actor %ALPHA_PARAM_ACTOR% --tau_actor %TAU_ACTOR% --tau_param_actor %TAU_PARAM_ACTOR% --loss %LOSS% --use_ornstein_noise %USE_ORNSTEIN_NOISE% --scale_actions %SCALE_ACTIONS% --clip_grad %CLIP_GRAD% --results_dir %RESULTS_DIR% --save_freq %SAVE_FREQ% --render_freq %RENDER_FREQ%

:: PAUSE