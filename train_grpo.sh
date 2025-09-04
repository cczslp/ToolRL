export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export ROLLOUT_TP_SIZE=1
# export VLLM_ATTENTION_BACKEND=XFORMERS

# All the env variables below are set to 0 by default
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

export CKPT_DIR="/root/autodl-tmp/ToolRL/ckpts"
export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="/root/autodl-tmp/models/qwen2.5-1.5b-it" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="grpo_qwen2.5_1.5b" # e.g., "grpo-qwen2.5-3b"
export SWANLAB_API_KEY="Put your own api key here"   # 设置在线跟踪模式API
export SWANLAB_LOG_DIR="/root/autodl-tmp/ToolRL/logs/"    # 设置本地日志存储路径
bash ./examples/grpo_trainer/run_grpo.sh
