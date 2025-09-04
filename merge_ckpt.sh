python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /root/autodl-tmp/ToolRL/ckpts/global_step_105/actor \
  --target_dir ./merged_hf/merged_qwen_1.5b