export MASTER_ADDR="115.145.175.206"
export MASTER_PORT="23111"
export WORLD_SIZE="4"
export RANK="0"
export LOCAL_RANK="0"
# export OPENAI_API_KEY=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
export TRANSFORMERS_CACHE=/home/franka/.cache/huggingface/hub
export RAVENS_ROOT=/home/franka/fr3_workspace/RealWorldLLM/ravens/

# CUDA_VISIBLE_DEVICES=1 python llm_eval.py task=towers-of-hanoi-seq-seen-colors \
#                                           agent_mode=2 \
#                                           mode=test \
#                                           record.save_video=True \
#                                           task_level=3

# CUDA_VISIBLE_DEVICES=1 python llm_eval.py task=towers-of-hanoi-seq-seen-colors \
#                                           agent_mode=2 \
#                                           mode=test \
#                                           record.save_video=True \
#                                           task_level=2
python3 llm_eval.py task=towers-of-hanoi-seq-seen-colors \
                                          agent_mode=2 \
                                          mode=test \
                                          record.save_video=False \
                                          task_level=1