export MASTER_ADDR="115.145.175.206"
export MASTER_PORT="23111"
export WORLD_SIZE="4"
export RANK="0"
export LOCAL_RANK="0"
# export OPENAI_API_KEY=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
export TRANSFORMERS_CACHE=/home/mnt/models/.cache/huggingface/hub
export RAVENS_ROOT=/home/pjw971022/Sembot/ravens/

python3 llm_eval.py task=towers-of-hanoi-seq-seen-colors \
                                          agent_mode=1 \
                                          mode=test \
                                          record.save_video=True \
                                          task_level=1