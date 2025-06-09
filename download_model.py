from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
model_path = "/workspace/deepseek"

# This will only download the files
AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=model_path,
    local_files_only=False  # Forces download
)

AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=model_path,
    local_files_only=False  # Forces download
)