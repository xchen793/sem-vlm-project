from huggingface_hub import whoami, model_info

print(whoami())  # Shows your username

# This should succeed if access is granted
info = model_info("meta-llama/Meta-Llama-3-8B-Instruct")
print("✅ Access confirmed:", info.modelId)
