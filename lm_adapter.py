import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

def load_mistral_cpu(rank: int = 8):
    print("🔄 Loading Mistral-7B (CPU, float16)…")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,    # use float32 if you lack RAM
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer


# quick sanity check
if __name__ == "__main__":
    model, tok = load_mistral_cpu()
    prompt = "Describe a plasma-clean reaction in one sentence."
    ids = tok(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=40)
    print(tok.decode(out[0], skip_special_tokens=True))
