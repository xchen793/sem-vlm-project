# infer.py  – memory-efficient SEM generation
#
# Requirements
#   • diffusers 0.28  • transformers 4.41  • accelerate 0.30
#   • adapters saved by train.py  in  checkpoints/{vis_lora,lm_lora,film_proj.pt}
#
# Run:
#   python infer.py

from pathlib import Path
import torch, json
from diffusers import StableDiffusionXLPipeline
from diffusers import utils as dutils
from accelerate import init_empty_weights
from peft import PeftModel
from lm_adapter import load_mistral_cpu     # returns (lm_adapter, lm_tok)
from train import FiLMProj                          # same class as in training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16

# ──────────────────────────────────────────────
# 1.  Load SDXL in *ultra-low-memory* mode
# ──────────────────────────────────────────────
print("🔄 loading SDXL-base … (fp16, streamed)")
pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,        # stream tensors
        offload_folder="offload",      # tensors swap here if RAM tight
)
pipe.to(DEVICE)
pipe.enable_model_cpu_offload()        # off-load modules when inactive
pipe.enable_attention_slicing()        # slice attn to save RAM

# attach visual LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet,
                                      "checkpoints/vis_lora",
                                      torch_dtype=DTYPE).to(DEVICE)

# ──────────────────────────────────────────────
# 2.  Load reasoning LM adapter (CPU float16)
# ──────────────────────────────────────────────
print("🔄 loading Mistral LoRA …")
lm_adapter, lm_tok = load_mistral_cpu(rank=8)  # returns adapter & tokenizer
lm_adapter.load_adapter("checkpoints/lm_lora", "default")
lm_adapter.to("cpu", dtype=DTYPE).eval()       # keep on CPU to save GPU RAM

# ──────────────────────────────────────────────
# 3.  Load FiLM projector and attach hooks
# ──────────────────────────────────────────────
film_proj = FiLMProj().to("cpu").to(DTYPE)
film_proj.load_state_dict(torch.load("checkpoints/film_proj.pt", map_location="cpu"))

def add_film_hooks(unet):
    for mod in unet.modules():
        if hasattr(mod, "conv2"):
            def _hook(module, inp, out):
                C = out.shape[1]
                γ, β = film_proj(unet.z_r.to(DTYPE), C)  # ensure dtype
                return out * (1 + γ[..., None, None]) + β[..., None, None]
            mod.register_forward_hook(_hook)

add_film_hooks(pipe.unet)

# ──────────────────────────────────────────────
# 4.  Generation wrapper
# ──────────────────────────────────────────────
@torch.inference_mode()
def generate_sem(proc_text: str,
                 prompt: str,
                 steps: int = 30,
                 guidance: float = 7.5,
                 seed: int = 42):
    torch.manual_seed(seed)

    # reasoning vector  (runs on CPU, stays fp16)
    ids = lm_tok(proc_text, return_tensors="pt").input_ids.to("cpu")
    z_r = lm_adapter.model.model.embed_tokens(ids).mean(1)   # [1,4096]
    pipe.unet.z_r = z_r                                      # FiLM uses this

    img = pipe(prompt,
               num_inference_steps=steps,
               guidance_scale=guidance).images[0]
    return img

# _____________________________
# 5.  Quick demo
# _____________________________
if __name__ == "__main__":
    procedure = (
        "Immersion experiment for PPy/PVA hydrogel conducted for 128 hours "
        "under standard laboratory conditions."
    )
    prompt = "Please generate a SEM image of the PPy/PVA double-network hydrogel after 128 hours of immersion"

    image = generate_sem(procedure, prompt)
    Path("output").mkdir(exist_ok=True)
    image.save("output/sem_gen.png")
    image.show()
    print("✅ image saved to output/sem_gen.png")
