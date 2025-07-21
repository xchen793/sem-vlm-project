# train.py – few-shot joint LoRA fine-tune

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from lm_adapter import load_mistral_cpu
from dataset import TextImageDataset

# ───────────── config ──────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
CSV_PATH  = Path("data/metadata.csv")
IMG_DIR   = Path("data")
OUT_DIR   = Path("checkpoints")

RESOLUTION = 512
EPOCHS      = 5
BATCH_SIZE  = 1
LR          = 1e-5
# ───────────────────────────────────

# reasoning LM (LoRA)
lm_adapter, lm_tok = load_mistral_cpu(rank=8)
lm_adapter.to(DEVICE).to(DTYPE).eval()

# SDXL backbone
vae   = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
tok1  = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
tok2  = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
te1   = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder",   torch_dtype=DTYPE).to(DEVICE)
te2   = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE).to(DEVICE)
unet  = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet",    torch_dtype=DTYPE).to(DEVICE)
sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

# visual-LoRA
vis_cfg = LoraConfig(
    r=4, lora_alpha=8, lora_dropout=0.1,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    bias="none", task_type="UNET",
)
unet = get_peft_model(unet, vis_cfg)

# LM projector (fix scope)
class LanguageProjection(torch.nn.Module):
    def __init__(self, dim_lm: int, dim_img: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim_lm, dim_img)

    def forward(self, x):
        return self.proj(x)

lm_proj = LanguageProjection(dim_lm=1280, dim_img=4096).to(DEVICE).to(DTYPE)

# dynamic FiLM projector
class FiLMProj(torch.nn.Module):
    def __init__(self,
                 in_dim: int = 4096,
                 block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)):
        super().__init__()
        self.in_dim = in_dim
        uniq_channels = sorted(set(block_out_channels))
        self.proj = torch.nn.ModuleDict({
            str(C): torch.nn.Linear(in_dim, 2 * C)
            for C in uniq_channels
        })

    def forward(self, z, C):
        key = str(C)
        if key not in self.proj:
            self.proj[key] = torch.nn.Linear(self.in_dim, 2 * C).to(z.dtype).to(z.device)
        gammabeta = self.proj[key](z)
        return gammabeta.chunk(2, dim=-1)

film_proj = FiLMProj(in_dim=4096,
                     block_out_channels=tuple(unet.config.block_out_channels)
                    ).to(DEVICE).to(DTYPE)

# attach hooks
def add_film_hooks(u):
    for m in u.modules():
        if hasattr(m, "conv2"):
            def _hook(mod, inp, out):
                C        = out.shape[1]
                gamma, beta     = film_proj(u.z_r, C)
                return out * (1 + gamma[..., None, None]) + beta[..., None, None]
            m.register_forward_hook(_hook)
add_film_hooks(unet)

# dataset & optimiser
ds = TextImageDataset(CSV_PATH, IMG_DIR, resolution=RESOLUTION)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

opt = torch.optim.AdamW(
    list(lm_adapter.parameters()) +
    list(unet.parameters()) +
    list(film_proj.parameters()) +
    list(lm_proj.parameters()),
    lr=LR
)

# ───────────── training ─────────────
print("training …")
for ep in range(EPOCHS):
    for step, batch in enumerate(dl, 1):
        caption = batch["text"][0]

        # reasoning vector z_r
        lm_ids  = lm_tok(caption, return_tensors="pt").input_ids.to(DEVICE)
        z_r     = lm_adapter.model.model.embed_tokens(lm_ids).mean(dim=1).to(DTYPE)
        unet.z_r = z_r

        # CLIP token embeddings
        ids1 = tok1(caption, truncation=True, max_length=77,
                    padding="max_length", return_tensors="pt").input_ids.to(DEVICE)
        ids2 = tok2(caption, truncation=True, max_length=77,
                    padding="max_length", return_tensors="pt").input_ids.to(DEVICE)
        h1   = te1(ids1).last_hidden_state
        h2o  = te2(ids2);  h2 = h2o.last_hidden_state
        pooled = h2o.last_hidden_state[:, 0, :].to(DTYPE)
        token_embeds = torch.cat([h1, h2], dim=-1).to(DTYPE)

        # diffusion step
        pix   = batch["pixel_values"].to(DEVICE, dtype=DTYPE)
        lat   = vae.encode(pix).latent_dist.sample() * 0.18215
        noise = torch.randn_like(lat)
        t     = torch.randint(0, sched.config.num_train_timesteps, (1,), device=DEVICE)
        noisy = sched.add_noise(lat, noise, t)

        pred  = unet(sample=noisy, timestep=t,
                     encoder_hidden_states=token_embeds,
                     added_cond_kwargs={
                         "text_embeds": pooled,
                         "time_ids": torch.zeros((1,6), device=DEVICE, dtype=DTYPE),
                     }).sample

        # alignment loss (InfoNCE)
        z_r_proj   = lm_proj(pooled) 
        z_r_comb   = z_r + z_r_proj
        z_r_norm   = torch.nn.functional.normalize(z_r_comb, dim=-1)
        clip_norm  = torch.nn.functional.normalize(z_r_proj, dim=-1)

        sim = torch.matmul(z_r_norm, clip_norm.T) / 0.07
        labels = torch.arange(sim.size(0), device=DEVICE)
        infonce_loss = torch.nn.functional.cross_entropy(sim, labels)

        # reconstruction loss
        loss_mse = torch.nn.functional.mse_loss(pred, noise)
        lambda_mse = 1
        lambda_infonce = 0.1

        mse_loss_norm = loss_mse
        infonce_loss_norm = infonce_loss

        total_loss = lambda_mse * mse_loss_norm + lambda_infonce * infonce_loss_norm

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if step % 5 == 0:
            print(f"[E{ep+1}/{EPOCHS}] {step}/{len(dl)}  total_loss={total_loss.item():.4f}")

print("training finished")
OUT_DIR.mkdir(parents=True, exist_ok=True)
lm_adapter.save_pretrained(OUT_DIR / "lm_lora")
unet.save_pretrained(OUT_DIR / "vis_lora")
torch.save(film_proj.state_dict(), OUT_DIR / "film_proj.pt")
print("adapters saved to", OUT_DIR)
