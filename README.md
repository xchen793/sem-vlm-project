# SEM-VLM Project

This repository implements a few-shot **semantic fine-tuning framework** for generating scanning electron microscopy (SEM) images from textual prompts. It combines:

- A pretrained **language model** (Mistral-7B-Instruct) for reasoning and prompt encoding
- A pretrained **UNet diffusion model** (based on SDXL) for image generation
- A **FiLM projection bridge** to modulate the UNet based on language representations
- **LoRA** adapters to perform parameter-efficient fine-tuning on limited data

---


## Project Structure

sem-vlm-project/ 

├── train.py # Fine-tunes the UNet backbone using few-shot SEM data

├── infer.py # Generates SEM images from text prompts using the fine-tuned model

├── lm_adapter.py # Loads the language model and processes textual input into FiLM parameters

├── dataset.py # Dataset class for loading paired text-image training data

├── data/

│ ├── metadata.csv # CSV file mapping text prompts to image filenames

│ └── images/ # Directory containing SEM images

├── environment.yml # Conda environment specification

└── README.md # This documentation


---

## Example Use Case

**Prompt:**  "Please generate a SEM image of the PPy/PVA double-network hydrogel with aligned fibrous structure and high porosity."
**Generated Output:**  
- An SEM-style synthetic image reflecting fibrous alignment and microstructural details
- Realism improved by fine-tuning with few-shot real SEM examples

## Environment Setup

### Clone the Repository

```bash
git clone https://github.com/xchen793/sem-vlm-project.git
cd sem-vlm-project
conda env create -f environment.yml
conda activate sem_vlm_env
```
## Training: Fine-Tune the Generator
```bash
python train.py
```
## Inference: Generate SEM Image from Prompt

```bash
python infer.py --prompt "Generate a SEM image of nanoporous hydrogel with aligned fiber bundles"
```
