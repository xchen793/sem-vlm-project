
# SEM-VLM Project

This project implements a few-shot **semantic fine-tuning framework** that leverages a pretrained language model (Mistral-7B) and a diffusion-based vision generator (SDXL) to generate **SEM-like images** conditioned on natural language prompts.

---

## 📁 Project Structure

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
