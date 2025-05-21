### MakeupSimulator

Hi there,
This is a project for 'make-up simulator'😎😎
You could make up your face images in simulator!!


# Our Features

1. Make up transfer
2. Two types of Color control
  1) filter-based
  2) user-based
3. Enhance image equaity
4. Recommend make-up products in real
# 🧠 MakeupSimulator

A FastAPI-based backend project that uses [CodeFormer](https://github.com/sczhou/CodeFormer) to restore and enhance face images.  
This repository is designed for local testing and can be connected to a frontend application later.

---

## 🚀 Features

- Upload face images through an API
- Automatically enhance face quality using CodeFormer
- Download enhanced images
- Modular architecture for future integration (e.g., frontend, real-time webcam input)

---

## 💻 Requirements

- Python 3.8+
- pip or conda
- torch, torchvision
- FastAPI, Uvicorn
- basicsr, facexlib, gfpgan, xformers

Install all dependencies:
```bash
pip install -r requirements.txt



▶️ How to Run
1. Start the server:
uvicorn app:app --reload

✅ Automatic download (recommended)
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer

Refer: 
https://github.com/sczhou/CodeFormer
https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0