# 🌍 EuroSAT CNN Land Cover Classification

> 🚀 University AI for Space Project — Scored **90%**

This project applies a **Convolutional Neural Network (CNN)** model based on **ResNet-18** to classify satellite images from the **EuroSAT dataset** into various land cover categories.

---

## 📂 Project Files

- `eurosat_cnn_classification.ipynb` — Jupyter Notebook for model training & evaluation
- `Land_Cover_CNN_Report.pdf` — Final project report (graded 90%)
- `Land_Cover_CNN_Presentation.pptx` — Project presentation
- `demo_video.mp4` — Short video demo of the model
- `README.md` — Project overview and setup instructions

---

## 🛰️ Problem Statement

Classify land cover types from RGB satellite imagery such as:
- Forest
- Sea/Lake
- Residential
- River
- Annual Crop

Using deep learning methods to support Earth observation and remote sensing automation.

---

## 📊 Results

- ✅ **Model Used:** Pretrained ResNet-18
- 📈 **Test Accuracy:** ~94.69%
- 🔎 **Validation Accuracy:** ~93.54%
- 📋 **Precision/Recall:** >90% for major classes

Visuals include confusion matrix, classification report, and accuracy/loss curves.

---

## ⚙️ Tech Stack

- Python 3.8+
- PyTorch
- Torchvision
- Scikit-learn
- Seaborn
- Matplotlib

---

## 🚀 Setup Instructions

> 💡 Recommended: Use Google Colab or a local Python environment

Install required libraries:
```bash
pip install torch torchvision scikit-learn seaborn matplotlib
```

To train and evaluate:
1. Open `eurosat_cnn_classification.ipynb`
2. Run cells sequentially to:
   - Load EuroSAT RGB dataset
   - Fine-tune ResNet-18
   - Evaluate model
   - Visualize predictions

---

## 👨‍💻 Contributors

- **Satyawan Singh**
- **Shreyash Wankhade**

---

## 🏅 Recognition

This project was submitted as part of the **AI for Space** module and received **90% marks** for its performance, clarity, and practical application of AI in remote sensing.

---

## 📎 Acknowledgements

- Dataset: [EuroSAT](https://github.com/phelber/eurosat) by Patrick Helber
- Frameworks: PyTorch, Torchvision
- University of Leicester — MSc AI for Business Intelligence

---

## 📽️ Demo

Watch the short video demonstration: `demo_video.mp4`

---

## 🌐 License

Open for academic use and showcasing only.
Commercial or large-scale deployments should cite original dataset and model authors.
