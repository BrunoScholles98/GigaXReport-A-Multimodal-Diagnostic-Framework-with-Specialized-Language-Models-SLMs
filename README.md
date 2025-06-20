# GigaXReport: A Multimodal Diagnostic Framework with Specialized Language Models (SLMs)

**This project is a key component of GigaSistêmica, a collaborative initiative between GigaCandanga and the University of Brasília. GigaSistêmica aims to revolutionize diagnostic and predictive capabilities for systemic diseases through the integration of AI and medical imaging technologies.**

## Overview

GigaXReport is an advanced diagnostic framework that leverages Specialized Language Models (SLMs) to provide comprehensive, multimodal analysis of medical images. By combining state-of-the-art deep learning models for image classification, segmentation, and detection with powerful language models tailored for medical applications, GigaXReport delivers detailed, explainable, and clinically relevant reports.

This framework is designed to support healthcare professionals in diagnosing and predicting systemic diseases, with a particular focus on bone health and atheroma detection. The integration of SLMs enables the system to generate expert-level textual descriptions and insights based on both image data and AI-driven predictions.

## Key Features

- **Multimodal Analysis:** Integrates image-based deep learning models (e.g., EfficientNet, UNet) with specialized language models for text generation and explanation.
- **SLM-Powered Reporting:** Utilizes models like MedGemma to generate detailed, context-aware medical reports from image and classification results.
- **Atheroma and Osteoporosis Pipelines:** Includes dedicated modules for atheroma detection/classification/segmentation and osteoporosis diagnosis.
- **Web Application:** User-friendly interface for uploading images, viewing results, and downloading PDF reports.
- **Collaborative and Extensible:** Built as part of the GigaSistêmica initiative, fostering collaboration between research groups and clinical partners.

## Project Structure

- `run_app.py`: Main web application for interactive diagnosis and report generation.
- `run_gemma_model.py`: Script for running the MedGemma SLM on medical images and generating textual reports.
- `giga_segmentation/`: Contains segmentation models and utilities, including atheroma segmentation.
- `static/`: Static assets such as logos for the web app and PDF reports.
- `model_download.py`: Utility for downloading and caching required language models.

## Getting Started

1. **Clone the repository and install dependencies** (see `giga_env.yml` in the parent directory for environment setup).
2. **Download required models** using the provided scripts.
3. **Run the web application**:
   ```bash
   python run_app.py
   ```
4. **Access the interface** via your browser to upload images and generate reports.

## Applications

- Automated bone health assessment (osteoporosis detection)
- Atheroma detection, classification, and segmentation
- Generation of expert-level, explainable medical reports
- Research and development in AI-driven medical diagnostics

---

## 4. Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at [brunoscholles98@gmail.com](mailto:brunoscholles98@gmail.com). Additionally, you can contact me via WhatsApp at +351 913 686 499.

## 5. Thanks

Special thanks to my advisors Mylene C. Q. Farias, André Ferreira Leite, and Nilce Santos de Melo. Also, a special thanks to my colleague Matheus Virgílio.

---
