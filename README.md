## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Models Summary](#models-summary)
- [Getting Started](#getting-started)
- [Applications](#applications)
- [Model Access](#model-access)
- [Contact](#contact)
- [Thanks](#thanks)

---

# GigaXReport: A Multimodal Diagnostic Framework with Specialized Language Models (SLMs)

**This project is a key component of GigaSistêmica, a collaborative initiative between GigaCandanga and the University of Brasília. GigaSistêmica aims to revolutionize diagnostic and predictive capabilities for systemic diseases through the integration of AI and medical imaging technologies.**

🔗 **Main Project Repository**: [GigaSistêmica – Advancing Systemic Health Diagnostics through AI](https://github.com/BrunoScholles98/GigaSistemica-Advancing-Systemic-Health-Diagnostics-through-AI)

---

<a name="overview"></a>
## Overview

GigaXReport is an advanced diagnostic framework that leverages Specialized Language Models (SLMs) to provide comprehensive, multimodal analysis of medical images. By combining state-of-the-art deep learning models for image classification, segmentation, and detection with powerful language models tailored for medical applications, GigaXReport delivers detailed, explainable, and clinically relevant reports.

This framework is designed to support healthcare professionals in diagnosing and predicting systemic diseases, with a particular focus on bone health and atheroma detection. The integration of SLMs enables the system to generate expert-level textual descriptions and insights based on both image data and AI-driven predictions.

![](https://raw.githubusercontent.com/BrunoScholles98/GigaXReport-A-Multimodal-Diagnostic-Framework-with-Specialized-Language-Models-SLMs/refs/heads/main/static/MainPage_Example.png)

<a name="key-features"></a>
## Key Features

- **Multimodal Analysis:** Integrates image-based deep learning models (e.g., EfficientNet, UNet) with specialized language models for text generation and explanation.
- **SLM-Powered Reporting:** Utilizes models like MedGemma to generate detailed, context-aware medical reports from image and classification results.
- **Atheroma and Osteoporosis Pipelines:** Includes dedicated modules for atheroma detection/classification/segmentation and osteoporosis diagnosis.
- **Web Application:** User-friendly interface for uploading images, viewing results, and downloading PDF reports.
- **Collaborative and Extensible:** Built as part of the GigaSistêmica initiative, fostering collaboration between research groups and clinical partners.

<a name="models-summary"></a>
## Models Summary

GigaXReport integrates a comprehensive suite of state-of-the-art AI models to provide multimodal medical image analysis:

<a name="getting-started"></a>
## Getting Started

### **Primary Usage: Web Application**

The main way to interact with GigaXReport is through the **`run_app.py`** web application, which provides a comprehensive interface for medical image analysis:

1. **Clone the repository and install dependencies** (see `giga_env.yml` in the parent directory for environment setup).
2. **Model Access**: The trained models used in this framework are available by request. Please contact the development team to obtain access to the model files.
3. **Launch the Web Application**:
   ```bash
   python run_app.py
   ```
4. **Access the Interface**: Open your browser and navigate to the provided URL to access the GigaXReport web interface.
5. **Upload and Analyze**:
   - Upload your medical X-ray image
   - Provide a custom prompt for the analysis
   - View the comprehensive results including:
     - Osteoporosis classification with Grad-CAM visualization
     - Atheroma detection and segmentation
     - AI-generated medical report
   - Download the complete analysis as a PDF report

### **Alternative Usage: Command Line**

For advanced users, individual model components can be accessed through dedicated scripts:
- Use `run_gemma_model.py` for direct MedGemma model inference
- Access segmentation models through the `utils/` directory

<a name="applications"></a>
## Applications

- Automated bone health assessment (osteoporosis detection)
- Atheroma detection, classification, and segmentation
- Generation of expert-level, explainable medical reports
- Research and development in AI-driven medical diagnostics

---

<a name="model-access"></a>
## Model Access

The trained AI models used in this framework (e.g., MedGemma, EfficientNet, FastViT, DC-UNet) are **available upon request**. If you are a researcher, clinician, or developer interested in accessing the model weights, please contact the development team using the information below.

---

<a name="contact"></a>
## Contact

Please feel free to reach out with any comments, questions, reports, or suggestions via email at [brunoscholles98@gmail.com](mailto:brunoscholles98@gmail.com) or [matheusvirgiliomcp@gmail.com](mailto:matheusvirgiliomcp@gmail.com).  Additionally, you can contact us via WhatsApp at +351 913 686 499 or +55 61 98275-5573.

---

<a name="thanks"></a>
## Thanks

Special thanks to my advisors Mylene C. Q. Farias, André Ferreira Leite, and Nilce Santos de Melo.

---
