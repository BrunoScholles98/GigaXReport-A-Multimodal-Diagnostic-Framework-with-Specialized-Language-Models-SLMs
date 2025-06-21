# GigaXReport: A Multimodal Diagnostic Framework with Specialized Language Models (SLMs)

**This project is a key component of GigaSistêmica, a collaborative initiative between GigaCandanga and the University of Brasília. GigaSistêmica aims to revolutionize diagnostic and predictive capabilities for systemic diseases through the integration of AI and medical imaging technologies.**

## Overview

GigaXReport is an advanced diagnostic framework that leverages Specialized Language Models (SLMs) to provide comprehensive, multimodal analysis of medical images. By combining state-of-the-art deep learning models for image classification, segmentation, and detection with powerful language models tailored for medical applications, GigaXReport delivers detailed, explainable, and clinically relevant reports.

This framework is designed to support healthcare professionals in diagnosing and predicting systemic diseases, with a particular focus on bone health and atheroma detection. The integration of SLMs enables the system to generate expert-level textual descriptions and insights based on both image data and AI-driven predictions.

![](https://raw.githubusercontent.com/BrunoScholles98/GigaXReport-A-Multimodal-Diagnostic-Framework-with-Specialized-Language-Models-SLMs/refs/heads/main/static/MainPage_Example.png)

## Key Features

- **Multimodal Analysis:** Integrates image-based deep learning models (e.g., EfficientNet, UNet) with specialized language models for text generation and explanation.
- **SLM-Powered Reporting:** Utilizes models like MedGemma to generate detailed, context-aware medical reports from image and classification results.
- **Atheroma and Osteoporosis Pipelines:** Includes dedicated modules for atheroma detection/classification/segmentation and osteoporosis diagnosis.
- **Web Application:** User-friendly interface for uploading images, viewing results, and downloading PDF reports.
- **Collaborative and Extensible:** Built as part of the GigaSistêmica initiative, fostering collaboration between research groups and clinical partners.

## Models Summary

GigaXReport integrates a comprehensive suite of state-of-the-art AI models to provide multimodal medical image analysis:

### **Language Models**
- **MedGemma 4B-IT**: A multimodal vision-language model developed by Google Research and DeepMind, specifically fine-tuned for medical applications. This model generates expert-level radiological reports by combining image analysis with natural language understanding, providing detailed clinical insights and explanations.

### **Osteoporosis Detection Pipeline**
- **EfficientNet-B7**: A highly efficient convolutional neural network architecture for bone health classification. This model analyzes X-ray images to detect signs of osteoporosis, providing binary classification (healthy vs. osteoporotic) with high accuracy.
- **Grad-CAM Visualization**: Implements gradient-weighted class activation mapping to provide explainable AI insights, highlighting the regions of the image that most influenced the osteoporosis classification decision.

### **Atheroma Detection Pipeline**
- **FastViT Classifier**: A fast vision transformer model for atheroma classification, determining whether calcifications/atheromas are present in the image.
- **Faster R-CNN (ONNX)**: An object detection model for localizing and detecting individual atheroma plaques within the image, providing bounding box coordinates and confidence scores.
- **DC-UNet**: A deep convolutional U-Net architecture with dense connections for precise atheroma segmentation. This model creates pixel-level masks identifying calcified regions, enabling detailed analysis of plaque distribution and morphology.

### **Image Processing**
- **Rolling Ball Background Subtraction**: Advanced image preprocessing technique that enhances image quality by removing background noise and improving contrast for better model performance.

### **Integration Framework**
- **Flask Web Application**: Provides a user-friendly interface for uploading medical images, viewing AI-generated analysis results, and downloading comprehensive PDF reports that combine all model outputs into a single, clinically relevant document.

## Project Structure

- `run_app.py`: **Main web application** for interactive diagnosis and report generation - the primary interface for users to interact with the GigaXReport system.
- `run_gemma_model.py`: Script for running the MedGemma SLM on medical images and generating textual reports.
- `utils/DC_UNet.py`: Implementation of the DC-UNet architecture for atheroma segmentation.
- `static/`: Static assets such as logos for the web app and PDF reports.
- `model_download.py`: Utility for downloading and caching required language models.

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
