import os
import re
import json
import base64
from io import BytesIO

import numpy as np
import cv2
import onnxruntime
import torch
import torch._dynamo
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from flask import Flask, request, render_template_string, send_file, url_for
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth
from transformers import AutoProcessor, AutoModelForImageTextToText
from cv2_rolling_ball import subtract_background_rolling_ball
import timm.models.fastvit as fv

# Fix for FastViT compatibility - some versions don't have the 'se' attribute
if not hasattr(fv.ReparamLargeKernelConv, 'se'):
    fv.ReparamLargeKernelConv.se = torch.nn.Identity()

# Disable PyTorch's dynamic compilation to avoid compatibility issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ---------------- Configuration and Paths ----------------
# Path to the trained EfficientNet model for osteoporosis classification
OSTEO_MODEL_PATH = '/mnt/nas/BrunoScholles/Gigasistemica/Models/efficientnet-b7_FULL_IMG_C1_C3.pth'
# MedGemma model identifier for medical image analysis
MEDGEMMA_MODEL_ID = 'google/medgemma-4b-it'
# Directory to cache the MedGemma model
CACHE_DIR = '/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma/cache'

# Atheroma detection pipeline model paths
ATHEROMA_CLASSIFIER_PATH      = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/model_epoch_4_val_loss_0.264005.pt'
ATHEROMA_DETECTION_MODEL_PATH = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/faster_end2end.onnx'
ATHEROMA_SEGMENTATION_PATH    = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/checkpoint.pth'

# Logo files for the web interface and PDF generation
LOGO_FILENAME     = 'giga_logo_app.png'
PDF_LOGO_FILENAME = 'giga_logo_pdf.png'

# ---------------- Device Setup ----------------
# Use GPU if available, otherwise fall back to CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch._dynamo.disable()

# ---------------- Osteoporosis Model Setup ----------------
# Extract the model name from the path (e.g., 'efficientnet-b7')
match = re.search(r'efficientnet-(b\d)', OSTEO_MODEL_PATH)
MODEL_NAME = 'efficientnet-' + match.group(1) if match else None
# Image resize dimensions for the model input
RESIZE = (449, 954)

# Load the pre-trained EfficientNet model and set it to evaluation mode
model_eff = EfficientNet.from_pretrained(MODEL_NAME, OSTEO_MODEL_PATH).to(device).eval()
# Freeze all parameters to prevent training during inference
for p in model_eff.parameters():
    p.requires_grad = False

# Standard ImageNet normalization for input preprocessing
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
# Inverse normalization for visualization
inv_normalize_transform = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
# Complete transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(),
    normalize_transform
])

# Mapping from model predictions to human-readable diagnoses
diag_sentences = {0: 'the patient is healthy', 1: 'the patient has osteoporosis'}

def rolling_ball_bg(gray_array, radius=180):
    """
    Apply rolling ball background subtraction to remove uneven illumination
    from grayscale images. This is commonly used in medical imaging preprocessing.
    """
    bg, _ = subtract_background_rolling_ball(
        gray_array, radius, light_background=False,
        use_paraboloid=True, do_presmooth=True
    )
    return bg

def compute_saliency(pil_img: Image.Image):
    """
    Generate saliency maps using gradient-based visualization (Grad-CAM).
    This helps understand which parts of the image the model focuses on
    when making osteoporosis predictions.
    """
    # Prepare input tensor and enable gradient computation
    inp = transform(pil_img).unsqueeze(0).to(device)
    inp.requires_grad = True
    
    # Forward pass and get predictions
    preds = model_eff(inp)
    _, idx = torch.max(preds, 1)
    
    # Backward pass to compute gradients with respect to the predicted class
    preds.backward(torch.zeros_like(preds).scatter_(1, idx.unsqueeze(1), 1.0))
    
    # Extract and process gradients to create saliency map
    grad = torch.abs(inp.grad[0].cpu())
    sal_map, _ = torch.max(grad, 0)
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    
    # Prepare original image for overlay
    img = inv_normalize_transform(inp[0].cpu())
    orig = np.clip(np.transpose(img.detach().numpy(), (1, 2, 0)), 0, 1)
    
    # Create heatmap overlay using matplotlib's hot colormap
    cmap = plt.cm.hot(sal_map.numpy())[..., :3]
    red = np.clip(cmap[:, :, 0] * 1.5, 0, 1)
    overlay = np.clip(orig + red[:, :, None], 0, 1)
    
    return orig, cmap, overlay, idx.item()

# ---------------- Atheroma Pipeline Functions ----------------
def load_classifier_model(path=ATHEROMA_CLASSIFIER_PATH):
    """
    Load the FastViT classifier model for atheroma detection.
    This model determines whether an image contains atheromas or not.
    """
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model

def predict_classifier_model(image_path, model, device, class_names):
    """
    Run inference on the atheroma classifier model.
    Returns the predicted class name (e.g., "Ateroma" or "Nao_Ateroma").
    """
    # Standard preprocessing for the classifier
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize_transform
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out, 1)
    return class_names[pred.item()]

def load_detection_model(path=ATHEROMA_DETECTION_MODEL_PATH):
    """
    Load the ONNX Faster R-CNN model for atheroma detection and localization.
    This model can identify specific regions where atheromas are present.
    """
    sess = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0].name
    return sess, inp

def predict_detection_model(session, input_name, image_path,
                            input_size=(1333, 800), conf_threshold=0.5):
    """
    Run the Faster R-CNN model to detect and localize atheromas in the image.
    Returns bounding boxes, confidence scores, and labels for detected regions.
    """
    def preprocess(img_path, size):
        """Preprocess image for the detection model"""
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        orig_shape = im.shape[:2]
        resized = cv2.resize(im, size)
        t = resized.transpose(2,0,1).astype(np.float32)/255.0
        mean = np.array([123.675,116.28,103.53])/255.0
        std  = np.array([58.395,57.12,57.375])/255.0
        t = (t - mean[:,None,None]) / std[:,None,None]
        return np.expand_dims(t,0).astype(np.float32), im, orig_shape

    def postproc(boxes, scores, labels, orig_shape, size):
        """Post-process detection results and filter by confidence threshold"""
        h_scale = orig_shape[0]/size[1]
        w_scale = orig_shape[1]/size[0]
        boxes *= np.array([w_scale, h_scale, w_scale, h_scale])
        fb, fs, fl = [], [], []
        for b, s, l in zip(boxes, scores, labels):
            if s >= conf_threshold:
                fb.append(b); fs.append(float(s)); fl.append(int(l))
        return fb, fs, fl

    tensor, _, orig_shape = preprocess(image_path, input_size)
    outs = session.run(None, {input_name: tensor})
    boxes_scores = outs[0][0]
    scores       = boxes_scores[:,4]
    boxes        = boxes_scores[:,:4]
    labels_out   = outs[1][0]
    return postproc(boxes, scores, labels_out, orig_shape, input_size)

def load_segmentation_model(path=ATHEROMA_SEGMENTATION_PATH):
    """
    Load the DC-UNet model for atheroma segmentation.
    This model creates pixel-level masks showing exactly where calcifications are located.
    """
    from utils import DC_UNet
    mdl = DC_UNet.DC_Unet(1).to(device)
    ckpt = torch.load(path, map_location=device)
    mdl.load_state_dict(ckpt['state_dict'])
    mdl.eval()
    return mdl

def predict_segmentation_model(image_path, model, device,
                               test_size=352, save_mask=False):
    """
    Generate segmentation masks for atheromas using a sliding window approach.
    The image is divided into cells and processed individually to handle large images.
    """
    img = Image.open(image_path).convert('L')
    w,h = img.size
    rows,cols = 2,3
    cw,ch = w//cols, h//rows
    mask = np.zeros((h,w),dtype=np.uint8)
    cells = [(1,0),(1,2)]  # Only process specific cells where atheromas are likely
    
    for r,c in cells:
        left,upper = c*cw, r*ch
        right = w if c==cols-1 else (c+1)*cw
        lower = h if r==rows-1 else (r+1)*ch
        cell = img.crop((left,upper,right,lower)).resize((test_size,test_size))
        t = transforms.ToTensor()(cell)
        t = transforms.Normalize([0.5],[0.5])(t).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(t)
        pred = pred.sigmoid().cpu().numpy().squeeze()
        pred = (pred - pred.min())/(pred.max()-pred.min()+1e-8)
        binm = (pred>=0.5).astype(np.uint8)
        resized = Image.fromarray(binm*255).resize((right-left, lower-upper), Image.NEAREST)
        mask[upper:lower, left:right] = np.array(resized)//255
    
    if save_mask:
        Image.fromarray(mask*255).save("mask.png")
    return mask

# Load all atheroma models at startup
atheroma_classifier_model   = load_classifier_model()
detection_session, detection_input_name = load_detection_model()
atheroma_segmentation_model = load_segmentation_model()
atheroma_class_names = ["Nao_Ateroma", "Ateroma"]

# ---------------- MedGemma Setup ----------------
# Load the multimodal vision-language model for medical image analysis
model_med = AutoModelForImageTextToText.from_pretrained(
    MEDGEMMA_MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16
).to(device)
proc_med = AutoProcessor.from_pretrained(
    MEDGEMMA_MODEL_ID, cache_dir=CACHE_DIR
)

# ---------------- Flask Web Application ----------------
app = Flask(__name__, static_folder='static', static_url_path='/static')
last_report = None  # Store the last generated PDF report

# HTML template for the web interface
HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>GigaXReport</title>
  <style>
    body {
      margin: 2rem 8rem;
      padding: 1rem;
      background: #121212;
      color: #e0e0e0;
    }
    textarea,input[type=file],select{
      width:100%; margin-bottom:1rem;
      background:#1e1e1e; color:#e0e0e0; border:1px solid #333; padding:.5rem;
    }
    button{padding:.5rem 1rem; background:#3b82f6; color:#fff;
      border:none; cursor:pointer; font-weight:bold;}
    button:hover{background:#2563eb;}
    img{max-width:100%; height:auto; margin-bottom:1rem;}
    pre{background:#1e1e1e; padding:1rem; overflow-x:auto; border:1px solid #333;}
    a.pdf-link{display:block; margin:1rem 0; font-weight:bold; color:#3b82f6;
      text-decoration:none;}
    a.pdf-link:hover{text-decoration:underline;}
    footer{font-size:.85rem; color:#999; border-top:1px solid #333;
      margin-top:2rem; padding-top:1rem;}
    a{color:#90cdf4;} a:hover{color:#63b3ed;}
    .form-group{margin-bottom:1rem;}
    label{display:block; margin-bottom:0.5rem; font-weight:bold;}
  </style>
</head>
<body>
  <img src="{{ url_for('static', filename=logo) }}" alt="Logo"
       style="max-width:200px; display:block; margin:0 auto 1rem;">
  <h1>GigaXReport</h1>
  <form method="post" enctype="multipart/form-data">
    <div class="form-group">
      <label>Language / Idioma:</label>
      <select name="language" required>
        <option value="english" {{ "selected" if language == "english" else "" }}>English</option>
        <option value="portuguese" {{ "selected" if language == "portuguese" else "" }}>Português</option>
      </select>
    </div>
    <div class="form-group">
      <label>Prompt:</label>
      <textarea name="prompt" rows="3" required></textarea>
    </div>
    <div class="form-group">
      <label>X-ray Image:</label>
      <input type="file" name="image" accept="image/*" required>
    </div>
    <button type="submit">Submit</button>
  </form>

  {% if plot_url %}
    <h2>Combined Pipeline Plot:</h2>
    <img src="{{ plot_url }}">
  {% endif %}
  {% if result %}
    <h2>Model Response:</h2>
    <pre>{{ result }}</pre>
  {% endif %}
  {% if pdf_ready %}
    <a href="/download" class="pdf-link">Download PDF Report</a>
  {% endif %}

  <footer>
    <p>
      This application integrates advanced AI models for biomedical imaging: <strong>MedGemma 4B</strong>, a multimodal vision-language model by <strong>Google Research</strong> and <strong>DeepMind</strong>, fine-tuned for radiological reporting; <strong>EfficientNetB7</strong> for osteoporosis image classification; and a dedicated atheroma pipeline combining <strong>FastViT</strong> for classification, <strong>Faster R-CNN (ONNX)</strong> for plaque detection, and <strong>DC-UNet</strong> for calcification segmentation. For more on MedGemma, visit <a href="https://huggingface.co/google/medgemma-4b-it" target="_blank">huggingface.co/google/medgemma-4b-it</a>.
    </p>
  </footer>
</body>
</html>
'''

@app.route('/', methods=['GET','POST'])
def index():
    """
    Main route that handles both GET (display form) and POST (process image) requests.
    Integrates all three AI pipelines: osteoporosis detection, atheroma analysis, and MedGemma reporting.
    """
    global last_report
    result = None
    plot_url = None
    pdf_ready = False
    language = 'english'  # Default language

    if request.method == 'POST':
        language = request.form.get('language', 'english')
        prompt = request.form.get('prompt','').strip()
        f      = request.files.get('image')
        if prompt and f:
            img_bytes = f.read()

            # Step 1: Osteoporosis Analysis with Grad-CAM
            # Convert to grayscale and apply background correction
            pil_gray = Image.open(BytesIO(img_bytes)).convert('L')
            bg = rolling_ball_bg(np.array(pil_gray))
            pil_rgb = Image.fromarray(bg.astype('uint8'), 'L').convert('RGB')
            
            # Generate saliency maps and get osteoporosis prediction
            orig, sal, ovl, idx = compute_saliency(pil_rgb)
            osteoporosis_diag = diag_sentences[idx]

            # Step 2: Atheroma Detection Pipeline
            # Save uploaded image temporarily for processing
            tmp_path = '/tmp/upload.png'
            with open(tmp_path,'wb') as tmp:
                tmp.write(img_bytes)

            # Run atheroma classifier to determine if atheromas are present
            raw_pred_ath = predict_classifier_model(
                tmp_path, atheroma_classifier_model, device, atheroma_class_names
            )
            has_ath = (raw_pred_ath.lower() == "ateroma")
            pred_ath = (
                "positive for atheromas / calcifications"
                if has_ath else
                "negative for atheromas / calcifications"
            )

            # If atheromas detected, run detection and segmentation
            if has_ath:
                # Detect bounding boxes around atheromas
                boxes, scores, labels_ = predict_detection_model(
                    detection_session, detection_input_name, tmp_path
                )
                # Generate segmentation mask
                mask = predict_segmentation_model(
                    tmp_path, atheroma_segmentation_model, device
                )
                
                # Visualize results on the image
                cv_img = cv2.imread(tmp_path)
                # Draw bounding boxes
                for b,s,lbl in zip(boxes, scores, labels_):
                    x0,y0,x1,y1 = map(int, b)
                    cv2.rectangle(cv_img, (x0,y0), (x1,y1), (0,255,0), 2)
                    cv2.putText(cv_img, f"{lbl}:{s:.2f}",
                                (x0, max(0,y0-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # Overlay segmentation mask in red
                inds = np.where(mask>0)
                cv_img[inds] = (0,0,255)
                detection_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                # If no atheromas, just use the original image
                detection_img = np.array(
                    Image.open(BytesIO(img_bytes)).convert('RGB')
                )

            # Step 3: Create Combined Visualization
            # Generate a compact plot showing all analysis results
            fig = plt.figure(figsize=(12, 6))
            gs  = gridspec.GridSpec(
                2, 2,
                height_ratios=[1, 1],
                hspace=0      # Minimal vertical spacing
            )
            ax0 = fig.add_subplot(gs[0, 0])  # Original image
            ax1 = fig.add_subplot(gs[0, 1])  # Grad-CAM overlay
            ax2 = fig.add_subplot(gs[1, :])  # Atheroma detection/segmentation

            # Top row: original and overlay without titles
            ax0.imshow(orig)
            ax0.axis('off')

            ax1.imshow(ovl)
            ax1.axis('off')

            # Bottom row: atheroma detection results
            ax2.imshow(detection_img)
            ax2.axis('off')
            if not has_ath:
                ax2.text(
                    0.5, 0.5, 'NO CACS DETECTED',
                    transform=ax2.transAxes,
                    color='red', fontsize=20,
                    ha='center', va='center'
                )

            # Save combined visualization
            fig.subplots_adjust(hspace=0)
            buf = BytesIO()
            fig.savefig(
                buf, format='png',
                bbox_inches='tight', pad_inches=0,
                dpi=300
            )
            plt.close(fig)
            buf.seek(0)
            combined_data = buf.getvalue()
            plot_url = "data:image/png;base64," + base64.b64encode(combined_data).decode()

            # Step 4: MedGemma Analysis and Report Generation
            # Prepare the combined image for MedGemma
            combined_pil = Image.open(BytesIO(combined_data)).convert('RGB')
            
            # Define system text based on selected language
            if language == 'portuguese':
                sys_txt = ("Você é um radiologista especialista. "
                          "O diagnóstico do modelo EfficientNet para osteoporose é exatamente "
                          f"\"{osteoporosis_diag}\", e o diagnóstico do pipeline de ateroma é exatamente "
                          f"\"{pred_ath}\". Essas saídas dos modelos devem aparecer literalmente no seu relatório nas "
                          "seções \"Saúde Óssea\" e \"Diagnóstico de Ateroma\". Você deve explicar por que essas "
                          "classificações podem ter ocorrido, baseado em indicadores radiológicos comuns. "
                          "Linha superior da imagem: esquerda = original, direita = sobreposição grad-cam apenas para osteoporose. "
                          "Linha inferior: segmentação/detecção apenas para ateromas (se disponível). "
                          "Se a classificação de ateroma for positiva mas nenhuma detecção/segmentação foi possível, "
                          "deixe claro que o modelo classificou como tal mas não foi capaz de localizar nenhuma placa. "
                          "Sua resposta deve cobrir: Impressão Geral, Estrutura Óssea, Saúde Óssea, "
                          "Diagnóstico de Ateroma, Detecção/Segmentação de Ateroma e Resumo. Seja detalhado, medicamente correto, "
                          "e descreva a imagem de entrada. Não se esqueça de descrever a detecção/segmentação na imagem (se houver)."
                          "Não coloque Paciente, Data, Exame ou similar na sua resposta."
                          "Não fale sobre os dentes na sua resposta. Além disso, você deve obedecer ao que o usuário pede."
                          "Se eles pedirem um relatório em um idioma específico ou em um formato específico, você deve responder nesse idioma ou formato."
                )
            else:  # English (default)
                sys_txt = ("You are an expert radiologist. "
                          "The diagnosis from the EfficientNet osteoporosis model is exactly "
                          f"\"{osteoporosis_diag}\", and the diagnosis from the atheroma pipeline is exactly "
                          f"\"{pred_ath}\". These model outputs must appear verbatim in your report under "
                          "the sections \"Bone Health\" and \"Atheroma Diagnosis\". You must explain why these "
                          "classifications might have occurred, based on common radiological indicators. "
                          "Top row of the image: left = original, right = grad-cam overlay just for osteoporosis. "
                          "Bottom row: segmentation/detection just for atheromas (if available). "
                          "If the atheroma classification is positive but no detection/segmentation was possible, "
                          "make clear that the model classified it as such but was unable to localize any plaque. "
                          "Your response should cover: General Impression, Bone Structure, Bone Health, "
                          "Atheroma Diagnosis, Atheroma Detection/Segmentation and Summary. Be detailed, medically sound, "
                          "and describe the input image. Don't forget to describe the detection/segmentation in the image (if it has)."
                          "Do not put Patient, Date, Exam or similar in your response."
                          "Do not talk about the teeth in your response. Also, you must obay what the user asks for."
                          "If they ask for a report in a specific language or in a specific format, you must respond in that language or format."
                )
            
            # Prepare messages for MedGemma
            msgs = [
                {"role":"system", "content":[{"type":"text","text":sys_txt}]},
                {"role":"user",   "content":[
                    {"type":"text","text":prompt},
                    {"type":"image","image":combined_pil}
                ]}
            ]
            
            # Run MedGemma inference
            inputs = proc_med.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt'
            ).to(device, torch.bfloat16)
            ilen = inputs['input_ids'].shape[-1]
            
            with torch.inference_mode():
                gids = model_med.generate(**inputs, max_new_tokens=2000, do_sample=True, temperature=0.5, top_p=0.9, top_k=50)
            result = proc_med.decode(gids[0][ilen:], skip_special_tokens=True)

            # Step 5: PDF Report Generation
            # Create a professional PDF report with the results
            buf_pdf = BytesIO()
            c = canvas.Canvas(buf_pdf, pagesize=letter)
            w,h = letter
            
            # Add company logo to PDF
            logo_p = os.path.join(app.root_path, 'static', PDF_LOGO_FILENAME)
            disp_h = 0
            if os.path.exists(logo_p):
                img_l = Image.open(logo_p)
                lw,lh = img_l.size
                dw=150; dh=int(dw*(lh/lw))
                c.drawInlineImage(img_l,50,h-dh-20,width=dw,height=dh)
                disp_h = dh
            
            # Add the combined analysis image
            pw,ph = combined_pil.size
            pw_pdf = 450
            ph_pdf = int(pw_pdf * (ph/pw))
            y0 = h - (disp_h+40) - ph_pdf
            c.drawInlineImage(combined_pil,50,y0,width=pw_pdf,height=ph_pdf)
            
            # Register fonts for text formatting
            try:
                pdfmetrics.registerFont(TTFont('Helvetica-Bold','Helvetica-Bold.ttf'))
            except:
                pass
            
            # Process and format the MedGemma response for PDF
            clean = re.sub(r'(?<!\*)\*(?!\*)','-', result)
            paras = clean.split('\n')
            
            def segs(line):
                """Split text into segments for bold formatting"""
                parts = re.split(r'(\*\*[^*]+\*\*)', line)
                out = []
                for p in parts:
                    if p.startswith('**') and p.endswith('**'):
                        out.append((p[2:-2], True))
                    else:
                        out.append((p, False))
                return out
            
            # Add formatted text to PDF
            txt = c.beginText(50, y0-20)
            txt.setLeading(12)
            for para in paras:
                if not para.strip():
                    txt.textLine(''); continue
                curr_w = 0
                for seg, bold in segs(para):
                    font = 'Helvetica-Bold' if bold else 'Helvetica'
                    for tok in re.split(r'(\s+)', seg):
                        if not tok: continue
                        tw = stringWidth(tok, font, 8)
                        if curr_w + tw > 500 and tok.strip():
                            txt.textLine(''); curr_w = 0
                        txt.setFont(font, 8)
                        txt.textOut(tok)
                        curr_w += tw
                txt.textLine('')
            c.drawText(txt)
            c.showPage(); c.save()
            buf_pdf.seek(0)
            last_report = buf_pdf.read()
            pdf_ready = True

    return render_template_string(
        HTML,
        logo=LOGO_FILENAME,
        plot_url=plot_url,
        result=result,
        pdf_ready=pdf_ready,
        language=language
    )

@app.route('/download')
def download():
    """
    Route to download the generated PDF report.
    Returns the last generated report or a 404 if none exists.
    """
    if last_report:
        return send_file(
            BytesIO(last_report),
            as_attachment=True,
            download_name='GigaXReport.pdf',
            mimetype='application/pdf'
        )
    return ('No report', 404)

if __name__ == '__main__':
    # Start the Flask development server
    app.run(host='0.0.0.0', port=5000)