import os
# Disable TorchDynamo before importing PyTorch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# --- IMPORTS ---
import re
import tempfile
from io import BytesIO
import logging
import numpy as np
import cv2
import onnxruntime
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText
from cv2_rolling_ball import subtract_background_rolling_ball
import timm.models.fastvit as fv
from functools import lru_cache
from utils.pdf_generator import generate_pdf_report
from utils import DC_UNet

# Fix for FastViT compatibility
if not hasattr(fv.ReparamLargeKernelConv, 'se'):
    setattr(fv.ReparamLargeKernelConv, 'se', torch.nn.Identity())

# --- CONFIGURATION AND PATHS ---
# Input/Output
IMAGE_PATH          = "/mnt/nas/BrunoScholles/Gigasistemica/test_imgs/ATHUB2018-9.jpg"
PDF_OUTPUT_DIR      = '/mnt/nas/BrunoScholles/Gigasistemica/test_imgs'
PDF_LOGO_FILENAME   = 'giga_logo_pdf.png'
STATIC_DIR          = 'static'
# Prompt Configuration
USER_PROMPT_ENGLISH     = "Based on the analysis results, please provide a detailed radiological report."
USER_PROMPT_PORTUGUESE  = "Com base nos resultados da análise, forneça um relatório radiológico detalhado."
LANGUAGE                = 'english' # 'english' or 'portuguese'
# Models
OSTEO_MODEL_PATH      = '/mnt/nas/BrunoScholles/Gigasistemica/Models/efficientnet-b7_FULL_IMG_C1_C3.pth'
MEDGEMMA_MODEL_ID     = '/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma/cache/models--google--medgemma-4b-it/snapshots/698f7911b8e0569ff4ebac5d5552f02a9553063c'
ADAPTER_MODEL_PATH    = '/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaTrained'
CACHE_DIR             = '/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma/cache'
ATHEROMA_CLASSIFIER_PATH      = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/model_epoch_4_val_loss_0.264005.pt'
ATHEROMA_DETECTION_MODEL_PATH = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/faster_end2end.onnx'
ATHEROMA_SEGMENTATION_PATH    = '/mnt/ssd/brunoscholles/GigaSistemica/Models/Atheroma/checkpoint.pth'

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- DEVICE SETUP ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
try:
    torch._dynamo.disable()
except (AttributeError, ImportError):
    pass

# --- MODEL CONFIG ---
match = re.search(r'efficientnet-(b\d)', OSTEO_MODEL_PATH)
MODEL_NAME = 'efficientnet-' + match.group(1) if match else None
RESIZE = (449, 954)
diag_sentences = {0: 'the patient is healthy', 1: 'the patient has osteoporosis'}
atheroma_class_names = ["Nao_Ateroma", "Ateroma"]

# --- IMAGE PREPROCESSING TRANSFORMS ---
normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
inv_normalize_transform = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
transform = transforms.Compose([transforms.Resize(RESIZE), transforms.ToTensor(), normalize_transform])

# --- LAZY MODEL LOADERS ---
@lru_cache(maxsize=1)
def get_osteoporosis_model():
    """Load EfficientNet model on first use."""
    logger.info("Loading EfficientNet model for osteoporosis analysis ...")
    if MODEL_NAME is None:
        raise RuntimeError("Could not infer EfficientNet model name from OSTEO_MODEL_PATH")
    mdl = EfficientNet.from_pretrained(MODEL_NAME, OSTEO_MODEL_PATH).to(device).eval()
    for p in mdl.parameters():
        p.requires_grad = False
    logger.info("EfficientNet model loaded and ready.")
    return mdl

@lru_cache(maxsize=1)
def get_med_model():
    """Load MedGemma model on first use."""
    logger.info("Loading MedGemma vision-language model ...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}
    )
    logger.info(f"Loading LoRA adapter from {ADAPTER_MODEL_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
    logger.info("Moving model to device ...")
    model = model.to(device)
    logger.info("MedGemma model and adapter loaded.")
    return model

@lru_cache(maxsize=1)
def get_med_processor():
    """Load MedGemma processor on first use."""
    logger.info("Loading MedGemma processor ...")
    return AutoProcessor.from_pretrained(MEDGEMMA_MODEL_ID, cache_dir=CACHE_DIR)

@lru_cache(maxsize=1)
def get_atheroma_classifier():
    logger.info("Loading FastViT classifier for atheroma detection ...")
    return load_classifier_model()

@lru_cache(maxsize=1)
def get_detection_models():
    logger.info("Loading Faster R-CNN ONNX session for atheroma localization ...")
    return load_detection_model()

@lru_cache(maxsize=1)
def get_segmentation_model():
    logger.info("Loading DC-UNet model for atheroma segmentation ...")
    return load_segmentation_model()

# --- PIPELINE FUNCTIONS (COPIED FROM RUN_APP.PY) ---
def rolling_ball_bg(gray_array, radius=180):
    bg, _ = subtract_background_rolling_ball(gray_array, radius, light_background=False, use_paraboloid=True, do_presmooth=True)
    return bg

def compute_saliency(pil_img: Image.Image):
    inp = transform(pil_img).unsqueeze(0).to(device)  # type: ignore[attr-defined]
    inp.requires_grad = True
    model_eff = get_osteoporosis_model()
    preds = model_eff(inp)
    _, idx = torch.max(preds, 1)
    preds.backward(torch.zeros_like(preds).scatter_(1, idx.unsqueeze(1), 1.0))
    grad = torch.abs(inp.grad[0].cpu())
    sal_map, _ = torch.max(grad, 0)
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    img = inv_normalize_transform(inp[0].cpu())
    orig = np.clip(np.transpose(img.detach().numpy(), (1, 2, 0)), 0, 1)
    cmap = plt.cm.hot(sal_map.numpy())[..., :3]
    red = np.clip(cmap[:, :, 0] * 1.5, 0, 1)
    overlay = np.clip(orig + red[:, :, None], 0, 1)
    return orig, cmap, overlay, idx.item()

def load_classifier_model(path=ATHEROMA_CLASSIFIER_PATH):
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model

def predict_classifier_model(image_path, model, device, class_names):
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize_transform])
    img = Image.open(image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(device)  # type: ignore[attr-defined]
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out, 1)
    return class_names[pred.item()]

def load_detection_model(path=ATHEROMA_DETECTION_MODEL_PATH):
    available_providers = onnxruntime.get_available_providers()
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(path, providers=providers)
    inp = sess.get_inputs()[0].name
    return sess, inp

def predict_detection_model(session, input_name, image_path, input_size=(1333, 800), conf_threshold=0.5):
    def preprocess(img_path, size):
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
    boxes_scores = outs[0][0]; scores = boxes_scores[:,4]; boxes = boxes_scores[:,:4]; labels_out = outs[1][0]
    return postproc(boxes, scores, labels_out, orig_shape, input_size)

def load_segmentation_model(path=ATHEROMA_SEGMENTATION_PATH):
    mdl = DC_UNet.DC_Unet(1).to(device)
    ckpt = torch.load(path, map_location=device)
    mdl.load_state_dict(ckpt['state_dict'])
    mdl.eval()
    return mdl

def predict_segmentation_model(image_path, model, device, test_size=352):
    img = Image.open(image_path).convert('L'); w,h = img.size; rows,cols = 2,3; cw,ch = w//cols, h//rows
    mask = np.zeros((h,w),dtype=np.uint8); cells = [(1,0),(1,2)]
    for r,c in cells:
        left,upper = c*cw, r*ch; right = w if c==cols-1 else (c+1)*cw; lower = h if r==rows-1 else (r+1)*ch
        cell = img.crop((left,upper,right,lower)).resize((test_size,test_size))
        t = transforms.ToTensor()(cell); t = transforms.Normalize([0.5],[0.5])(t).unsqueeze(0).to(device)
        with torch.no_grad(): pred = model(t)
        pred = pred.sigmoid().cpu().numpy().squeeze(); pred = (pred - pred.min())/(pred.max()-pred.min()+1e-8)
        binm = (pred>=0.5).astype(np.uint8)
        try: resample_nearest = Image.Resampling.NEAREST
        except AttributeError: resample_nearest = Image.NEAREST  # type: ignore[attr-defined]
        resized = Image.fromarray(binm*255).resize((right-left, lower-upper), resample_nearest)
        mask[upper:lower, left:right] = np.array(resized)//255
    return mask

# --- MAIN EXECUTION ---
def main():
    # Step 0: Load input image
    logger.info(f"Loading image from: {IMAGE_PATH}")
    with open(IMAGE_PATH, 'rb') as f:
        img_bytes = f.read()

    # Step 1: Osteoporosis Analysis with Grad-CAM
    logger.info("Running osteoporosis analysis...")
    pil_gray = Image.open(BytesIO(img_bytes)).convert('L')
    bg = rolling_ball_bg(np.array(pil_gray))
    pil_rgb = Image.fromarray(bg.astype('uint8'), 'L').convert('RGB')
    orig, sal, ovl, idx = compute_saliency(pil_rgb)
    has_osteo = (idx == 1)
    logger.info(f"Osteoporosis analysis complete. Result: {diag_sentences[int(idx)]}")

    # Step 2: Atheroma Detection Pipeline
    logger.info("Running atheroma detection pipeline...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name
    
    ath_clf_model = get_atheroma_classifier()
    raw_pred_ath = predict_classifier_model(tmp_path, ath_clf_model, device, atheroma_class_names)
    has_ath = (raw_pred_ath.lower() == "ateroma")
    logger.info(f"Atheroma classification result: {'Positive' if has_ath else 'Negative'}")

    if has_ath:
        logger.info("Atheroma positive. Running detection and segmentation...")
        detection_session, detection_input_name = get_detection_models()
        boxes, scores, labels_ = predict_detection_model(detection_session, detection_input_name, tmp_path)
        seg_model = get_segmentation_model()
        mask = predict_segmentation_model(tmp_path, seg_model, device)
        
        cv_img = cv2.imread(tmp_path)
        for b,s,lbl in zip(boxes, scores, labels_):
            x0,y0,x1,y1 = map(int, b)
            cv2.rectangle(cv_img, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(cv_img, f"{lbl}:{s:.2f}", (x0, max(0,y0-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        inds = np.where(mask>0)
        cv_img[inds] = (0,0,255) # Blue mask for segmentation
        detection_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    else:
        detection_img = np.array(Image.open(BytesIO(img_bytes)).convert('RGB'))
    
    os.remove(tmp_path) # Clean up temporary file

    # Step 3: Create Combined Visualization
    logger.info("Creating combined visualization plot...")
    fig = plt.figure(figsize=(12, 6))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0)
    ax0, ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ax0.imshow(orig); ax0.axis('off')
    ax1.imshow(ovl); ax1.axis('off')
    ax2.imshow(detection_img); ax2.axis('off')
    if not has_ath:
        ax2.text(0.5, 0.5, 'NO CACS DETECTED', transform=ax2.transAxes, color='red', fontsize=20, ha='center', va='center')
    fig.subplots_adjust(hspace=0)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    buf.seek(0)
    combined_data = buf.getvalue()
    combined_pil = Image.open(BytesIO(combined_data)).convert('RGB')
    logger.info("Combined visualization created.")

    # Step 4: MedGemma Analysis
    logger.info("Starting MedGemma analysis...")
    model_med = get_med_model()
    proc_med = get_med_processor()

    if LANGUAGE == 'portuguese':
        sys_txt_template = "Você é um radiologista especialista. Sua tarefa é analisar a imagem fornecida e fornecer um relatório detalhado e medicamente preciso. O paciente tem indicações de Ateroma: {atheroma} e Osteoporose: {osteo}."
        sys_txt = sys_txt_template.format(atheroma=str(has_ath), osteo=str(has_osteo))
        user_prompt = USER_PROMPT_PORTUGUESE
    else:  # English (default)
        sys_txt_template = "You are an expert radiologist. Your task is to analyze the provided image and provide a detailed, and medically accurate report. This patient has indications of Atheroma: {atheroma} and Osteoporosis: {osteo}."
        sys_txt = sys_txt_template.format(atheroma=str(has_ath), osteo=str(has_osteo))
        user_prompt = USER_PROMPT_ENGLISH

    # Load and append diagnosis template
    try:
        with open("prompt/generic_diagnosis_template.txt", "r") as f:
            diagnosis_template = f.read()
        template_instruction = f"\\n\\nPlease follow this diagnosis template:\\n\\n{diagnosis_template}" if LANGUAGE == 'english' else f"\\n\\nPor favor, siga este modelo de diagnóstico:\\n\\n{diagnosis_template}"
        user_prompt += template_instruction
    except FileNotFoundError:
        logger.warning("Diagnosis template not found. Proceeding without it.")

    msgs = [
        {"role":"system", "content":[{"type":"text","text":sys_txt}]},
        {"role":"user",   "content":[{"type":"text","text":user_prompt}, {"type":"image","image":combined_pil}]}
    ]
    
    inputs = proc_med.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(device, torch.bfloat16)
    ilen = inputs['input_ids'].shape[-1]
    
    with torch.inference_mode():
        gids = model_med.generate(**inputs, max_new_tokens=2000, do_sample=True, temperature=0.5, top_p=0.9, top_k=50)
    result = proc_med.decode(gids[0][ilen:], skip_special_tokens=True)
    
    print("\n--- Generated Report ---")
    print(result)

    # Step 5: PDF Generation and Saving
    logger.info("Generating PDF report...")
    pdf_bytes = generate_pdf_report(combined_pil, result, STATIC_DIR, PDF_LOGO_FILENAME)
    
    base_image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    pdf_filename = f"{base_image_name}_report.pdf"
    pdf_output_path = os.path.join(PDF_OUTPUT_DIR, pdf_filename)
    
    with open(pdf_output_path, "wb") as f:
        f.write(pdf_bytes)
    
    logger.info(f"PDF report saved to: {pdf_output_path}")

if __name__ == '__main__':
    main()
