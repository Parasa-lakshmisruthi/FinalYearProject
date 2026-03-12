import sys
import os
import re
import json
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# --- Path Setup ---
# Add all model directories to the list of paths Python checks.
# This allows us to import their specific modules.
sys.path.append(os.path.join(os.path.dirname(__file__), 'facexray_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lipforensics_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'efficientnet_model'))

# --- Model-Specific Imports ---
# For Face X-ray
from HRNet import get_net
from utils import load_checkpoint

# For LipForensics
from models.spatiotemporal_net import Lipreading

# For MTCNN, used by multiple detectors now
from mtcnn import MTCNN

# For EfficientNet
from training.zoo.classifiers import DeepFakeClassifier


# ==============================================================================
# HELPER CLASSES FOR EFFICIENTNET (Moved from kernel_utils.py to fix CUDA error)
# ==============================================================================
class VideoReader:
    """Helper class for reading videos"""
    def __init__(self, verbose=False, layout="CHW"):
        self.verbose = verbose
        self.layout = layout

    def read_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

class FaceExtractor:
    """Helper class for extracting faces from videos"""
    def __init__(self, video_read_fn):
        self.video_read_fn = video_read_fn
        # DEFINITIVE FIX: Import the correct MTCNN from facenet_pytorch locally
        from facenet_pytorch import MTCNN as FaceNetMTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = FaceNetMTCNN(margin=14, thresholds=[0.6, 0.7, 0.7], device=device)

    def process_video(self, video_path):
        frames = self.video_read_fn(video_path)
        if not frames:
            return []
            
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        
        all_faces = []
        # Use the facenet-pytorch batch detection method
        faces_tensors = self.detector(frames_rgb)
        
        for face_tensor in faces_tensors:
            if face_tensor is not None:
                # Convert tensor back to numpy image for processing
                face_numpy = face_tensor.permute(1, 2, 0).numpy()
                # Convert from range [0,1] or [-1,1] to [0,255] uint8
                face_numpy = (face_numpy - face_numpy.min()) / (face_numpy.max() - face_numpy.min())
                face_numpy = (face_numpy * 255).astype(np.uint8)
                all_faces.append(face_numpy)
                
        return all_faces


# ==============================================================================
# DETECTOR 1: FACE X-RAY
# ==============================================================================
def predict_with_facexray(video_path, model_path, config_path):
    """Analyzes a video with the Face X-ray model using MTCNN for face detection."""
    face_detector = MTCNN()
    devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    
    net = get_net(cfg_file=config_path, devices=devices)
    state_dict = load_checkpoint(model_path, devices)[0]
    net.load_state_dict(state_dict)
    net.eval()
    
    post_function = nn.Softmax(dim=1)
    
    reader = cv2.VideoCapture(video_path)
    frame_predictions = []
    pbar = tqdm(total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analyzing with Face X-ray")

    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break
        pbar.update(1)
        
        faces = face_detector.detect_faces(image)

        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + width, y + height
            
            cropped_face = image[y:y2, x:x2]
            
            if cropped_face.size == 0:
                continue

            resized_face = cv2.resize(cropped_face, (256, 256))
            tensor_face = transforms.ToTensor()(resized_face).unsqueeze(0)
            
            if torch.cuda.is_available():
                tensor_face = tensor_face.cuda()

            with torch.no_grad():
                _, output = net(tensor_face)
                probs = post_function(output)
                fake_prob = probs[0][1].item()
                frame_predictions.append(fake_prob)

    reader.release()
    pbar.close()

    if not frame_predictions:
        return 0.0
    return sum(frame_predictions) / len(frame_predictions)


# ==============================================================================
# DETECTOR 2: LIPFORENSICS (Corrected Class Instantiation)
# ==============================================================================
def predict_with_lipforensics(video_path, model_path):
    """Analyzes a video with the LipForensics model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    config_path = os.path.join(os.path.dirname(__file__), 'lipforensics_model', 'models', 'configs', 'lrw_resnet18_mstcn.json')
    with open(config_path, "r") as f:
        args_loaded = json.load(f)

    tcn_options = {
        "num_layers": args_loaded["tcn_num_layers"],
        "kernel_size": args_loaded["tcn_kernel_size"],
        "dropout": args_loaded["tcn_dropout"],
        "dwpw": args_loaded["tcn_dwpw"],
        "width_mult": args_loaded["tcn_width_mult"],
    }
    model = Lipreading(
        num_classes=1,
        tcn_options=tcn_options,
        relu_type=args_loaded["relu_type"]
    )

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    face_detector = MTCNN()

    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop((88, 88)),
        transforms.ToTensor(),
        transforms.Normalize((0.421,), (0.165,))
    ])

    reader = cv2.VideoCapture(video_path)
    frame_predictions = []
    pbar = tqdm(total=int(reader.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Analyzing with LipForensics")

    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        pbar.update(1)

        faces = face_detector.detect_faces(frame)
        
        if len(faces) > 0:
            keypoints = faces[0]['keypoints']
            mouth_left = keypoints['mouth_left']
            mouth_right = keypoints['mouth_right']
            
            x_min, x_max = mouth_left[0], mouth_right[0]
            y_min, y_max = mouth_left[1], mouth_right[1]
            
            y_center, x_center = (y_min + y_max) // 2, (x_min + x_max) // 2
            crop_size = 120
            y_min_crop = max(0, int(y_center - crop_size // 2))
            y_max_crop = min(frame.shape[0], int(y_center + crop_size // 2))
            x_min_crop = max(0, int(x_center - crop_size // 2))
            x_max_crop = min(frame.shape[1], int(x_center + crop_size // 2))
            
            mouth_crop = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

            if mouth_crop.size == 0:
                continue

            mouth_crop_rgb = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(mouth_crop_rgb)
            
            mouth_tensor = preprocess(pil_image)
            mouth_tensor = mouth_tensor.unsqueeze(0).unsqueeze(2)
            mouth_tensor = mouth_tensor.to(device)

            with torch.no_grad():
                prediction = torch.sigmoid(model(mouth_tensor, lengths=[1])).item()
                frame_predictions.append(prediction)

    reader.release()
    pbar.close()

    if not frame_predictions:
        return 0.0
    return sum(frame_predictions) / len(frame_predictions)


# ==============================================================================
# DETECTOR 3: EFFICIENTNET (Corrected Implementation)
# ==============================================================================
def predict_with_efficientnet(video_path, model_path):
    """Analyzes a video with a high-performance EfficientNet model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIX: Removed the 'num_classes' argument as it's not accepted by this specific class
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to(device)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    del checkpoint

    input_size = 380
    frames_per_video = 32
    
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    
    faces = face_extractor.process_video(video_path)
    
    if not faces:
        return 0.0

    predictions = []
    pbar = tqdm(total=len(faces), desc="Analyzing with EfficientNet")

    to_tensor = transforms.ToTensor()

    for face in faces:
        pbar.update(1)
        if face is None or face.size == 0:
            continue
        resized_face = cv2.resize(face, (input_size, input_size))
        
        tensor_face = to_tensor(resized_face).to(device)
        
        with torch.no_grad():
            output = model(tensor_face.unsqueeze(0))
            # Use sigmoid for single-logit output
            prediction = torch.sigmoid(output).item()
            predictions.append(prediction)
    
    pbar.close()

    if not predictions:
        return 0.0
        
    return sum(predictions) / len(predictions)


# ==============================================================================
# --- Example Test Block (Running all detectors) ---
# ==============================================================================
if __name__ == '__main__':
    # !!! IMPORTANT: DOUBLE-CHECK ALL PATHS ARE CORRECT ON YOUR SYSTEM !!!
    video_to_test = "D:/Business_Internship/Deepfake/Videos/1.mp4"
    
    # --- Paths for Face X-ray Model ---
    face_xray_model_path = 'D:/Business_Internship/Deepfake/facexray_model/best_model.pth.tar'
    hrnet_config_path = 'D:/Business_Internship/Deepfake/facexray_model/Face-X-Ray-master/Face-X-Ray-master/HRNet/hrnet_config/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    
    # --- Path for LipForensics Model ---
    lip_forensics_model_path = 'D:/Business_Internship/Deepfake/lipforensics_model/lipforensics_ff.pth'

    # --- Path for EfficientNet Model ---
    efficientnet_model_path = 'D:/Business_Internship/Deepfake/efficientnet_model/weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'

    # --- Run All Predictions ---
    print("--- Running Detector 1: Face X-ray ---")
    face_xray_score = predict_with_facexray(video_to_test, face_xray_model_path, hrnet_config_path)
    print(f"\nFace X-ray Score: {face_xray_score:.4f}")
    
    print("\n--- Running Detector 2: LipForensics ---")
    lip_forensics_score = predict_with_lipforensics(video_to_test, lip_forensics_model_path)
    print(f"LipForensics Score: {lip_forensics_score:.4f}")

    print("\n--- Running Detector 3: EfficientNet ---")
    efficientnet_score = predict_with_efficientnet(video_to_test, efficientnet_model_path)
    print(f"EfficientNet Score: {efficientnet_score:.4f}")
