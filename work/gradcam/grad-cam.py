from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
import cv2
import numpy as np
import os
import torch
import argparse
from Models import ResNet18
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-mn', '--model_name', type=str, default='resnet18', help='the model to train')
args = parser.parse_args()
args = args.__dict__

num_channels =3
num_classes = 10    
net = None
if args['model_name'] == 'resnet18':
    net = ResNet18(in_channels=num_channels, num_classes=num_classes)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)
model = net.to(dev)

model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./Saved_model/2023/Savedmodel/resnet18/final.pth', map_location=torch.device('cpu')).items()})
model.eval()
target_layer = [model.layer4[-1]]

image_path = './data/A-examples/mnist/mnist8.png'    
rgb_img = cv2.imread(image_path, 1)              
rgb_img = np.float32(rgb_img) / 255

input_tensor = preprocess_image(rgb_img, mean=[0.5], std=[0.5])
rgb_img = np.expand_dims(rgb_img, axis=-1)
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)      
targets = None
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite(f'./GradCAM-test/mnist/resnet18/1.jpg', visualization)  

