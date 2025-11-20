import os
import glob
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
import numpy as np  

class Classifier():
    def __init__(self, model_file):
        self.model_file = model_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()


    def load_model(self):
        self.model = torch.load(self.model_file,map_location='cpu')
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully")
        
        
    def load_image(self, image_dir):
        img_sitk = sitk.ReadImage(image_dir)
        image = sitk.GetArrayFromImage(img_sitk)
        image = self.transform_image(image)
        return image.to(self.device).unsqueeze(0)

    def predict(self, image):
        # 1. load image
        img = self.load_image(image) if isinstance(image, str) else self.load_array(image)
        # 2. predict
        pred_proba = -1

        with torch.no_grad(): 
            ###########  predict ##############
            outputs = self.model(img)
            preds = torch.softmax(outputs, dim=1)
            torch.cuda.empty_cache()
            pred_proba = float(preds.squeeze(0).cpu().numpy()[1])
        return pred_proba

    def transform_image(self, image):
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        return image

if __name__ == '__main__':
    # set the path of the data
    image_path = './test_non_contrast_ccrcc_image.nii.gz'
    # set the path of the model and parameters
    param_path = './Resnet50_tumor_class_phase1_v2_best_ckpt.pt'
    classifier = Classifier(model_file=param_path)
    pred = classifier.predict(image_path)
    print('The probability of clear cell renal carcinoma is: %.4f' % pred)

    