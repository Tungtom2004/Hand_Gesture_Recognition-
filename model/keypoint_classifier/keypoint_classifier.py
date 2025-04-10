import torch 
import torch.nn as nn 
import numpy as np  

class KeyPointModel(nn.Module):
    def __init__(self):
        super(KeyPointModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(42,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,4),
            nn.Softmax(dim=-1)  # Use Softmax for multi-class classification
        )

    def forward(self,x):
        return self.model(x)

class KeyPointClassifier(nn.Module):
    def __init__(self, model_path = 'F:\TTCS\model\keypoint_classifier\converted_model.pth'):
        super(KeyPointClassifier,self).__init__()
        self.model = KeyPointModel()
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=False)
        self.model.eval()
    
    def __call__(self,landmark_list):
        input_tensor = torch.tensor([landmark_list],dtype = torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor)
        
        result_index = torch.argmax(output,dim = 1).item()
        return result_index 
