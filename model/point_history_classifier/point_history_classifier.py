import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class PointHistoryModel(nn.Module):
    def __init__(self):
        super(PointHistoryModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32,24),
            nn.ReLU(),
            nn.Linear(24,10),
            nn.ReLU(),
            nn.Linear(10,4),
            nn.Softmax(dim = -1)
        )
    
    def forward(self,x):
        return self.model(x)

class PointHistoryClassifier(nn.Module):
    def __init__(self,model_path = 'F:\TTCS\model\point_history_classifier\converted_model.pth', score_th = 0.5, invalid_value = 0): 
        super(PointHistoryClassifier, self).__init__()
        self.model = PointHistoryModel()
        self.model.load_state_dict(torch.load(model_path,map_location = torch.device('cpu')),strict = False)
        self.model.eval()
        self.score_th = score_th
        self.invalid_value = invalid_value 
    
    def forward(self, point_history):
        input_tensor = torch.tensor(point_history,dtype = torch.float32)
        with torch.no_grad():
            result = self.model(input_tensor)

        result_probs = F.softmax(result,dim = -1).numpy().squeeze()
        result_index = np.argmax(result_probs)
        if result_probs[result_index] < self.score_th:
            result_index = self.invalid_value 
        return result_index 





