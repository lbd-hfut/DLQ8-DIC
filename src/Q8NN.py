import torch
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import os

from src.FCNN import MscaleDNN

class Q8main:
    def __init__(self, Train_params, DNN_params, Data_params, device):
        
        self.Iref = torch.tensor(Data_params["RG"], dtype=torch.float32)
        self.Idef = torch.tensor(Data_params["DG"], dtype=torch.float32)
        self.ROI = torch.tensor(Data_params["ROI"], dtype=torch.bool)
        self.node = torch.tensor(Data_params["node"], dtype=torch.float32)
        self.element = torch.tensor(Data_params["element"], dtype=torch.int32)
        self.Inform = torch.tensor(Data_params["Inform"], dtype=torch.float32)
        self.scale = torch.tensor(Data_params["SCALE"])
        H,L = self.Iref.shape
        y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L)
        IX, IY = np.meshgrid(x, y)
        IX = torch.tensor(IX, dtype=torch.float32)
        IY = torch.tensor(IY, dtype=torch.float32)
        self.XY = torch.stack((IX, IY), dim=2).unsqueeze(0).to(device)
        
        # 加载深度学习网络
        self.dnn = MscaleDNN(
            input_dim=DNN_params["dim"],
            hidden_units=DNN_params["hidden_units"],
            output_dim=2,
            scales=DNN_params["scales"],
            activation=DNN_params["activation"]
        )

        self.epoch = 0          # 记录当前训练的epoch
        self.freq = Train_params["print_feq"]
        self.mae_list = []      # 保存每epoch的mae
        self.device = device
        
    def Reload(self, DG, scale):
        self.Idef = torch.tensor(DG, dtype=torch.float32)
        self.Idef = self.Idef.to(self.device)
        self.scale = torch.tensor(scale)
        self.scale = self.scale.to(self.device)
        self.epoch = 0
        
    def to_device(self, device):
        self.Iref = self.Iref.to(device)
        self.Idef = self.Idef.to(device)
        self.ROI = self.ROI.to(device)
        self.node = self.node.to(device)
        self.element = self.element.to(device)
        self.Inform = self.Inform.to(device)
        self.XY = self.XY.to(device)
        self.scale = self.scale.to(device)
        self.dnn = self.dnn.to(device)
        
    def set_optim(self, Train_params, method="LBFGS"):
        # 设置优化器
        if method == "LBFGS":
            self.optimizer = torch.optim.LBFGS(
                self.dnn.parameters(), lr=1, 
                max_iter = Train_params["LBFGS_max_iter"], 
                max_eval = Train_params["LBFGS_max_eval"],
                history_size = Train_params["LBFGS_history_size"], 
                tolerance_grad = Train_params["LBFGS_tolerance_grad"], 
                tolerance_change = 0.5 * np.finfo(float).eps,
                line_search_fn = Train_params["LBFGS_search_fn"]
                )
        elif method == "ADAM":
            self.optimizer_adam = torch.optim.Adam(
                self.dnn.parameters(),                      
                lr = Train_params["adam_lr"],                  
                weight_decay = Train_params["adam_decay"]   
                )
        elif method == "ADJUST":
            for param_group in self.optimizer_adam.param_groups:
                param_group['lr'] = Train_params["main_lr"]
        else:
            raise ValueError("options for method: LBFGS, ADAM, ADJUST")
    
    def shapef(self, xi, eta):
        N = torch.zeros(xi.shape[0], 8).to(self.device)
        N[:, 0] = (1 - xi) * (1 - eta) * (-xi - eta - 1) / 4
        N[:, 1] = (1 - xi ** 2) * (1 - eta) / 2
        N[:, 2] = (1 + xi) * (1 - eta) * (xi - eta - 1) / 4
        N[:, 3] = (1 - eta ** 2) * (1 + xi) / 2
        N[:, 4] = (1 + xi) * (1 + eta) * (xi + eta - 1) / 4
        N[:, 5] = (1 - xi ** 2) * (1 + eta) / 2
        N[:, 6] = (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4
        N[:, 7] = (1 - eta ** 2) * (1 - xi) / 2
        return N

    def Q8_uv(self):
        U = self.dnn(self.node[:,1:])
        # 初始化节点位移
        Node_u = torch.cat([self.node, U[:, 0].view(-1, 1)], dim=1)  # U[:, 0] 为 x 方向的位移
        Node_v = torch.cat([self.node, U[:, 1].view(-1, 1)], dim=1)  # U[:, 1] 为 y 方向的位移

        # 创建全局位移场
        u_global = torch.zeros_like(self.Iref).to(self.device)  # 每个像素点的 x 位移
        v_global = torch.zeros_like(self.Iref).to(self.device)  # 每个像素点的 y 位移

        u_local = torch.stack([Node_u[self.element[:, 1]-1, 3], Node_u[self.element[:, 5]-1, 3], 
                               Node_u[self.element[:, 2]-1, 3], Node_u[self.element[:, 6]-1, 3], 
                               Node_u[self.element[:, 3]-1, 3], Node_u[self.element[:, 7]-1, 3], 
                               Node_u[self.element[:, 4]-1, 3], Node_u[self.element[:, 8]-1, 3]], 
                               dim=1) # self.element[:, 1]-1 -1操作是torch索引从0开始
        v_local = torch.stack([Node_v[self.element[:, 1]-1, 3], Node_v[self.element[:, 5]-1, 3], 
                               Node_v[self.element[:, 2]-1, 3], Node_v[self.element[:, 6]-1, 3], 
                               Node_v[self.element[:, 3]-1, 3], Node_v[self.element[:, 7]-1, 3], 
                               Node_v[self.element[:, 4]-1, 3], Node_v[self.element[:, 8]-1, 3]], 
                               dim=1) # shape: [num_elements, 8]
        
        # 将每个单元的位移值通过形函数与像素点关联，更新每个像素点的位移值
        for i in range(self.element.shape[0]):
            x_global = self.Inform[self.Inform[:, 4] == i+1, 0:2] #读取单元内的全局
            x_local  = self.Inform[self.Inform[:, 4] == i+1, 2:4] #读取对应的局部坐标
            N_ij = self.shapef(x_local[:, 0], x_local[:, 1]) # 计算形函数 N_ij:-> [num_points, 8]
            u_gloabl_i = torch.matmul(N_ij, u_local[i, :].T)
            v_global_i = torch.matmul(N_ij, v_local[i, :].T)
            u_global[x_global[:, 1].long(), x_global[:, 0].long()] = u_gloabl_i
            v_global[x_global[:, 1].long(), x_global[:, 0].long()] = v_global_i
        
        return u_global, v_global

    def warm_loss(self):
        self.optimizer.zero_grad()
        self.optimizer_adam.zero_grad()

        U, V = self.Q8_uv()
        # Interpolate a new deformed image
        target_height = self.Idef.shape[0]
        target_width  = self.Idef.shape[1]
        u = U/(target_width/2); v = V/(target_height/2)
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        X_new = self.XY + uv_displacement
        
        # 插值新的散斑图
        new_Iref = F.grid_sample(
            self.Idef.view(1, 1, target_height, target_width), 
            X_new.view(1, target_height, target_width, 2), 
            mode='bilinear', align_corners=True
            )
        
        # 计算两张图的相关数
        abs_error = (new_Iref[0, 0] - self.Iref)**2 * self.ROI
        abs_error = torch.log(1+abs_error)
        loss = torch.sum(abs_error) / self.Inform.shape[0]
        loss.backward()
        mae = torch.abs(new_Iref[0, 0] - self.Iref) * self.ROI
        mae = torch.sum(mae) / self.Inform.shape[0]
        self.mae_list.append(mae.item())
        self.epoch = self.epoch+1
        self.dnn.Earlystop(mae.item(), self.epoch)
        if self.epoch % self.freq == 1:   
            print(f'Epoch [{self.epoch}], Loss: {mae.item():.4f}')
        return loss

    def main_loss(self):
        self.optimizer.zero_grad()
        self.optimizer_adam.zero_grad()

        U, V = self.Q8_uv()
        # Interpolate a new deformed image
        target_height = self.Idef.shape[0]
        target_width  = self.Idef.shape[1]
        u = U/(target_width/2); v = V/(target_height/2)
        uv_displacement = torch.stack((u, v), dim=2).unsqueeze(0)
        X_new = self.XY + uv_displacement
        
        # 插值新的散斑图
        new_Iref = F.grid_sample(
            self.Idef.view(1, 1, target_height, target_width), 
            X_new.view(1, target_height, target_width, 2), 
            mode='bilinear', align_corners=True
            )
        
        # 计算两张图的相关数
        abs_error = (new_Iref[0, 0] - self.Iref)**2 * self.ROI
        loss = torch.sum(abs_error) / self.Inform.shape[0]
        loss.backward()
        mae = torch.abs(new_Iref[0, 0] - self.Iref) * self.ROI
        mae = torch.sum(mae) / self.Inform.shape[0]
        self.mae_list.append(mae.item())
        self.epoch = self.epoch+1
        self.dnn.Earlystop(mae.item(), self.epoch)
        if self.epoch % self.freq == 1:   
            print(f'Epoch [{self.epoch}], Loss: {mae.item():.4f}')
        return loss
    
    def predict(self):
        self.dnn.eval()
        with torch.no_grad():
            U, V = self.Q8_uv()
        u = U.cpu().detach().numpy()
        v = V.cpu().detach().numpy()
        return u, v

    