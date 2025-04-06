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

class PhysicsInformedNN1:
    def __init__(self, Train_params, DNN_params, Data_params, device):
        
        self.Iref = torch.tensor(Data_params["RG"], dtype=torch.float32)
        self.Idef = torch.tensor(Data_params["DG"], dtype=torch.float32)
        self.ROI = torch.tensor(Data_params["ROI"], dtype=torch.bool)
        self.XY = torch.tensor(Data_params["XY"], dtype=torch.float32)
        self.XY_ROI = torch.tensor(Data_params["XY_ROI"], dtype=torch.long)
        self.Ixy = torch.tensor(Data_params["Ixy"], dtype=torch.float32)
        self.scale = torch.tensor(Data_params["SCALE"])
        
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
        self.XY = self.XY.to(device)
        self.XY_ROI = self.XY_ROI.to(device)
        self.Ixy = self.Ixy.to(device)
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

    def warm_loss(self):
        self.optimizer.zero_grad()
        self.optimizer_adam.zero_grad()
        UV = self.dnn(self.Ixy)
        # Adjust the shape of the vector to match the shape of the image
        U = torch.zeros_like(self.Iref)
        V = torch.zeros_like(self.Iref)
        y_coords, x_coords = self.XY_ROI[:, 0], self.XY_ROI[:, 1]
        U[y_coords, x_coords] = UV[:, 0] * self.scale[0] + self.scale[2] 
        V[y_coords, x_coords] = UV[:, 1] * self.scale[1] + self.scale[3] 
        
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
        loss = torch.sum(abs_error) / self.XY_ROI.shape[0]
        loss.backward()
        mae = torch.abs(new_Iref[0, 0] - self.Iref) * self.ROI
        mae = torch.sum(mae) / self.XY_ROI.shape[0]
        self.mae_list.append(mae.item())
        self.epoch = self.epoch+1
        self.dnn.Earlystop(mae.item(), self.epoch)
        if self.epoch % self.freq == 1:   
            print(f'Epoch [{self.epoch}], Loss: {mae.item():.4f}')
        return loss

    def main_loss(self):
        self.optimizer.zero_grad()
        self.optimizer_adam.zero_grad()
        UV = self.dnn(self.Ixy)
        # Adjust the shape of the vector to match the shape of the image
        U = torch.zeros_like(self.Iref)
        V = torch.zeros_like(self.Iref)
        y_coords, x_coords = self.XY_ROI[:, 0], self.XY_ROI[:, 1]
        U[y_coords, x_coords] = UV[:, 0] * self.scale[0] + self.scale[2] 
        V[y_coords, x_coords] = UV[:, 1] * self.scale[1] + self.scale[3] 
        
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
        loss = torch.sum(abs_error) / self.XY_ROI.shape[0]
        loss.backward()
        mae = torch.abs(new_Iref[0, 0] - self.Iref) * self.ROI
        mae = torch.sum(mae) / self.XY_ROI.shape[0]
        self.mae_list.append(mae.item())
        self.epoch = self.epoch+1
        self.dnn.Earlystop(mae.item(), self.epoch)
        if self.epoch % self.freq == 1:   
            print(f'Epoch [{self.epoch}], Loss: {mae.item():.4f}')
        return loss
    
    def predict(self):
        self.dnn.eval()
        UV = self.dnn(self.Ixy)
        U = torch.zeros_like(self.Iref)
        V = torch.zeros_like(self.Iref)
        y_coords, x_coords = self.XY_ROI[:, 0], self.XY_ROI[:, 1]
        U[y_coords, x_coords] = UV[:, 0] * self.scale[0] + self.scale[2] 
        V[y_coords, x_coords] = UV[:, 1] * self.scale[1] + self.scale[3]
        u = U.cpu().detach().numpy()
        v = V.cpu().detach().numpy()
        return u, v
    
    def Strain_predict(self, xy_loader):
        ux = torch.zeros_like(self.Iref); ux_numpy = ux.cpu().detach().numpy()
        uy = torch.zeros_like(self.Iref); uy_numpy = uy.cpu().detach().numpy()
        vx = torch.zeros_like(self.Iref); vx_numpy = vx.cpu().detach().numpy()
        vy = torch.zeros_like(self.Iref); vy_numpy = vy.cpu().detach().numpy()
        xmax, _ = self.XY_ROI[:,1].max(dim=0); xmin, _ = self.XY_ROI[:,1].min(dim=0)
        ymax, _ = self.XY_ROI[:,0].max(dim=0); ymin, _ = self.XY_ROI[:,0].min(dim=0)
        xmax = xmax.item(); xmin = xmin.item(); ymax = ymax.item(); ymin = ymin.item()
        self.dnn.eval()
        for Ixy_batch, y_coords, x_coords in xy_loader:
            y_coords = y_coords.cpu().detach().numpy()
            x_coords = x_coords.cpu().detach().numpy()
            Ixy_batch.requires_grad_(True)
            
            x = Ixy_batch[:,0].view(-1,1); y = Ixy_batch[:,1].view(-1,1)
            XY = torch.cat((x, y), dim=1)
            
            UV = self.dnn(XY)
            u = UV[:,0]; v = UV[:,1]
            
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True, allow_unused=True)[0]
            u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True, allow_unused=True)[0]
            v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True, allow_unused=True)[0]
            v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True, allow_unused=True)[0]

            ux_numpy[y_coords, x_coords] = u_x[:,0].cpu().detach().numpy() / (xmax - xmin) * 2 * self.scale[0].item()
            uy_numpy[y_coords, x_coords] = u_y[:,0].cpu().detach().numpy() / (ymax - ymin) * 2 * self.scale[0].item()
            vx_numpy[y_coords, x_coords] = v_x[:,0].cpu().detach().numpy() / (xmax - xmin) * 2 * self.scale[1].item()
            vy_numpy[y_coords, x_coords] = v_y[:,0].cpu().detach().numpy() / (ymax - ymin) * 2 * self.scale[1].item()
        return ux_numpy, uy_numpy, vx_numpy, vy_numpy
    
def result_plot(model, parameter_path ,IX, IY):
    model.dnn.load_state_dict(torch.load(parameter_path))
    u1,v1 = model.predict(IX, IY)
    plt.figure(dpi=300)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    plt.figure(figsize=(8, 3))
    plt.imshow(v1, cmap='jet', interpolation='nearest', norm=norm)
    plt.colorbar()
    plt.axis('off')
    plt.title("Predicted displacement field v")  