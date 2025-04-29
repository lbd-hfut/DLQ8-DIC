import os
import torch
import numpy as np
import random
import sys
import time
from scipy.io import loadmat
import csv

from utils.solve import Q8_DIC_Solver
from src.Q8NN import Q8main
from utils.read_img import Img_Dataset, collate_fn, XY_Dataset, collate_fn_D
from utils.fig_plot import result_plot, contourf_plot, strain_result_plot
from utils.save_data import to_matlab, to_txt, Strain_to_matlab, Strain_to_txt
from utils.scale_select import scalelist_fun

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
seed_everything(42)

Train_params = {
    "img_path":'C:/02Project/Research/DIC_Boundary_comparison/Data_Gen/circle_continue/generate_fig/N00/S05/',
    "checkpoint": './Checkpoint/Q8-DIC/',
    "save_data_path": '',
    "adam_lr": 0.001,
    "warm_lr": 0.001,
    "main_lr": 0.0001,
    "adam_decay": 1e-2,
    "LBFGS_max_iter": 20,
    "LBFGS_max_eval": 20,
    "LBFGS_history_size": 20,
    "LBFGS_tolerance_grad": 1e-05,
    "LBFGS_search_fn": "strong_wolfe",
    "warm_adam_epoch": 100,
    "warm_bfgs_epoch": 100,
    "main_adam_epoch": 100,
    "main_bfgs_epoch": 100,
    "patience_adam": 100,
    "patience_lbfgs": 100,
    "delta_warm_adam": 0.5,
    "delta_warm_lbfgs": 0.1,
    "delta_main_adam": 0.05,
    "delta_main_lbfgs": 0.01,
    "print_feq": 10,
}
Train_params["save_data_path"] = Train_params["img_path"]+"Q8DIC/"

sift_params = {
    "max_matches": 3000, # Specify the maximum number of matches
    "safety_factor": 1., # Adjustment coefficient used as a safety margin
    'space': 30,
    'search_radius': 10,
}

Plot_params = {
    "plot_flag": True,
    "layout" : [1,2],
    "WH"     : [5,4],
    "contourN": 15,
}

DNN_params = {
    "dim": 2,
    "hidden_units": [50, 50, 50],
    "scales": [1,4,8,16],
    "activation": "tanh",
}

Data_params = {
    "RG": 0,    # Placeholder character, 
    "DG": 0,    # waiting for the program to 
    "ROI": 0,   # read and assign a value
    "node": 0,
    "element": 0,
    "Inform": 0,
    "SCALE": [1,1,0,0],  # [umax-umean, vmax-vmean, umean, vmean]
}


if __name__ == '__main__':
    img_dataset = Img_Dataset(Train_params['img_path'])
    Data_params["RG"], Data_params["ROI"], Data_params["node"], Data_params["element"], Data_params["Inform"] = img_dataset.data_collect_numpy()
    img_loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=1, 
        shuffle=False, collate_fn=collate_fn)
    # Data_params["DG"] = next(iter(img_loader))[0]
    
    xy_dataset = XY_Dataset(Data_params["ROI"], device)
    xy_loader = torch.utils.data.DataLoader(
        xy_dataset, batch_size=64*64, 
        shuffle=False, collate_fn=collate_fn_D)
    
    model = Q8main(
        Train_params=Train_params,
        DNN_params=DNN_params,
        Data_params=Data_params, 
        device=device
        )
    model.to_device(device=device)
    
    for idx, DimageL in enumerate(img_loader):
        coord_ref, uv_sift = scalelist_fun(sift_params, Train_params, idx)
        csv_file = Train_params["save_data_path"]+f'scale_information/SCALE.csv'  # {idx+1:03d}
        # 提示用户检查 csv_file
        # input(f"请检查文件 {csv_file} 是否符合标准，按任意键继续...")
        scale_list = []
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            second_row = next(csv_reader)  # Read the second row
            scale_list = [float(element) for element in second_row[:4]]
        
        model.Reload(DG=DimageL[0], scale=scale_list)
        model.dnn.initialize_weights()
        
        u, v = Q8_DIC_Solver(model=model, Train_params=Train_params)

        uv_file_name = f"uv_{idx+1:03d}"
        strain_file_name = f"strain_{idx+1:03d}"
        # result_plot
        result_plot(
            u, v,
            layout = Plot_params["layout"], 
            WH = Plot_params["WH"], 
            save_dir = Train_params["save_data_path"] + "imshow/", 
            filename = uv_file_name+".png"
            )

        contourf_plot(
            u, v, N=Plot_params["contourN"], 
            layout = Plot_params["layout"], 
            WH = Plot_params["WH"], 
            save_dir = Train_params["save_data_path"] + "contourf/", 
            filename = uv_file_name+".png"
            )
        
        to_matlab(Train_params["save_data_path"], uv_file_name, u, v)
        to_txt(Train_params["save_data_path"], uv_file_name, u, v, Data_params["ROI"])