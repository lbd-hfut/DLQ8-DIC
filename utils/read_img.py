import torch,os
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset

class Img_Dataset(Dataset):
    def __init__(self, train_root):
        
        image_files = np.array([x.path for x in os.scandir(train_root)
                             if (x.name.endswith(".bmp") or
                             x.name.endswith(".png") or 
                             x.name.endswith(".JPG") or 
                             x.name.endswith(".tiff"))
                             ])
        image_files.sort()
        
        self.rfimage_files = [image_files[0]]
        self.mask_files = [image_files[-1]]
        self.rfimage = self.open_image(self.rfimage_files[0])
        self.mask = self.open_image(self.mask_files[0])
        self.dfimage_files = image_files[1:-1]
        
    def __len__(self):
        return len(self.dfimage_files)
    
    def __getitem__(self, idx):
        # Open images
        df_image = self.open_image(self.dfimage_files[idx])
        # df_image = self.to_tensor(df_image)
        return df_image
    
    def open_image(self,name):
        img = Image.open(name).convert('L')
        img = np.array(img)
        return img
    
    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        elif isinstance(array, torch.Tensor):
            if array.dtype != torch.float32:
                array = array.to(torch.float32)
            return array
        else:
            raise TypeError("Unsupported type for to_tensor")
    
    def data_collect_tensor(self, device):
        unique_values = np.unique(self.mask) # Get the unique value in the mask
        if len(unique_values) == 2:
            self.mask = (self.mask > 0)
        else:
            self.mask = (self.mask == 255)
        
        RG = self.to_tensor(self.rfimage).to(device)
        ROI = self.to_tensor(self.mask).to(device)
        
        H,L = self.rfimage.shape
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, L)
        IX, IY = np.meshgrid(x, y)
        IX = self.to_tensor(IX)
        IY = self.to_tensor(IY)
        XY = torch.stack((IX, IY), dim=2).unsqueeze(0).to(device)
        XY_roi = np.column_stack(np.where(ROI == 1))
        XY_roi = torch.tensor(XY_roi).to(device)
        Ixy = torch.zeros_like(XY_roi)
        Ixy = Ixy.to(torch.float32)
        # Ixy->, [-1,1]
        Ixy[:,0] = 2 * (XY_roi[:, 1] - XY_roi[:, 1].min()) / \
            (XY_roi[:, 1].max() - XY_roi[:, 1].min()) - 1
        Ixy[:,1] = 2 * (XY_roi[:, 0] - XY_roi[:, 0].min()) / \
            (XY_roi[:, 0].max() - XY_roi[:, 0].min()) - 1
        Ixy = Ixy.to(device)
        return RG, ROI, XY, XY_roi, Ixy
    
    def data_collect_numpy(self):
        unique_values = np.unique(self.mask) # Get the unique value in the mask
        if len(unique_values) == 2:
            self.mask = (self.mask > 0)
        else:
            self.mask = (self.mask == 255)
        
        RG = self.rfimage
        ROI = self.mask
        
        H,L = self.rfimage.shape
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, L)
        IX, IY = np.meshgrid(x, y)
        XY = np.stack((IX, IY), axis=2)
        XY = XY[np.newaxis, ...]
        XY_roi = np.column_stack(np.where(ROI == 1))
        Ixy = np.zeros_like(XY_roi).astype(float)
        # Ixy->, [-1,1]
        Ixy[:,0] = 2 * (XY_roi[:, 1] - XY_roi[:, 1].min()) / (XY_roi[:, 1].max() - XY_roi[:, 1].min()) - 1
        Ixy[:,1] = 2 * (XY_roi[:, 0] - XY_roi[:, 0].min()) / (XY_roi[:, 0].max() - XY_roi[:, 0].min()) - 1
        return RG, ROI, XY, XY_roi, Ixy
    
def collate_fn(batch):
    return batch  

class XY_Dataset(Dataset):
    def __init__(self, roi, device):
        self.mask = roi
        # Get indices where the mask is greater than 0, returning (y, x) coordinates
        self.XY_roi = np.column_stack(np.where(self.mask > 0))
        self.Ixy = np.zeros_like(self.XY_roi).astype(float)
        self.Ixy[:,0] = 2 * (self.XY_roi[:, 1] - self.XY_roi[:, 1].min()) / (self.XY_roi[:, 1].max() - self.XY_roi[:, 1].min()) - 1
        self.Ixy[:,1] = 2 * (self.XY_roi[:, 0] - self.XY_roi[:, 0].min()) / (self.XY_roi[:, 0].max() - self.XY_roi[:, 0].min()) - 1 
        self.device = device
        self.XY_roi = torch.tensor(self.XY_roi, dtype=torch.long).to(self.device)
        self.Ixy = torch.tensor(self.Ixy, dtype=torch.float32).to(self.device)
    
    def __len__(self):
        N, _ = self.XY_roi.shape
        return N
    
    def __getitem__(self, idx):
        # Get the (y, x) coordinates corresponding to the index
        y_batch, x_batch = self.XY_roi[idx, :]
        Ixy_batch = self.Ixy[idx, :]
        return Ixy_batch, y_batch, x_batch

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for to_tensor")
        
        
def collate_fn_D(batch):
    # Separate the u, v samples and (y, x) coordinates from the batch and convert to numpy arrays
    Ixy_batch = torch.cat([item[0].view(1,2) for item in batch], dim=0)
    y_batch = torch.stack([item[1] for item in batch])
    x_batch = torch.stack([item[2] for item in batch])
    return Ixy_batch, y_batch, x_batch