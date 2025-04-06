import os
import numpy as np
import scipy.io as io

def to_matlab(path, filename, u, v):
    if not os.path.exists(path):
        os.makedirs(path)
    io.savemat(path+filename+'.mat',{'u':u, 'v':v})
    print(f"The .mat file has been saved to: {path+filename+'.mat'}")

def Strain_to_matlab(path, filename, exx, eyy, exy):
    if not os.path.exists(path):
        os.makedirs(path)
    io.savemat(path+filename+'.mat',{'exx':exx, 'eyy':eyy})
    print(f"The .mat file has been saved to: {path+filename+'.mat'}")
    
def to_txt(path, filename, u, v, roi):
    if not os.path.exists(path):
        os.makedirs(path)
    output_file_path = path+filename+'.txt'
    y, x = np.where(roi != 0)
    uValue = u[roi != 0]
    vValue = v[roi != 0]
    xyuv = np.hstack(
        [x.reshape(-1, 1), \
         y.reshape(-1, 1), \
         uValue.reshape(-1, 1), \
         vValue.reshape(-1, 1)]
        )
    
    with open(output_file_path, 'w') as f:
        f.write("The first and second columns represent the pixel coordinates x, y. "
                "The third and fourth columns represent the displacement values u, v\n\n")
        f.write("\n")
        np.savetxt(f, xyuv, fmt='%.5f')
        f.write("\n")
    print(f"The .txt file has been saved to: {path+filename+'.txt'}")
    
def Strain_to_txt(path, filename, exx, eyy, exy, roi):
    if not os.path.exists(path):
        os.makedirs(path)
    output_file_path = path+filename+'.txt'
    y, x = np.where(roi != 0)
    exxValue = exx[roi != 0]
    eyyValue = eyy[roi != 0]
    exyValue = exy[roi != 0]
    xye = np.hstack(
        [x.reshape(-1, 1), \
         y.reshape(-1, 1), \
         exxValue.reshape(-1, 1), \
         eyyValue.reshape(-1, 1), \
         exyValue.reshape(-1, 1)]
        )
    
    with open(output_file_path, 'w') as f:
        f.write("The first and second columns represent the pixel coordinates x, y. "
                "The third and fourth columns represent the displacement values u, v\n\n")
        f.write("\n")
        np.savetxt(f, xye, fmt='%.5f')
        f.write("\n")
    print(f"The .txt file has been saved to: {path+filename+'.txt'}")