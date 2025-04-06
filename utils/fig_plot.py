import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.io as io

def zero_to_nan(matrix):
    matrix = np.array(matrix) 
    matrix[matrix==0] = np.nan
    return matrix

def find_max_min(u, v):
    u_non_zero = u[u != 0]  # Extract non-zero elements from u
    v_non_zero = v[v != 0]  # Extract non-zero elements from v

    # Find the minimum and maximum among the non-zero elements
    umin = np.min(u_non_zero) if u_non_zero.size > 0 else None
    umax = np.max(u_non_zero) if u_non_zero.size > 0 else None
    vmin = np.min(v_non_zero) if v_non_zero.size > 0 else None
    vmax = np.max(v_non_zero) if v_non_zero.size > 0 else None
    return umin, umax, vmin, vmax

def Strain_find_max_min(ux, vy, uy, vx):
    e = (uy + vx)/2
    
    ux_non_zero = ux[ux != 0]  # Extract non-zero elements from ux
    vy_non_zero = vy[vy != 0]  # Extract non-zero elements from vy
    e_non_zero = e[e != 0]  # Extract non-zero elements from exy

    # Find the minimum and maximum among the non-zero elements
    exxmin = np.min(ux_non_zero) if ux_non_zero.size > 0 else None
    exxmax = np.max(ux_non_zero) if ux_non_zero.size > 0 else None
    eyymin = np.min(vy_non_zero) if vy_non_zero.size > 0 else None
    eyymax = np.max(vy_non_zero) if vy_non_zero.size > 0 else None
    exymin = np.min(e_non_zero)  if e_non_zero.size  > 0 else None
    exymax = np.max(e_non_zero)  if e_non_zero.size  > 0 else None
    return exxmin, exxmax, eyymin, eyymax, exymin, exymax

def result_plot(u, v,layout = [1,2], WH=[5,4], save_dir=None, filename=None):
    u_min, u_max, v_min, v_max = find_max_min(u=u, v=v)
    u = zero_to_nan(u); v = zero_to_nan(v)
    plt.figure(figsize=(WH[1]*layout[1], WH[0]*layout[0]), dpi=200)
    normu = matplotlib.colors.Normalize(vmin=u_min, vmax=u_max)
    normv = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(u, cmap='jet', interpolation='nearest', norm=normu)
    plt.colorbar()
    plt.axis('off')
    plt.title("solved: u ", fontsize=10)
    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(v, cmap='jet', interpolation='nearest', norm=normv)
    plt.colorbar()
    plt.axis('off')
    plt.title("solved: v ", fontsize=10)
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing
    
def contourf_plot(u, v, N, layout = [1,2], WH=[4,4], save_dir=None, filename=None):
    u_min, u_max, v_min, v_max = find_max_min(u=u, v=v)
    u_sub1 = zero_to_nan(u); v_sub1 = zero_to_nan(v)
    # u_sub1 = np.flip(u, axis=0); v_sub1 = np.flip(v, axis=0)
    H,L = u_sub1.shape; 
    y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
    IX,IY = np.meshgrid(x, y)
    fig, (ax1, ax2) = plt.subplots(layout[0], layout[1], figsize=(WH[0]*layout[1], WH[1]*layout[0]), dpi=200)
    c1 = ax1.contourf(IX, IY, u_sub1, N, cmap='jet')
    plt.colorbar(c1, ax=ax1, orientation='vertical')
    ax1.axis('off'); c1.set_clim(u_min, u_max)
    ax1.invert_yaxis()
    
    c2 = ax2.contourf(IX, IY, v_sub1, N, cmap='jet')
    plt.colorbar(c2, ax=ax2, orientation='vertical')
    ax2.axis('off'); c2.set_clim(v_min, v_max)
    ax2.invert_yaxis()
    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing
    

def strain_result_plot(ux, uy, vx, vy,layout = [3,1], WH=[5,4], save_dir=None, filename=None):
    exxmin, exxmax, eyymin, eyymax, exymin, exymax = Strain_find_max_min(ux, vy, uy, vx)
    exx = zero_to_nan(ux); eyy = zero_to_nan(vy); exy = zero_to_nan((vx + uy)/2)
    
    plt.figure(figsize=(WH[1]*layout[1], WH[0]*layout[0]), dpi=200)
    normexx = matplotlib.colors.Normalize(vmin=exxmin, vmax=exxmax)
    normeyy = matplotlib.colors.Normalize(vmin=eyymin, vmax=eyymax)
    normexy = matplotlib.colors.Normalize(vmin=exymin, vmax=exymax)
    
    plt.subplot(layout[0], layout[1], 1)
    plt.imshow(exx, cmap='jet', interpolation='nearest', norm=normexx)
    plt.colorbar()
    plt.axis('off')
    plt.title("solved: exx ", fontsize=10)
    plt.subplot(layout[0], layout[1], 2)
    plt.imshow(eyy, cmap='jet', interpolation='nearest', norm=normeyy)
    plt.colorbar()
    plt.axis('off')
    plt.title("solved: eyy ", fontsize=10)
    plt.subplot(layout[0], layout[1], 3)
    plt.imshow(exy, cmap='jet', interpolation='nearest', norm=normexy)
    plt.colorbar()
    plt.axis('off')
    plt.title("solved: exy ", fontsize=10)

    # Save the figure if save directory and filename are provided
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing