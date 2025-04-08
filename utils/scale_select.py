import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.interpolate import griddata

def select_seed_points(roi_ima_pth, space):
    # Load the ROI image
    roi_image = cv2.imread(roi_ima_pth, cv2.IMREAD_GRAYSCALE)
    height, width = roi_image.shape
    seed_points = []
    
    for y in range(space // 2, height, space):
        for x in range(space // 2, width, space):
            # 只选取ROI图像中灰度值为255的点
            if roi_image[y, x]  > 0:
                seed_points.append((x, y))
    
    return seed_points

def sift_matching_within_roi(reference_img_path, deformed_img_path, roi_img_path, seed_points, space, max_matches=5000):
    # Load images
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    deformed_img = cv2.imread(deformed_img_path, cv2.IMREAD_GRAYSCALE)
    roi_img = cv2.imread(roi_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the ROI image is binary
    _, roi_mask = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
    
    # Create a SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect SIFT features and compute descriptors in the reference and deformed images
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference_img, roi_mask)
    keypoints_def, descriptors_def = sift.detectAndCompute(deformed_img, roi_mask)
    
    # Match features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    descriptors_ref = descriptors_ref.astype(np.float32)  
    descriptors_def = descriptors_def.astype(np.float32)
    
    matches = flann.knnMatch(descriptors_ref, descriptors_def, k=2)
    
    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Limit the number of good matches
    if len(good_matches) > max_matches:
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
    
    # Calculate displacement of matched points
    displacements = []
    matched_pts_ref = []
    matched_pts_def = []
    
    for match in good_matches:
        ref_pt = np.array(keypoints_ref[match.queryIdx].pt)
        def_pt = np.array(keypoints_def[match.trainIdx].pt)
        displacement = def_pt - ref_pt
        
        displacements.append(displacement)
        matched_pts_ref.append(ref_pt)
        matched_pts_def.append(def_pt)
    
    displacements = np.array(displacements)
    matched_pts_ref = np.array(matched_pts_ref)
    matched_pts_def = np.array(matched_pts_def)
    
    # Filter matched points based on seed points
    filtered_matched_pts_ref = []
    filtered_matched_pts_def = []
    filtered_displacements = []
    search_radius = space 
    
    for seed_point in seed_points:
        seed_x, seed_y = seed_point
        distances = np.linalg.norm(matched_pts_ref - np.array([seed_x, seed_y]), axis=1)
        within_radius = distances <= search_radius
        if np.any(within_radius):
            indices = np.where(within_radius)[0]
            selected_index = np.random.choice(indices)
            filtered_matched_pts_ref.append(matched_pts_ref[selected_index])
            filtered_matched_pts_def.append(matched_pts_def[selected_index])
            filtered_displacements.append(displacements[selected_index])
    
    filtered_matched_pts_ref = np.array(filtered_matched_pts_ref)
    filtered_matched_pts_def = np.array(filtered_matched_pts_def)
    filtered_displacements = np.array(filtered_displacements)
    
    data = np.hstack((filtered_matched_pts_ref, filtered_matched_pts_def, filtered_displacements))
    return data

def match_plot(seedPoint, search_r, data, reference_img_path, deformed_img_path, save_dir=None, filename=None):
    matched_pts_ref, matched_pts_def = data[:,0:2], data[:,2:4]
    reference_img = cv2.imread(reference_img_path, cv2.IMREAD_GRAYSCALE)
    deformed_img = cv2.imread(deformed_img_path, cv2.IMREAD_GRAYSCALE)
    # Visualization
    reference_img_color = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)
    deformed_img_color = cv2.cvtColor(deformed_img, cv2.COLOR_GRAY2BGR)

    for pt_ref, pt_def in zip(matched_pts_ref, matched_pts_def):
        pt_ref = tuple(np.round(pt_ref).astype(int))
        pt_def = tuple(np.round(pt_def).astype(int))
        
        cv2.circle(reference_img_color, pt_ref, 5, (0, 255, 0), -1)  # Draw green circles in reference image
        cv2.circle(deformed_img_color, pt_def, 5, (0, 0, 255), -1)  # Draw red circles in deformed image
        cv2.line(deformed_img_color, pt_def, pt_ref, (255, 0, 0), 2) # Draw a line between matched points
    for seed_points in seedPoint:
        cv2.rectangle(reference_img_color, 
                      (int(seed_points[0] - search_r), int(seed_points[1] - search_r)), 
                      (int(seed_points[0] + search_r), int(seed_points[1] + search_r)), 
                      (255, 255, 255), 2)

    # Show the visualization
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.title('Reference Image')
    plt.imshow(cv2.cvtColor(reference_img_color, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title('Deformed Image')
    plt.imshow(cv2.cvtColor(deformed_img_color, cv2.COLOR_BGR2RGB))
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        print(f"SIFT match Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing 
    

def scalelist_fun(sift_params, Train_params, imgth):
    max_matches = sift_params["max_matches"]
    safety_factor = sift_params["safety_factor"]
    space = sift_params["space"]
    search_radius = sift_params["search_radius"]
    
    path = Train_params['save_data_path']
    if not os.path.exists(path+'scale_information'):
        os.makedirs(path+'scale_information')
    directory = path +'scale_information'
    
    image_files = np.array([x.path for x in os.scandir(Train_params["img_path"])
                         if (x.name.endswith(".bmp") or
                        #  x.name.endswith(".png") or 
                         x.name.endswith(".JPG") or 
                         x.name.endswith(".tiff"))
                         ])
    image_files.sort()
    
    rfimage_files = [image_files[0]]
    mask_files = [image_files[-1]]
    dfimage_files = image_files[1:-1]
    # Check if the image index is out of range
    N = len(dfimage_files)
    if imgth >= N:
        raise ValueError(f"Image index {imgth} is out of range. The total number of images is {N}.")
    # Select seed points
    seed_points = select_seed_points(mask_files[0], space)
    # Match seed points
    data = sift_matching_within_roi(
        rfimage_files[0], dfimage_files[imgth], mask_files[0], 
        seed_points, space = search_radius, max_matches = max_matches
        )
    match_plot(
        seed_points, search_radius,
        data, rfimage_files[0], dfimage_files[imgth], 
        save_dir=directory, filename=f'example{imgth+1:03d}_match.png'
        )
    u_max = np.max(data[:,4])
    v_max = np.max(data[:,5])
    u_min = np.min(data[:,4])
    v_min = np.min(data[:,5])
    u_scale = ((u_max - u_min)/2) * safety_factor
    v_scale = ((v_max - v_min)/2) * safety_factor
    u_scale = 1 if round(u_scale) == 0 else u_scale
    v_scale = 1 if round(v_scale) == 0 else v_scale
    u_mean = np.mean(data[:,4:5])
    v_mean = np.mean(data[:,5:6])

    SCALE_LIST = []
    SCALE_LIST.append([
        u_scale if round(u_scale) == 0 or abs(int(u_scale)) <= 2 else round(u_scale), 
        v_scale if round(v_scale) == 0 or abs(int(v_scale)) <= 2 else round(v_scale), 
        u_mean  if int(u_mean) == 0 or abs(int(u_mean)) <= 5 else int(u_mean), 
        v_mean  if int(v_mean) == 0 or abs(int(v_mean)) <= 5 else int(v_mean)
        ])
    
    # SCALE_LIST save to CSV file
    csv_filename = directory + f"/SCALE{imgth+1:03d}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write the tltle LIne
        writer.writerow(['u_scale', 'v_scale', 'u_mean', 'v_mean'])
        # write scale data line
        for scale_data in SCALE_LIST:
            writer.writerow(scale_data)
    print("The scale list is saved to "+directory+'/SCALE.csv')

    coord_ref, displacement = data[:,0:2], data[:,4:6]
    u_interp, v_interp = interpolate_displacements(coord_ref, displacement, mask_files[0])
    visualize_displacements(u_interp, v_interp, save_dir=directory, filename=f'interp{imgth+1:03d}.png')
    return coord_ref, displacement


def interpolate_displacements(coord_ref, displacement, roi_img_pth):
    roi_img = cv2.imread(roi_img_pth, cv2.IMREAD_GRAYSCALE)
    height, width = roi_img.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Interpolate the displacement values
    u_interp = griddata(coord_ref, displacement[:, 0], (grid_x, grid_y), method='cubic', fill_value=0)
    v_interp = griddata(coord_ref, displacement[:, 1], (grid_x, grid_y), method='cubic', fill_value=0)
    roi = roi_img > 0
    u_interp[~roi] = np.nan
    v_interp[~roi] = np.nan
    return u_interp, v_interp


def visualize_displacements(u_interp, v_interp, save_dir=None, filename=None):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title('U Displacement')
    plt.imshow(u_interp, cmap='jet')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title('V Displacement')
    plt.imshow(v_interp, cmap='jet')
    plt.colorbar()
    if save_dir and filename:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        print(f"SIFT match Figure saved to {file_path}")
    plt.close()  # Close the figure if not showing 

def normalized_coordinates(coords, ROI):
    XY_roi = np.column_stack(np.where(ROI > 0))
    X_max, X_min = XY_roi[:,0].max(), XY_roi[:,0].min()
    Y_max, Y_min = XY_roi[:,1].max(), XY_roi[:,1].min()
    coords[:,0] = (coords[:,0] - X_min) / (X_max - X_min) * 2 - 1
    coords[:,1] = (coords[:,1] - Y_min) / (Y_max - Y_min) * 2 - 1
    return coords


if __name__ == "__main__":
    sift_params = {
        "max_matches": 3000,
        'space': 50,
        'search_radius': 10,
        'safety_factor': 1.5,
    }
    Train_params = {
        "img_path": "C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle5/",
        "save_data_path": "C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle5/siftTest/",
    }
    coord_ref, displacement = scalelist_fun(sift_params, Train_params, imgth=0)
    # # Test   
    # ref_path = "C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle1/001.bmp"
    # def_path = "C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle1/002.bmp"
    # roi_img_path = "C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle1/003.bmp"
    # # 设置间隔
    # space = 30
    # # 选取种子点
    # reference_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    # roi_img = cv2.imread(roi_img_path, cv2.IMREAD_GRAYSCALE)
    # seed_points = select_seed_points(reference_img, space, roi_img)
    # # 匹配种子点
    # data = sift_matching_within_roi(ref_path, def_path, roi_img_path, seed_points, 10, max_matches=5000)
    # match_plot(data, ref_path, def_path, 
    #            save_dir="C:/02Project/Research/DIC_Boundary_comparison/Data_test/circle1/", 
    #            filename='test5.png')