import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as io
import math
import csv

def sift_matching_within_roi(reference_img_path, deformed_img_path, roi_img_path, max_matches=100):
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
    
    data = np.hstack((matched_pts_ref, matched_pts_def, displacements))
    return data
  
def remove_outliers(data, threshold=1):
    # Check the input data type  
    if not isinstance(data, np.ndarray):  
        raise ValueError("Input data must be a numpy array")  
  
    # Copy the data to avoid modifying the original data  
    cleaned_data = data.copy()  
    flag = np.ones((data.shape[0],), dtype=bool)  # Use boolean type for clarity  
    for col in range(cleaned_data.shape[1] - 2, cleaned_data.shape[1]):  # Process only the last two columns  
        mean = np.mean(cleaned_data[:, col])  
        std = np.std(cleaned_data[:, col])  
        mask = (cleaned_data[:, col] >= mean - threshold * std) & (cleaned_data[:, col] <= mean + threshold * std)  
        flag = np.logical_and(flag, mask)  
    cleaned_data = cleaned_data[flag, :]  
    return cleaned_data

def match_plot(data, reference_img_path, deformed_img_path, save_dir=None, filename=None):
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
    

def scalelist_fun(sift_params, Train_params):
    max_matches = sift_params["max_matches"]
    safety_factor = sift_params["safety_factor"]
    threshold = sift_params["threshold"]
    
    path = Train_params['save_data_path']
    if not os.path.exists(path+'scale_information'):
        os.makedirs(path+'scale_information')
    directory = path +'scale_information'
    
    image_files = np.array([x.path for x in os.scandir(Train_params["img_path"])
                         if (x.name.endswith(".bmp") or
                         x.name.endswith(".png") or 
                         x.name.endswith(".JPG") or 
                         x.name.endswith(".tiff"))
                         ])
    image_files.sort()
    
    rfimage_files = [image_files[0]]
    mask_files = [image_files[-1]]
    dfimage_files = image_files[1:-1]
    
    N = len(dfimage_files)
    
    SCALE_LIST = []
    for i in range(N):
        data = sift_matching_within_roi(
            rfimage_files[0], dfimage_files[i], mask_files[0], max_matches
            )
        data = remove_outliers(data, threshold)
        match_plot(
            data, rfimage_files[0], dfimage_files[i], 
            save_dir=directory, filename=f'example{i+1:03d}_match.png'
            )
        displacements = np.abs(data[:,4:6])
        u_max = np.max(data[:,4])
        v_max = np.max(data[:,5])
        u_min = np.min(data[:,4])
        v_min = np.min(data[:,5])
        u_scale = ((u_max - u_min)/2) * safety_factor
        v_scale = ((v_max - v_min)/2) * safety_factor
        u_scale = 1 if round(u_scale) == 0 else u_scale
        v_scale = 1 if round(v_scale) == 0 else v_scale
        # u_mean = int(np.mean(displacements[:,0]))/u_scale
        # v_mean = int(np.mean(displacements[:,1]))/v_scale
        u_mean = np.mean(data[:,4:5])
        v_mean = np.mean(data[:,5:6])

        SCALE_LIST.append([
            u_scale if round(u_scale) == 0 or abs(int(u_scale)) <= 2 else round(u_scale), 
            v_scale if round(v_scale) == 0 or abs(int(v_scale)) <= 2 else round(v_scale), 
            u_mean  if int(u_mean) == 0 or abs(int(u_mean)) <= 5 else int(u_mean), 
            v_mean  if int(v_mean) == 0 or abs(int(v_mean)) <= 5 else int(v_mean)
            ])
    
    # SCALE_LIST save to CSV file
    csv_filename = directory + '/SCALE.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # write the tltle LIne
        writer.writerow(['u_scale', 'v_scale', 'u_mean', 'v_mean'])
        # write scale data line
        for scale_data in SCALE_LIST:
            writer.writerow(scale_data)
    # io.savemat(directory+'/SCALE.mat',{'scale':SCALE_LIST})
    # print("The scale list is saved to "+directory+'/SCALE.mat')
    
    return SCALE_LIST
    
    
    
    
    
    