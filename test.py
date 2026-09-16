import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ImagesCameras import DepthTensor, ImageTensor
from matplotlib import colors, cm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

if __name__ == "__main__":
    # model = YOLO('yolo26n-seg.pt')
    # results = model.predict(
    #     source="/home/godeta/Images/images_rapport/4th part/Lap_on_comp_gray.png",
    #     save=False,
    #     imgsz=640,
    #     conf=0.25
    # )
    #
    # res = results[0]
    # annotator = Annotator(res.orig_img)
    #
    # # 2. Iterate through detected boxes and draw them in blue
    # if res.boxes is not None:
    #     for box in res.boxes:
    #         xyxy = box.xyxy[0]
    #         cls = int(box.cls[0])
    #         label = f"{model.names[cls]} {box.conf[0]:.2f}"
    #         # Blue bounding box
    #         annotator.box_label(xyxy, label, color=(0, 0, 255))
    #
    # # 3. Iterate and draw matching upscaled masks
    # if res.masks is not None:
    #     # Use res.masks.xy to get contours that match the original image size exactly
    #     orig_h, orig_w = res.orig_img.shape[:2]
    #
    #     # Loop over each individual mask to safely overlay them
    #     for mask_tensor in res.masks.data:
    #         # Move to CPU, convert to numpy
    #         mask_np = mask_tensor.cpu().numpy()
    #
    #         # Resize mask to exactly match original image dimensions if there's an internal mismatch
    #         if mask_np.shape != (orig_h, orig_w):
    #             mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    #
    #         # Convert to boolean mask
    #         mask_bool = mask_np.astype(bool)
    #
    #         # Create an empty overlay image to apply the mask color elegantly
    #         overlay = annotator.im.copy()
    #         overlay[mask_bool] = (0, 0, 255)  # Apply pure blue
    #
    #         # Blend the colored mask back into the main annotator canvas
    #         alpha = 0.4
    #         cv2.addWeighted(overlay, alpha, annotator.im, 1 - alpha, 0, dst=annotator.im)
    #
    # # 4. Extract final image matrix and save
    # annotated_img = annotator.result()
    # output_path = "/home/godeta/Documents/These/4 - Fusion Methods: Multifold problem/images/Intro/yolo26n-seg/decision_result.png"
    # cv2.imwrite(output_path, annotated_img)



#     left_path = "/home/godeta/PycharmProjects/WAFT-Stereo/assets/Cars/left.png"  # Replace with actual path to left frames
#     right_path = "/home/godeta/PycharmProjects/WAFT-Stereo/assets/Cars/right.png"  # Replace with actual path to left frames
    #
    # # 2. Configure the Semi-Global Block Matching (SGBM) parameters
    # # A standard window size (block_size) is between 3 and 7 pixels
    # block_size = 5
    # min_disp = 0
    # num_disp = 64 - min_disp  # Must be divisible by 16
    #
    # # Calculate smoothness penalties P1 and P2 based on Heiko Hirschmüller's logic
    # # P1 penalizes small disparity changes (e.g., 1 pixel), P2 penalizes larger changes
    # P1 = 8 * 1 * block_size ** 2
    # P2 = 32 * 1 * block_size ** 2
    #
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=min_disp,
    #     numDisparities=num_disp,
    #     blockSize=block_size,
    #     P1=P1,
    #     P2=P2,
    #     disp12MaxDiff=1,  # Left-Right consistency check threshold
    #     uniquenessRatio=10,  # Margin percentage to avoid ambiguous matches
    #     speckleWindowSize=100,  # Noise filtering window size
    #     speckleRange=1,  # Max disparity variation within a speckle component
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Uses a highly efficient 3-path cost aggregation
    # )
    #
    # stereo_bm = cv2.StereoBM_create(
    #     numDisparities=64,  # Search range: Must be a positive multiple of 16
    #     blockSize=15  # Linear size of the sliding window (w). Must be an ODD number >= 5
    # )
    #
    # # 3. Fine-tune BM-specific pre/post-filtering parameters for robustness
    # stereo_bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)  # Standard gradient pre-filter
    # stereo_bm.setPreFilterSize(9)  # Sobel window size
    # stereo_bm.setPreFilterCap(31)  # Truncation cap for pixel values
    #
    # stereo_bm.setMinDisparity(0)
    # stereo_bm.setUniquenessRatio(15)  # Threshold to reject ambiguous matches
    # stereo_bm.setTextureThreshold(10)  # Rejects matches in dead flat/untextured areas
    # stereo_bm.setSpeckleWindowSize(100)  # Group size for blob/noise reduction
    # stereo_bm.setSpeckleRange(32)
    #
    # print(f"Left: {left_path}, Right: {right_path}")
    # left_frame = cv2.imread(left_path)  # Shape: [H, W, 3]
    # right_frame = cv2.imread(right_path)    # Shape: [H, W, 3]
    # gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY) if len(left_frame.shape) == 3 else left_frame
    # gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY) if len(right_frame.shape) == 3 else right_frame
    # # 3. Compute the disparity map
    # # OpenCV returns a 16-bit signed integer image where values are multiplied by 16
    # disparity_16s_bm = stereo_bm.compute(gray_left, gray_right)
    # disparity_16s = stereo.compute(left_frame, right_frame)
    # # 4. Convert the disparity map to float32 and scale it back to true pixel values
    # disparity_bm = disparity_16s_bm.astype('float32') / 16.0
    # disparity = disparity_16s.astype('float32') / 16.0
    # # disparity = disparity_16s.astype('float32')
    # # 5. Visualize the disparity map
    # max_disp_value = disparity.max()
    # disparity[disparity==-1] = 0  # Set invalid disparities to 0 for visualization
    # disparity_bm[disparity_bm==-1] = 0  # Set invalid disparities to 0 for visualization
    # ImageTensor(disparity/70).RGB('turbo').save("/home/godeta/Documents/These/3 - Spatial Registration/images/",
    #                                                                                                 name = "disparity_map")
    # ImageTensor(disparity_bm/70).RGB('turbo').save(
    #     "/home/godeta/Documents/These/3 - Spatial Registration/images/",
    #     name="disparity_map_bm"
    #     )
    # img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    # img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    # height, width = img_left.shape
    # anaglyph = np.zeros((height, width, 3), dtype=np.uint8)
    #
    # # 3. Assign channels (OpenCV uses BGR format instead of RGB)
    # anaglyph[:, :, 2] = img_left  # Red channel gets Left image
    # anaglyph[:, :, 1] = img_right  # Green channel gets Right image
    # anaglyph[:, :, 0] = img_right  # Blue channel gets Right image
    #
    # # 4. Save the final red-cyan composite asset
    # cv2.imwrite('/home/godeta/Documents/These/3 - Spatial Registration/images/stereo_anaglyph.png', anaglyph)

    # 1. Match the exact parameters used in your main image plot
    # vmin = 0
    # vmax = 70
    # colormap_name = 'jet'
    #
    # # 2. Create a narrow, small figure dedicated to the colorbar
    # fig, ax = plt.subplots(figsize=(0.25, 5))
    #
    # # 3. Define the scalar mapping based on your disparity range
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # mappable = cm.ScalarMappable(norm=norm, cmap=colormap_name)
    #
    # # 4. Draw the colorbar onto the dedicated axis
    # cbar = fig.colorbar(mappable, cax=ax, orientation='vertical')
    #
    # # 5. Format the label and ticks to make it clean for your thesis
    # cbar.set_label('Disparity Scale (pixels)', rotation=270, labelpad=15, fontsize=8)
    # ax.tick_params(labelsize=10)
    #
    # # 6. Save just the colorbar asset with a tight crop
    # # Using a high DPI (e.g., 300) ensures it looks crisp in a LaTeX document
    # plt.savefig('/home/godeta/Documents/These/3 - Spatial Registration/images/disparity_scale_only.png', dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()