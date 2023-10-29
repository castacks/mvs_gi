
import copy
import cv2
import numpy as np
import torch

def add_string_to_image(img, string, h_ratio=0.05):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 1
    font_color = (255, 255, 255)

    # Calculate the text size.
    text_size = cv2.getTextSize(string, font, font_scale, font_thickness)[0]

    # True scale.
    H = img.shape[0]
    h_text = np.ceil(H * h_ratio)
    font_scale = h_text / text_size[1]

    # Calculate the text size again.
    text_size = cv2.getTextSize(string, font, font_scale, font_thickness)[0]

    # Set the margins.
    margin_vertical = int( np.ceil( 0.5 * text_size[1] ) )

    # Copy and augment the image.
    img = copy.deepcopy(img)

    # Agument.
    cv2.putText(img, 
                string, 
                (0, text_size[1] + margin_vertical), 
                font, font_scale, font_color, font_thickness)

    return img

def tensor_2_cv2_img(tensor):
    '''Assuming that tensor is in BCHW format.
    '''
    img = ( torch.clamp(tensor.detach(), 0, 1) * 255 ).permute( 0, 2, 3, 1).to('cpu').numpy()
    return img.astype(np.uint8)

def multi_view_tensor_2_cv2_img(tensor):
    '''Assuming that tensor has a shape of [B, X, C, H, W] with X being the number of views.
    '''
    # Get the number of views.
    B = tensor.shape[0]

    # Put everything in the batch dimension.
    tensor = tensor.reshape(-1, *tensor.shape[2:])

    # Convert to cv2 images.
    imgs = tensor_2_cv2_img(tensor)

    # Split the according to the original batch dimension.
    multi_view_imgs = np.split(imgs, B, axis=0)

    return multi_view_imgs

def multi_view_tensor_2_stacked_cv2_img(tensor):
    '''Assuming that tensor has a shape of [B, X, C, H, W] with X being the number of views.
    '''

    # Get a list of [ X, H, W, C ] arrays.
    multi_view_imgs = multi_view_tensor_2_cv2_img(tensor)

    # Reshape every element to [ XH, W, C ]
    W, C = multi_view_imgs[0].shape[2:]
    return [ imgs.reshape( (-1, W, C) ) for imgs in multi_view_imgs ]

def render_visualization(inv_dist, min_val=None, max_val=None):
    min_val = min_val if min_val is not None else inv_dist.min()
    max_val = max_val if max_val is not None else inv_dist.max()
    inv_dist = (inv_dist.astype(np.float32) - min_val) / ( max_val - min_val )
    inv_dist = np.clip(inv_dist, 0, 1) * 255
    inv_dist = cv2.applyColorMap(inv_dist.astype(np.uint8), cv2.COLORMAP_JET)
    return inv_dist

def render_with_gt(gt_inv_dist_np, 
                   pred_ori_np, 
                   stacked_input_resized_np,
                   inv_dist_min=None,
                   inv_dist_max=None):
    '''All inputs are NumPy arrays. gt_inv_dist and pred_ori_np are 2D arrays.
    '''

    # Get the value range for visualization.
    min_val = gt_inv_dist_np.min() if inv_dist_min is None else inv_dist_min
    max_val = gt_inv_dist_np.max() if inv_dist_max is None else inv_dist_max

    # Create a visualization of the ground truth and the prediction.
    gt_vis = render_visualization(gt_inv_dist_np, min_val, max_val)
    pd_vis = render_visualization(pred_ori_np, min_val, max_val)

    # Compute the difference.
    diff = np.abs(gt_inv_dist_np - pred_ori_np)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    # Augment the prediction visualization.
    pd_vis_aug = add_string_to_image(pd_vis, 
                    f'Mean: {diff_mean:.3f}, Std: {diff_std:.3f}' )

    # Concatenate things together.
    stacked = np.concatenate(
        (pd_vis_aug, gt_vis, stacked_input_resized_np), axis=0 )
    
    return stacked