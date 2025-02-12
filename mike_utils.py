import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


theta_obb_format_len_short = 5
theta_obb_format_len_combined = 9
box_list_size = 5
colors_cam = [(100.0,100.0,0.0),    # Unknown
              (0.0,100.0,100.0),    # Vehicle
              (100.0, 0.0, 100.0),  # Pedestrian
              (255.0,0.0,0.0),      # Sign
              (25.0,100.0,25.0)]    # Cyclist

# Get lidar boxes and convert to image space

def rotate_lidar_boxes(lidar_boxes):
        return np.transpose(np.array([
            lidar_boxes[:, 0],
            lidar_boxes[:, 2],
            lidar_boxes[:, 1],
            lidar_boxes[:, 4],
            lidar_boxes[:, 3],
            lidar_boxes[:, 5]]))

def lidar_boxes_to_image_space(lidar_box, configs, rotated=False):
    if rotated:
        flip_y=-1.0
    else:
        flip_y=1.0
    POINTCLOUD_X_INDEX = 1
    POINTCLOUD_Y_INDEX = 2
    lidar_boxes = np.transpose(np.array([
        np.array(lidar_box.type),
        ((np.array(lidar_box.box.center.y) * flip_y) - configs['lim_y'][0]) * configs['bev_width'] / (configs['lim_y'][1] - configs['lim_y'][0]), 
        (np.array(lidar_box.box.center.x) - configs['lim_x'][0]) * configs['bev_height'] / (configs['lim_x'][1] - configs['lim_x'][0]), 
        np.array(lidar_box.box.size.y) * configs['bev_width'] / (configs['lim_y'][1] - configs['lim_y'][0]), 
        np.array(lidar_box.box.size.x) * configs['bev_height'] / (configs['lim_x'][1] - configs['lim_x'][0]) , 
        np.array(lidar_box.box.heading) * -180.0 / np.pi]))
    
    # # Remove labels outside the image
    # mask = np.where((lidar_boxes[:, POINTCLOUD_X_INDEX] >= 0) & (lidar_boxes[:, POINTCLOUD_X_INDEX] <= configs['bev_width']) &
    #                 (lidar_boxes[:, POINTCLOUD_Y_INDEX] >= 0) & (lidar_boxes[:, POINTCLOUD_Y_INDEX] <= configs['bev_height']))# &
    #                 #(lidar_pcl[:, POINTCLOUD_Z_INDEX] >= configs['lim_z'][0]) & (lidar_pcl[:, POINTCLOUD_Z_INDEX] <= configs['lim_z'][1]))
    # lidar_boxes = lidar_boxes[mask]
    
    # Prune nan
    # lidar_boxes = lidar_boxes[~np.isnan(lidar_boxes).any(axis=1)]

    if rotated:
        lidar_boxes = rotate_lidar_boxes(lidar_boxes)
    
    return lidar_boxes

def camera_boxes_to_image_space(camera_box, normalized=None):
    if normalized is None:
        normalized = (1.0, 1.0)
    camera_boxes = np.transpose(np.array([
        np.array(camera_box.type),
        np.array(camera_box.box.center.x) / normalized[1], 
        np.array(camera_box.box.center.y) / normalized[0], 
        np.array(camera_box.box.size.x) / normalized[1], 
        np.array(camera_box.box.size.y) / normalized[0]]))
    # Prune nan
    # return camera_boxes[~np.isnan(camera_boxes).any(axis=1)]
    return camera_boxes

def resize_boxes(boxes, shape):
    return np.transpose(np.array([
        boxes[:,0],
        boxes[:,1] * shape[1], 
        boxes[:,2] * shape[0], 
        boxes[:,3] * shape[1], 
        boxes[:,4] * shape[0]]))

def normalize_boxes(boxes, img_shape):
    if (boxes.shape[1] <= 5):
        return np.transpose(np.stack(np.array([
        boxes[:,0],
        boxes[:,1] / img_shape[1],
        boxes[:,2] / img_shape[0],
        boxes[:,3] / img_shape[1],
        boxes[:,4] / img_shape[0]])))
    else: 
        return np.transpose(np.stack(np.array([
        boxes[:,0],
        boxes[:,1] / img_shape[1],
        boxes[:,2] / img_shape[0],
        boxes[:,3] / img_shape[1],
        boxes[:,4] / img_shape[0],
        boxes[:,5]])))

def remove_labels_outside_bev(lidar_boxes, camera_boxes, configs):
    POINTCLOUD_X_INDEX = 1
    POINTCLOUD_Y_INDEX = 2
    # Remove labels outside the image
    mask = np.where((lidar_boxes[:, POINTCLOUD_X_INDEX] >= 0) & (lidar_boxes[:, POINTCLOUD_X_INDEX] <= configs['bev_width']) &
                    (lidar_boxes[:, POINTCLOUD_Y_INDEX] >= 0) & (lidar_boxes[:, POINTCLOUD_Y_INDEX] <= configs['bev_height']))# &
                    #(lidar_pcl[:, POINTCLOUD_Z_INDEX] >= configs['lim_z'][0]) & (lidar_pcl[:, POINTCLOUD_Z_INDEX] <= configs['lim_z'][1]))
    lidar_boxes = lidar_boxes[mask]
    camera_boxes = camera_boxes[mask]
    
    # lidar_boxes = lidar_boxes[~np.isnan(lidar_boxes).any(axis=1)]
    # camera_boxes = camera_boxes[~np.isnan(camera_boxes).any(axis=1)]
    return lidar_boxes, camera_boxes

def read_boxes(filename, img):
    boxes = []
    with open(filename, 'r') as coco8_lbs_file:
        for line in coco8_lbs_file:
            line = [float(i) for i in line.split()]
            #  not pd.isnull(line[0])
            if len(line) >= theta_obb_format_len_short:
                line[1] = line[1] * img.shape[1]
                line[2] = line[2] * img.shape[0]
                line[3] = line[3] * img.shape[1]
                line[4] = line[4] * img.shape[0]
            # if line(line) >= theta_obb_format_len:
            #     line[5] = line[5]
            #     line[6] = line[6]
            #     line[7] = line[7]
            #     line[8] = line[8]
            boxes.append(line)
    return np.array(boxes)

def rotate_mtx(points, angle, center):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    p = points - center
    p = p.dot(rotation)
    # print(p.shape)
    return p + center

# Boxes must be 0-1
def crop_img_and_boxes(img, boxes, l_r_ratio=0.5):
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    delta = width - height
    left_boarder_img = int(delta * l_r_ratio)
    right_boarder_img = width - delta + left_boarder_img
    left_boarder_boxes = left_boarder_img / width
    right_boarder_boxes = right_boarder_img / width
    img_result = img[:,left_boarder_img:right_boarder_img,:]
    # mask = np.where((left_boarder_boxes < boxes[:,1]) & (boxes[:,1] < right_boarder_boxes))
    # boxes_result = boxes[mask]
    boxes[:,1] = (boxes[:,1] - left_boarder_boxes) / (right_boarder_boxes - left_boarder_boxes)
    boxes[:,3] = boxes[:,3] * width / (width - delta) 
    return (img_result, boxes)

def show_image(img, boxes):
    # img: jpg image
    # camera_box_: a box as the following list [class, cx, cy, w, h]
    
    plt.figure(figsize=(25, 20))
    
    """Display the given camera image."""
    ax = plt.subplot(*[1,1,1])
    plt.imshow(img)
    # plt.title(open_dataset.CameraName.Name.Name(camera_image_.name))
    plt.grid(False)
    plt.axis('off')

    # If the boxes data is 0-1, then expand
    if boxes[:,1].max() <= 10 and boxes[:,2].max() <= 10:
        print('Expanding 0-1 values')
        boxes = normalize_boxes(boxes, (1/img.shape[0], 1/img.shape[1]))

    nans = []
    for i, box in enumerate(boxes):
        
        if len(box) <= box_list_size:
            angle=0
        else:
            angle=(box[5] - 0.5) * 180 * 2
        #  not pd.isnull(box[0]) and
        if int(box[0]) < len(colors_cam):
            edgecolor=(np.array(colors_cam[int(box[0])])/255.0, 1.0)
            facecolor=(np.array(colors_cam[int(box[0])])/255.0, 0.1)
        else:
            nans.append(i)
            edgecolor = (0.0, 1.0, 0.0, 1.0)
            facecolor = (0.0, 1.0, 0.0, 0.1)
        # Draw the bounding box.
        rect = patches.Rectangle(
            xy=((box[1] - box[3] / 2), (box[2] - box[4] / 2)),
            width=box[3],
            height=box[4],
            angle=angle,
            rotation_point='center',
            linewidth=3,
            edgecolor=edgecolor,  # green
            facecolor=facecolor)  # opaque green
        ax.add_patch(rect)
        ax.annotate(i, ((box[1]), (box[2])), color='yellow', weight='bold', fontsize=10, ha='center', va='center')
    print(f'Indexes of nan objects: {nans}')
