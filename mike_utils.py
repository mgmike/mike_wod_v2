import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
import pandas as pd


theta_obb_format_len_short = 5
theta_obb_format_len_combined = 9
box_list_size = 5
colors_cam = [(100.0,100.0,0.0),    # Unknown
              (0.0,100.0,100.0),    # Vehicle
              (100.0, 0.0, 100.0),  # Pedestrian
              (255.0,0.0,0.0),      # Sign
              (25.0,100.0,25.0)]    # Cyclist

POINTCLOUD_X_INDEX = 3
POINTCLOUD_Y_INDEX = 4
POINTCLOUD_Z_INDEX = 5
POINTCLOUD_INTENSITY = 1

# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs, viz=False, verbose=False):
    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, POINTCLOUD_X_INDEX] >= configs['lim_x'][0]) & (lidar_pcl[:, POINTCLOUD_X_INDEX] <= configs['lim_x'][1]) &
                    (lidar_pcl[:, POINTCLOUD_Y_INDEX] >= configs['lim_y'][0]) & (lidar_pcl[:, POINTCLOUD_Y_INDEX] <= configs['lim_y'][1]))# &
                    #(lidar_pcl[:, POINTCLOUD_Z_INDEX] >= configs['lim_z'][0]) & (lidar_pcl[:, POINTCLOUD_Z_INDEX] <= configs['lim_z'][1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, POINTCLOUD_Z_INDEX] = lidar_pcl[:, POINTCLOUD_Z_INDEX] - configs['lim_z'][0]  

    if verbose:
        print(lidar_pcl[0,:])
        print('Min and max height, %f, %f' %(np.min(lidar_pcl[:,2]), np.max(lidar_pcl[:,2])))

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)

    ## step 1 : compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    delta_x_rw_meters = configs['lim_x'][1] - configs['lim_x'][0]
    delta_y_rw_meters = configs['lim_y'][1] - configs['lim_y'][0]
    meters_pixel_x = delta_x_rw_meters / configs['bev_height']
    meters_pixel_y = delta_y_rw_meters / configs['bev_width']

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates  
    lidar_pcl_copy = np.copy(lidar_pcl)
    lidar_pcl_copy[:, POINTCLOUD_X_INDEX] = np.int_(np.floor(lidar_pcl_copy[:, POINTCLOUD_X_INDEX] / meters_pixel_x))  

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_copy[:, POINTCLOUD_Y_INDEX] = np.int_(np.floor(lidar_pcl_copy[:, POINTCLOUD_Y_INDEX] / meters_pixel_y) + (configs['bev_width'] + 1) / 2)

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # if viz:
    #     show_pcl(lidar_pcl_copy)
   
   
    # Compute intensity layer of the BEV map

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs['bev_height'] + 1, configs['bev_width'] + 1))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    lidar_pcl_copy[lidar_pcl_copy[:, POINTCLOUD_INTENSITY] > 1.0, POINTCLOUD_INTENSITY] = 1.0
    index_vector_int = np.lexsort((-lidar_pcl_copy[:, POINTCLOUD_INTENSITY], lidar_pcl_copy[:, POINTCLOUD_Y_INDEX], lidar_pcl_copy[:, POINTCLOUD_X_INDEX]))
    lidar_pcl_top = lidar_pcl_copy[index_vector_int]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, idx_int_unique, counts = np.unique(lidar_pcl_top[:, POINTCLOUD_X_INDEX:POINTCLOUD_Z_INDEX], return_index=True, return_inverse=False, return_counts=True, axis=0)
    lidar_pcl_top = lidar_pcl_top[idx_int_unique]

    intensity_map = np.zeros((configs['bev_height'] + 1, configs['bev_width'] + 1))
    
    ## step 4 : assign the intensity value of each unique entry in lidar_pcl_top to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensity_map[np.int_(lidar_pcl_top[:, POINTCLOUD_X_INDEX]), np.int_(lidar_pcl_top[:, POINTCLOUD_Y_INDEX])] = lidar_pcl_top[:, POINTCLOUD_INTENSITY] / (np.amax(lidar_pcl_top[:, POINTCLOUD_INTENSITY]) - np.amin(lidar_pcl_top[:, POINTCLOUD_INTENSITY]))

    # if viz:
    #     analyze({'before': lidar_pcl_top_copy, 'after': lidar_pcl_top_copy_post}, title='Intensity Distribution', nqp=False)

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_intensity = intensity_map * 255
    img_intensity = img_intensity.astype(np.uint8)
    if viz:
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows

   

    # Compute height layer of the BEV map

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs['bev_height'] + 1, configs['bev_width'] + 1))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    # _, idx_height_unique, counts = np.unique(lidar_pcl_top[:, 0:2], return_index=True, return_inverse=False, return_counts=True, axis=0)
    # lidar_pcl_hei = lidar_pcl_top[idx_height_unique]
    height_map[np.int_(lidar_pcl_top[:, POINTCLOUD_X_INDEX]), np.int_(lidar_pcl_top[:, POINTCLOUD_Y_INDEX])] = lidar_pcl_top[:, POINTCLOUD_Z_INDEX] / float(np.abs(configs['lim_z'][1] - configs['lim_z'][0]))

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    if viz:
        while (1):
            cv2.imshow('img_height', img_height)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    # lidar_pcl_cpy = []
    # lidar_pcl_top = []
    # height_map = []
    # intensity_map = []

    # Compute density layer of the BEV map


    density_map = np.zeros((configs['bev_height'] + 1, configs['bev_width'] + 1))
    _, _, counts = np.unique(lidar_pcl_copy[:, POINTCLOUD_X_INDEX:POINTCLOUD_Z_INDEX], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, POINTCLOUD_X_INDEX]), np.int_(lidar_pcl_top[:, POINTCLOUD_Y_INDEX])] = normalizedCounts

    
    if viz:
        while (1):
            cv2.imshow('density map', density_map)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows
    
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs['bev_height'], configs['bev_width']))
    bev_map[2, :, :] = density_map[:configs['bev_height'], :configs['bev_width']]  # r_map
    bev_map[1, :, :] = height_map[:configs['bev_height'], :configs['bev_width']]  # g_map
    bev_map[0, :, :] = intensity_map[:configs['bev_height'], :configs['bev_width']]  # b_map
   

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    if viz: 
        bev_map_cpy = np.zeros((configs['bev_height'], configs['bev_width'], 3))
        bev_map_cpy[:, :, 2] = density_map[:configs['bev_height'], :configs['bev_width']] /  density_map.max() # r_map
        bev_map_cpy[:, :, 1] = height_map[:configs['bev_height'], :configs['bev_width']] / height_map.max() # g_map
        bev_map_cpy[:, :, 0] = intensity_map[:configs['bev_height'], :configs['bev_width']] / intensity_map.max() # b_map
        print(bev_map_cpy.shape)
        # bev_map_cpy = np.reshape(bev_map, (bev_map.shape[1], bev_map.shape[2], bev_map.shape[0]))
        print( intensity_map.max(), height_map.max(), density_map.max())
        while (1):
            cv2.imshow('bev_map', bev_map_cpy)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows
    # bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    # input_bev_maps = bev_maps.to(configs[device], non_blocking=True).float()

    # show_bev(input_bev_maps, configs)

    return bev_maps

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
        np.array(lidar_box.box.heading) / (-2 * np.pi) + 0.5]))
    # need to convert heading from radian of range -pi - pi to 0-1
    
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

def normalize_boxes(boxes, img_shape, bias=(0,0)):
    # print(bias)
    if (boxes.shape[1] <= 5):
        return np.transpose(np.stack(np.array([
        boxes[:,0],
        boxes[:,1] / img_shape[1] + bias[1],
        boxes[:,2] / img_shape[0] + bias[0],
        boxes[:,3] / img_shape[1],
        boxes[:,4] / img_shape[0]])))
    else: 
        return np.transpose(np.stack(np.array([
        boxes[:,0],
        boxes[:,1] / img_shape[1] + bias[1],
        boxes[:,2] / img_shape[0] + bias[0],
        boxes[:,3] / img_shape[1],
        boxes[:,4] / img_shape[0],
        boxes[:,5]])))


def remove_labels_outside(boxes):
    return np.delete(boxes, np.where((boxes > 1.0)[0]), axis=0)

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
            if len(line) >= theta_obb_format_len_short and not pd.isnull(line[0]):
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


# Dont need these anymore
#########################################################################


def distance_to_pixel(box, configs):
    w_ = configs['bev_height'] / (configs['lim_x'][1] - configs['lim_x'][0])
    l_ = configs['bev_width'] / (configs['lim_y'][1] - configs['lim_y'][0])
    x1 = (box[0] - box[2] / 2 - configs['lim_x'][0]) * w_
    x2 = (box[0] + box[2] / 2 - configs['lim_x'][0]) * w_
    y1 = (box[1] - box[3] / 2 - configs['lim_y'][0]) * l_
    y2 = (box[1] + box[3] / 2 - configs['lim_y'][0]) * l_
    # points = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
    return np.array([[y1,x1],[y1,x2],[y2,x2],[y2,x1]])
    # return rotate(points, box[4])
    
    # points = np.array([x1,y1,x2,y2]).astype(int)
    # return points

def distance_to_pixel2(box, configs):
    w_ = configs['bev_height'] / (configs['lim_x'][1] - configs['lim_x'][0])
    l_ = configs['bev_width'] / (configs['lim_y'][1] - configs['lim_y'][0])
    x1 = (box[0] - box[2] / 2 - configs['lim_x'][0]) * w_
    x2 = (box[0] + box[2] / 2 - configs['lim_x'][0]) * w_
    y1 = (box[1] - box[3] / 2 - configs['lim_y'][0]) * l_
    y2 = (box[1] + box[3] / 2 - configs['lim_y'][0]) * l_
    # points = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
    # return np.array([[y1,x1],[y1,x2],[y2,x2],[y2,x1]])
    # return rotate(points, box[4])
    
    points = np.array([x1,y1,x2,y2]).astype(int)
    return points


def show_image_cv2(img, boxes, configs):
    colors = [(0,100,100),(100,100,0), (100, 0, 100), (25,25,100), (25,100,25), (100,25,25)] #  Cyclists, Vehicles, Pedestrians, Signs
    for box in boxes:
        # print("x: ", box[0], " y: ", box[1], " w: ", box[2], " l: ", box[3])
        xyxy = distance_to_pixel(box[1:], configs)
        # print("xy top left: ", xyxy[0], " xy bottom right: ", xyxy[2])
        # cv2.rectangle(bev_map_cpy, (int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[2][0]), int(xyxy[2][1])), thickness=2, color=(100,0,100))
        # xyxy2 = distance_to_pixel2(box, configs)
        # cv2.rectangle(bev_map_cpy, (int(xyxy2[1]), int(xyxy2[0])), (int(xyxy2[3]), int(xyxy2[2])), thickness=2, color=(100,0,100))
    
        
        c = ((box[1] - configs['lim_x'][0]) * configs['bev_height'] / (configs['lim_x'][1] - configs['lim_x'][0]), 
             (box[2] - configs['lim_y'][0]) * configs['bev_width'] / (configs['lim_y'][1] - configs['lim_y'][0]))
        rotated = rotate_mtx(xyxy, box[5], (c[1],c[0]))
        cv2.drawContours(img, [rotated.astype(int)], -1, thickness=2, color=colors[int(box[0])])
        # cv2.putText(bev_map, names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(100,100,0), thickness=2)
    
    tmp = 0
    while (1): 
        rotated = rotate_mtx(xyxy, tmp, (c[1],c[0]))
        # cv2.drawContours(bev_map_cpy, [rotated.astype(int)], -1, thickness=2, color=(100,100,0))
        cv2.imshow('bev_map2', img)
        tmp = tmp + 0.5
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows

# project detected bounding boxes into birds-eye view
def project_detections_into_bev(bev_map, detections, configs, color=[]):
    for row in detections:
        # extract detection
        _id, _x, _y, _z, _h, _w, _l, _yaw = row

        # convert from metric into pixel coordinates
        x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        z = _z - configs.lim_z[0]
        w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        yaw = -_yaw

        # draw object bounding box into birds-eye view
        if not color:
            color = configs.obj_colors[int(_id)]
        
        # get object corners within bev image
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw # front left
        bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw 
        bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw # rear left
        bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw # rear right
        bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw # front right
        bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
        
        # draw object as box
        corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(bev_map, [corners_int], True, color, 2)

        # draw colored line to identify object front
        corners_int = bev_corners.reshape(-1, 2)
        cv2.line(bev_map, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)


def addBoxes(results):
    for result in results:
        boxes = result.boxes
        probs = result.probs
        img = result.orig_img
        classes = boxes.cls.cpu().numpy()
        names = result.names
        # print(boxes, ', prob:')
        for i, xyxy in enumerate(boxes.xyxy.cpu().numpy()):
            # print(xyxy[0])
            cls = classes[i]

            if int(cls) in colors_cam:
                color = colors_cam[cls]
                cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), thickness=2, color=color)
                cv2.putText(img, names[int(cls)], (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=color, thickness=2)
            
    cv2.imshow('Camera and detected objects', img)
    cv2.waitKey(0) 
    return img

from waymo_open_dataset import v2
def addBoxes2(lidar, range_image_orig):
    # Utility for fixing 
    range_image = v2.perception.lidar.RangeImage(lidar.range_image_return1.values[0], lidar.range_image_return1.shape[0])
    
    print(range_image.values)
    print(type(range_image))
    print(type(range_image_orig))
    rit = range_image.tensor
    print(rit)
    print(type(rit))
    # tf.convert_to_tensor(lidar_calibration.extrinsic.transform.tolist())
    
    points = v2.perception.utils.lidar_utils.convert_range_image_to_point_cloud(range_image, lc2, keep_polar_features=True)

#########################################################################

# Boxes must be 0-1
# 
def crop_img_and_boxes(img, boxes, desired_size, l_r_ratio=0.5, t_b_ratio=0.5):
    height = img.shape[0]
    width = img.shape[1]
    if width > desired_size:
        delta = width - desired_size
        left_boarder_img = int(delta * l_r_ratio)
        right_boarder_img = width - delta + left_boarder_img
        left_boarder_boxes = left_boarder_img / width
        right_boarder_boxes = right_boarder_img / width
        # print(f'width: {width} desired: {desired_size} delta: {delta} left: {left_boarder_img} right {right_boarder_img}')
        img = img[:,left_boarder_img:right_boarder_img,:]
        # mask = np.where((left_boarder_boxes < boxes[:,1]) & (boxes[:,1] < right_boarder_boxes))
        # boxes_result = boxes[mask]
        boxes[:,1] = (boxes[:,1] - left_boarder_boxes) / (right_boarder_boxes - left_boarder_boxes)
        boxes[:,3] = boxes[:,3] * width / (width - delta) 
        # print('boxes: ', boxes[0,:])
    if height > desired_size:
        delta = height - desired_size
        top_boarder_img = int(delta * t_b_ratio)
        bot_boarder_img = height - delta + top_boarder_img
        top_boarder_boxes = top_boarder_img / height
        bot_boarder_boxes = bot_boarder_img / height
        print(top_boarder_boxes, top_boarder_boxes)
        img = img[top_boarder_img:bot_boarder_img,:,:]
        # mask = np.where((left_boarder_boxes < boxes[:,1]) & (boxes[:,1] < right_boarder_boxes))
        # boxes_result = boxes[mask]
        boxes[:,2] = (boxes[:,2] - top_boarder_boxes) / (bot_boarder_boxes - top_boarder_boxes)
        boxes[:,4] = boxes[:,4] * height / (height - delta) 
    # print(f'Converted img from {height}, {width} to {img.shape}')  
    return (img, boxes)

def show_image(img_list, boxes_list, plot_size=[1,1]):
    # img: jpg image
    # camera_box_: a box as the following list [class, cx, cy, w, h]
    
    plt.figure(figsize=(25, 25))

    if plot_size[0] == 1 and plot_size[1] == 1:
        img_list = [img_list]
        boxes_list = [boxes_list]

    for i, img in enumerate(img_list):
        boxes = boxes_list[i]
    
        """Display the given camera image."""
        ax = plt.subplot(*[plot_size[0],plot_size[1],i + 1]) #, layout="constrained"
        plt.imshow(img)
        # plt.title(open_dataset.CameraName.Name.Name(camera_image_.name))
        plt.grid(False)
        plt.axis('off')
        
        # If the boxes data is 0-1, then expand
        if len(boxes) > 0 and boxes[:,1].max() <= 10 and boxes[:,2].max() <= 10:
            # print('Expanding 0-1 values')
            ratio_x = ratio_y = 1.0
            bias_x = bias_y = 0.0
            mod = 1
            if i % 2 == 1:
                mod = mod * -1

            if i + 1 == 2:
                mod = 1
                print(mod)
            
            if img.shape[0] > img.shape[1]:
                # print("X larger than Y")
                ratio_x = img.shape[0] / img.shape[1]
                bias_x = (img.shape[0] - img.shape[1]) / 2
            elif img.shape[1] > img.shape[0]:
                # print("Y larger than X")
                ratio_y = img.shape[1] / img.shape[0]
                bias_y = (img.shape[1] - img.shape[0]) / 2
            boxes = normalize_boxes(boxes, (1/img.shape[0], 1/img.shape[1]))
            # boxes = normalize_boxes(boxes, (ratio_x/img.shape[0], ratio_y/img.shape[1]), (mod * bias_x, mod * bias_y))
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



def write_2d_array_to_file(filename, objects):
    with open(filename, 'w') as file:
        for obj in objects:
            if not pd.isnull(obj[0]):
                if obj[0] is not int:
                    line = ' '.join([str(int(obj[0]))] + [str(x) for x in obj[1:]])
                else:
                    line = ' '.join(map(str, obj))
                file.write(line + '\n')
        print(f'Saving data to {filename}')

# Save data in yolov8 format
def save_data(configs, data_path, output_format, options, lidar_calibration_df, context_index, row):
    TAG='save_data:         '
    if output_format == 'jpg' or output_format == 'png':
        img_type = output_format
    else: return    
        
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(lidar_calibration_df)
    camera_image_cmb = v2.CameraImageComponent.from_dict(row)
    lidar_cmb = v2.LiDARComponent.from_dict(row)
    lidar_box_cmb = v2.LiDARBoxComponent.from_dict(row)
    projected_lidar_box_obj = v2.ProjectedLiDARBoxComponent.from_dict(row)
    
    print(
        f'Found {len(lidar_box_cmb.key.laser_object_id)} objects on'
        f' {lidar_cmb.key.segment_context_name=} {lidar_cmb.key.frame_timestamp_micros=}'
    )

    try:
        # Get Range image of top lidar. Had to recreate a LiDARCalibrationComponent as it is formatted incorrectly from waymo
        range_image = v2.perception.lidar.RangeImage(lidar_cmb.range_image_return1.values[0], lidar_cmb.range_image_return1.shape[0])
        temp_tfm = v2.column_types.Transform
        temp_tfm.transform = lidar_calibration.extrinsic.transform.tolist()[0]
        temp_bic = v2.perception.context.BeamInclination
        temp_bic.min = lidar_calibration.beam_inclination.min
        temp_bic.max = lidar_calibration.beam_inclination.max
        temp_bic.values = lidar_calibration.beam_inclination.values.tolist()[0]
        lc2 = v2.perception.context.LiDARCalibrationComponent(lidar_calibration.key, temp_tfm, temp_bic)
        # extrinsic = tf.convert_to_tensor(lidar_calibration.extrinsic.transform)
        # TODO: Eventually should add pixel_pose and frame_pose when mulitple cameras are used
        points = v2.perception.utils.lidar_utils.convert_range_image_to_point_cloud(range_image, lc2, keep_polar_features=True)
        # Generate a birds-eye-view voxel map from pointcloud data
        points_cpu = points.numpy()
    except:
        print(TAG, ':    Error getting range image and points')
        print(TAG, ':    lidar_cmb range image type', type(lidar_cmb.range_image_return1.values))
        print(TAG, ':    lidar_cmb range image len', len(lidar_cmb.range_image_return1.values))
        print(TAG, ':    lidar_cmb range image len len', len(lidar_cmb.range_image_return1.values[0]))
        print(TAG, ':    context_index: ', context_index)
        return

    
    # build combined label
    # projected_lidar_boxes = camera_boxes_to_image_space(projected_lidar_box_obj)
    img_to_save = tf.image.decode_jpeg(camera_image_cmb.image).numpy()
    waymo_img_shape = img_to_save.shape
    projected_lidar_boxes = camera_boxes_to_image_space(projected_lidar_box_obj, waymo_img_shape)
    lidar_boxes_rotated = lidar_boxes_to_image_space(lidar_box_cmb, configs, rotated=True)
    # These need to be the size that the bev will be. They will both be cropped later.
    lidar_boxes_rotated_normalized = normalize_boxes(lidar_boxes_rotated, (configs['bev_width'], configs['bev_height']))
    filename = f'{camera_image_cmb.key.segment_context_name}:{camera_image_cmb.key.frame_timestamp_micros}'

    # Convert from list of bytes to XxYx3 array and resize
    img_cropped, img_boxes_cropped = crop_img_and_boxes(img_to_save, projected_lidar_boxes, configs['yolo_width'] * 3)
    img_cropped_small = cv2.resize(img_cropped, dsize=(configs['yolo_width'], configs['yolo_height']), interpolation=cv2.INTER_CUBIC)
    
    bev_map = bev_from_pcl(points_cpu, configs, viz=False)
    bev_map_norm = np.zeros((configs['bev_height'], configs['bev_width'], 3))
    bev_map_norm[:, :, 2] = bev_map[0,2,:configs['bev_height'], :configs['bev_width']] / bev_map[0,2,:,:].max() # r_map
    bev_map_norm[:, :, 1] = bev_map[0,1,:configs['bev_height'], :configs['bev_width']] / bev_map[0,1,:,:].max() # g_map
    bev_map_norm[:, :, 0] = bev_map[0,0,:configs['bev_height'], :configs['bev_width']] / bev_map[0,0,:,:].max() # b_map
    bev_map_norm[:,:,0] = bev_map_norm[:,:,0] * 255 / bev_map_norm[:,:,0].max()
    bev_map_norm[:,:,1] = bev_map_norm[:,:,1] * 255 / bev_map_norm[:,:,1].max()
    bev_map_norm[:,:,2] = bev_map_norm[:,:,2] * 255 / bev_map_norm[:,:,2].max()
    bev_map_rb_swapped = np.zeros((configs['bev_height'], configs['bev_width'], 3))
    bev_map_rb_swapped[:,:,0] = bev_map_norm[:,:,2]
    bev_map_rb_swapped[:,:,1] = bev_map_norm[:,:,1]
    bev_map_rb_swapped[:,:,2] = bev_map_norm[:,:,0]
    bev_map_rb_swapped = np.round(bev_map_rb_swapped).astype(np.uint8)
    bev_map_rotated = np.rot90(bev_map_rb_swapped)
    
    bev_map_rotated_cropped, lidar_boxes_rotated_normalized_cropped = crop_img_and_boxes(
        bev_map_rotated, 
        lidar_boxes_rotated_normalized, 
        configs['yolo_width'], 
        l_r_ratio=0.35)
    
    # Save cam img and lidar bev imgs
    # with open(path_img, 'wb') as jpg:
    #     jpg.write(img_to_save)
    
    viz = False
    if viz:
        print(projected_lidar_boxes[0,:])
        imgs_list1 = [img_to_save, bev_map_rotated]
        imgs_list2 = [img_cropped_small, bev_map_rotated_cropped]
        box_list1 = [projected_lidar_boxes, lidar_boxes_rotated_normalized]
        box_list2 = [img_boxes_cropped, lidar_boxes_rotated_normalized_cropped]
        show_image(imgs_list1, box_list1, [1,2])
        show_image(imgs_list2, box_list2, [1,2])

    if 'test' in options or 'testing' in options:
        data_type = 'test'
    elif 'train' in options or 'training' in options:
        data_type = 'train'
    elif 'val' in options or 'validation' in options:
        data_type = 'val'
    else:
        print('No vaild data type entered. Please add test train or val in options')
        return

    # Set label dirs. If they must be split, they are overwritten
    cam_labels_dir = 'cam'
    lidar_labels_dir = 'lidar'
    combined_labels_dir = 'combined'
    if 'split-labels' in options or 'split-boxes' in options:
        cam_labels_dir = 'cam_labels'
        lidar_labels_dir = 'lidar_labels'
        combined_labels_dir = 'cam_lidar_labels'
        
    if 'cam-only' in options:
        cam_dir = 'cam'
        path_img = os.path.join(data_path, cam_dir, data_type, f'{filename}.{img_type}')
        path_img_labels = os.path.join(data_path, cam_labels_dir, data_type, f'{filename}.txt')

        # print(((boxes_combined[:,1:] < 0.0) & (boxes_combined[:,1:] >= 1.0)))
        # Remove all boxes bigger than 1.0 or anything outside either of the frames
        img_boxes_cropped = np.delete(img_boxes_cropped, np.where((img_boxes_cropped[:,1:] >= 1.0))[0], axis=0)
        img_boxes_cropped = np.delete(img_boxes_cropped, np.where((img_boxes_cropped[:,1:-1] < 0.0))[0], axis=0)

        try:
            mpimg.imsave(path_img, img_cropped_small)
            print('img saved')
            write_2d_array_to_file(path_img_labels, img_boxes_cropped)
        except:
            print('One of the following dirs doesnt exist: ', path_img, ', ', path_img_labels)
            try: 
                print('Attempting to create: ', os.path.join(data_path, cam_dir, data_type), ', ', os.path.join(data_path, cam_labels_dir, data_type))
                os.makedirs(os.path.join(data_path, cam_dir, data_type))
                os.makedirs(os.path.join(data_path, cam_labels_dir, data_type))
                mpimg.imsave(path_img, img_cropped_small)
                write_2d_array_to_file(path_img_labels, img_boxes_cropped)
            except:
                print('That didnt work. Somethings wrong')
                
    elif 'lidar-only' in options:
        lidar_dir = 'lidar'
        path_lidar = os.path.join(data_path, lidar_dir, data_type, f'{filename}.{img_type}')
        path_lidar_labels = os.path.join(data_path, lidar_labels_dir, data_type, f'{filename}.txt')
        
        # print(((boxes_combined[:,1:] < 0.0) & (boxes_combined[:,1:] >= 1.0)))
        # Remove all boxes bigger than 1.0 or anything outside either of the frames
        lidar_boxes_rotated_normalized_cropped = np.delete(
            lidar_boxes_rotated_normalized_cropped, 
            np.where((lidar_boxes_rotated_normalized_cropped[:,1:] >= 1.0))[0], 
            axis=0)
        lidar_boxes_rotated_normalized_cropped = np.delete(
            lidar_boxes_rotated_normalized_cropped, 
            np.where((lidar_boxes_rotated_normalized_cropped[:,1:-1] < 0.0))[0], 
            axis=0)
        try:
            mpimg.imsave(path_lidar, bev_map_rotated_cropped)
            write_2d_array_to_file(path_lidar_labels, lidar_boxes_rotated_normalized_cropped)
        except:
            print('One of the following dirs doesnt exist: ', path_lidar, ', ', path_lidar_labels)
            try: 
                os.makedirs(os.path.join(data_path, lidar_dir, data_type))
                os.makedirs(os.path.join(data_path, lidar_labels_dir, data_type))
                mpimg.imsave(path_lidar, bev_map_rotated_cropped)
                write_2d_array_to_file(path_lidar_labels, lidar_boxes_rotated_normalized_cropped)
            except:
                print(TAG, 'That didnt work. Somethings wrong')
    
    # Saves camera, lidar and boxes in one directory
    else:
        cam_dir = lidar_dir = 'combined'
        path_img = os.path.join(data_path, cam_dir, data_type, f'{filename}cam.{img_type}')
        path_lidar = os.path.join(data_path, lidar_dir, data_type, f'{filename}lidar.{img_type}')
        path_combined_labels = os.path.join(data_path, combined_labels_dir, data_type, f'{filename}.txt')

        boxes_combined = np.transpose(np.stack(np.array([
            img_boxes_cropped[:,0],
            img_boxes_cropped[:,1],
            img_boxes_cropped[:,2],
            img_boxes_cropped[:,3],
            img_boxes_cropped[:,4],
            lidar_boxes_rotated_normalized_cropped[:,1],
            lidar_boxes_rotated_normalized_cropped[:,2],
            lidar_boxes_rotated_normalized_cropped[:,3],
            lidar_boxes_rotated_normalized_cropped[:,4],
            lidar_boxes_rotated_normalized_cropped[:,5]
        ])))
        
        # print(((boxes_combined[:,1:] < 0.0) & (boxes_combined[:,1:] >= 1.0)))
        # Remove all boxes bigger than 1.0 or anything outside either of the frames
        boxes_combined = np.delete(boxes_combined, np.where((boxes_combined[:,1:] >= 1.0))[0], axis=0)
        boxes_combined = np.delete(boxes_combined, np.where((boxes_combined[:,1:-1] < 0.0))[0], axis=0)

        try:
            # Save images
            mpimg.imsave(path_img, img_cropped_small)
            mpimg.imsave(path_lidar, bev_map_rotated_cropped)
            write_2d_array_to_file(path_combined_labels, boxes_combined)
        except:
            print('Directory doesnt exist. Attempting to create', path_img, ' or ', path_lidar)
            try: 
                os.makedirs(path_img)
                os.makedirs(path_lidar)
                os.makedirs(path_combined_labels)
                write_2d_array_to_file(path_combined_labels, boxes_combined)
        
                # Save images
                mpimg.imsave(path_img, img_cropped_small)
                mpimg.imsave(path_lidar, bev_map_rotated_cropped)
            except:
                print('That didnt work. Somethings wrong')