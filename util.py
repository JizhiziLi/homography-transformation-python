import numpy as np
import cv2

def generate_point_search_area_array(point_coordinator, search_area):
    point_candidate_list = []
    index = 0
    for i in range(-search_area, search_area+1):
        for j in range(-search_area, search_area+1):
            index += 1
            point_candidate_list.append([point_coordinator[0]+i,point_coordinator[1]+j])
    return point_candidate_list


def crop_resize_and_calculate(img, start_point, end_point, compare_img):
    x1 = start_point[0]
    x2 = end_point[0]
    y1 = start_point[1]
    y2 = end_point[1]
    img = img[y1:y2, x1:x2, :]
    img = cv2.resize(img,(compare_img.shape[1],compare_img.shape[0]),interpolation = cv2.INTER_CUBIC)
    mse = np.square(compare_img-img).mean()
    return img, mse



#######
## get ratio by calculating ratio of nearby/farest points, then get median
def get_ratio(src_pts, dst_pts, choice):
    ratio_list = []
    if choice == 'nearby':
        for i in range(len(src_pts)-1):
            distance_1 = np.sqrt(np.square(src_pts[i+1][1]-src_pts[i][1])+np.square(src_pts[i+1][0]-src_pts[i][0]))
            distance_2 = np.sqrt(np.square(dst_pts[i+1][1]-dst_pts[i][1])+np.square(dst_pts[i+1][0]-dst_pts[i][0]))
            if distance_2 !=0:
                ratio = distance_1/distance_2
                ratio_list.append(ratio)
    elif choice == 'farest':
        for i in range(len(src_pts)):
            distance_key1_i=[]
            for j in range(len(src_pts)):
                distance_key1_i.append(np.sqrt(np.square(src_pts[j][0]-src_pts[i][0])+np.square(src_pts[j][1]-src_pts[i][1])))
                max_index = distance_key1_i.index(max(distance_key1_i))
                distance_1 = abs(distance_key1_i[max_index])
                distance_2 = np.sqrt(np.square(dst_pts[max_index][0]-dst_pts[i][0])+np.square(dst_pts[max_index][1]-dst_pts[i][1]))
                if distance_2 != 0:
                    ratio_list.append(distance_1/distance_2)

    ratio_median = np.median(ratio_list)
    return ratio_median


