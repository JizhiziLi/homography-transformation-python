import cv2
import numpy as np
import math
from config import *
from util import *
from tqdm import tqdm


def get_homography_shift_scale_rotate(root_path):

    print(f'--------------\n====> Start processing: shift scale and rotate')

    query_path = root_path+"crop.jpg"
    train_path = root_path+'original.jpg'
    query_img = cv2.imread(query_path,0)
    train_img = cv2.imread(train_path,0)
    MIN_MATCH_COUNT=10

    orb = cv2.ORB_create(
    nfeatures = 500,
    scaleFactor = 1.2,
    nlevels = 8,
    edgeThreshold = 31,
    firstLevel = 0,
    WTA_K = 2,
    scoreType = cv2.ORB_HARRIS_SCORE,
    patchSize = 31,
    fastThreshold = 20 )

    ##### get keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(query_img, None)
    kp2, des2 = orb.detectAndCompute(train_img, None)

    ##### use brutal force and knn with k=2 to get initial matches
    bf = cv2.BFMatcher()
    initial_matches = bf.knnMatch(des1, des2, k=2)
    ##### Apply ratio test ref. https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    good_matches = []
    for m, n in initial_matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    print(f'====> Initial matches length: {len(initial_matches)}')
    print(f'====> Filtered matches length: {len(good_matches)}')

    ##### Draw keypoints and matches out
    query_keypoints = cv2.drawKeypoints(query_img, kp1, None)
    cv2.imwrite(root_path+'query_keypoints.jpg', query_keypoints)
    train_keypoints = cv2.drawKeypoints(train_img, kp2, None)
    cv2.imwrite(root_path+'train_keypoints.jpg', train_keypoints)

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        H_ori_2_crop, mask_ori_2_crop = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = query_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,H)
        train_img = cv2.polylines(train_img,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    # draw matches in green color, draw only inliers
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = matchesMask,flags = 2)
    result_ref = cv2.drawMatches(query_img,kp1,train_img,kp2,good_matches,None,**draw_params)
    cv2.imwrite(root_path+'result_ref.jpg',result_ref)

    # query_path = root_path+"crop.jpg"
    original_color_path = root_path+'original.jpg'
    original_color_img = cv2.imread(original_color_path)
    crop_color_path = root_path+'crop.jpg'
    crop_color_img = cv2.imread(crop_color_path)
    result_crop = cv2.warpPerspective(original_color_img, H_ori_2_crop, (crop_color_img.shape[1], crop_color_img.shape[0]))
    cv2.imwrite(root_path+'result_crop.jpg',result_crop)

    print(H_ori_2_crop)

    result_read_again = cv2.imread(root_path+'result_crop.jpg')

    difference = abs(crop_color_img-result_read_again)
    cv2.imwrite(root_path+'difference.jpg',difference)
    print(difference.shape)
    print(np.sum(difference)/(difference.shape[0]*difference.shape[1]*difference.shape[2]))
    




def get_homography_shift_scale(root_path):

    print(f'--------------\n====> Start processing: shift and scale')

    query_path = root_path+"crop.jpg"
    train_path = root_path+'original.jpg'
    original_color_path = root_path+'original.jpg'
    crop_color_path = root_path+'crop.jpg'
    crop_color_img = cv2.imread(crop_color_path)
    original_color_img = cv2.imread(original_color_path)
    query_img = cv2.imread(query_path,0)
    train_img = cv2.imread(train_path,0)


    orb = cv2.ORB_create(
    nfeatures = 300,
    scaleFactor = 1.2,
    nlevels = 8,
    edgeThreshold = 31,
    firstLevel = 0,
    WTA_K = 2,
    scoreType = cv2.ORB_HARRIS_SCORE,
    patchSize = 31,
    fastThreshold = 20 )

    ##### get keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(query_img, None)
    kp2, des2 = orb.detectAndCompute(train_img, None)

    ##### use brutal force and knn with k=2 to get initial matches
    bf = cv2.BFMatcher()
    initial_matches = bf.knnMatch(des1, des2, k=2)
    ##### Apply ratio test ref. https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    good_matches = []
    for m, n in initial_matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    print(f'====> Initial matches length: {len(initial_matches)}')
    print(f'====> Filtered matches length: {len(good_matches)}')

    
    if len(good_matches) > MIN_MATCH_PAIRS:
        query_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
        train_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
        ratio_nearest = get_ratio(query_pts, train_pts, 'nearby')
        ratio_farest = get_ratio(query_pts, train_pts, 'farest')
        print(f'=====> Ratio nearest: {ratio_nearest}')
        print(f'=====> Ratio farest: {ratio_farest}')

        ratio = ratio_farest
        query_pts = query_pts/ratio
        height_real_crop = query_img.shape[0]/ratio_farest
        weight_real_crop = query_img.shape[1]/ratio_farest
        delta_metric = train_pts-query_pts

        start_point = [int(np.floor(np.median(delta_metric[:,0]))), int(np.floor(np.median(delta_metric[:,1])))]
        end_point = [int(np.floor(start_point[0]+weight_real_crop)),int(np.floor(start_point[1]+height_real_crop))]

        print(f'====> Start point: {start_point}')
        print(f'====> End point: {end_point}')
        print(f'====> Weight/Height: {end_point[0]-start_point[0]}/{end_point[1]-start_point[1]}')
        

        ######## post processing
        start_point_search_candidate = generate_point_search_area_array(start_point, SEARCH_RADIUS)
        end_point_search_candidate = generate_point_search_area_array(end_point, SEARCH_RADIUS)
        print(f'====> Search within {len(start_point_search_candidate)} start points and {len(end_point_search_candidate)} end points.')

        smallest_mse = 10000
        chosen_start_point = []
        chosen_end_point = []
        
        for start_point_candidate in tqdm(start_point_search_candidate):
            for end_point_candidate in end_point_search_candidate:

                crop_generate, mse = crop_resize_and_calculate(original_color_img, start_point_candidate, end_point_candidate, crop_color_img)
                if mse<smallest_mse:
                    chosen_start_point = start_point_candidate
                    chosen_end_point = end_point_candidate
                    smallest_mse = mse

                
        print(f'=====> Smallest mse: {smallest_mse}')
        print(f'=====> Chosen start point: {chosen_start_point}')
        print(f'=====> Chosen end point: {chosen_end_point}')
        print(f'=====> Chosen weight/height: {chosen_end_point[0]-start_point_candidate[0]}/{chosen_end_point[1]-start_point_candidate[1]}')
        print(f'=====> Saving final result to result.jpg')
        crop_generate, mse = crop_resize_and_calculate(original_color_img, chosen_start_point, chosen_end_point, crop_color_img)
        cv2.imwrite(root_path+'result.jpg',crop_generate)

        ## draw difference if needed
        # old_crop = cv2.imread(query_path)
        # new_crop = cv2.imread(root_path+'crop_generate.jpg')
        # difference = abs(new_crop-old_crop)
        # cv2.imwrite(root_path+'difference.jpg',difference)


        print(f'=====> Saving keypoints and features matching to *_keypoints.jpg, features_matching.jpg')
        #### Draw keypoints and feature matching
        query_keypoints = cv2.drawKeypoints(crop_color_img, kp1, None)
        cv2.imwrite(root_path+'query_keypoints.jpg', query_keypoints)
        train_keypoints = cv2.drawKeypoints(original_color_img, kp2, None)
        cv2.imwrite(root_path+'train_keypoints.jpg', train_keypoints)

        draw_params = dict(matchColor = (0,255,0), singlePointColor = None,flags = 2)
        original_color_img = cv2.rectangle(original_color_img, (chosen_start_point[0], chosen_start_point[1]), (chosen_end_point[0], chosen_end_point[1]), (51,255,255), 3)
        features_match = cv2.drawMatches(crop_color_img,kp1,original_color_img,kp2,good_matches,None,**draw_params)
        cv2.imwrite(root_path+'features_matching.jpg',features_match)



if __name__ == '__main__':

    get_homography_shift_scale(ROOT_PATH)









