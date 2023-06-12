import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import transformations as tf

calibfile = 'calib.txt'

with open(calibfile , 'r') as f:
    cali_data = f.readlines()
    print(cali_data)

left_intrinsic = np.zeros((3,3))
distortion_left = np.zeros(5)
distortion_right = np.zeros(5)
left_translation = np.zeros(3)
right_translation = np.zeros(3)
rectified_value = False  

for i in range(len(cali_data)):
    line = cali_data[i]
    if line.startswith("K_02"):
        cali_values = line.split()[1:]
        left_intrinsic = np.array(cali_values[:9]).reshape((3,3)).astype(float)
 
for i in range(len(cali_data)):
    line = cali_data[i]
    if line.startswith("T_02:"):
        cali_values = line.split()[1:]
        left_translation = np.array(cali_values[:3]).reshape(3).astype(float)

for i in range(len(cali_data)):
    line = cali_data[i]
    if line.startswith("T_03:"):
        cali_values = line.split()[1:]
        right_translation = np.array(cali_values[:3]).reshape(3).astype(float)

datasetpath = 'testing'
print(right_translation)
imagefile_folder = os.path.join(datasetpath , 'image_2' )
image_files = sorted(os.listdir(imagefile_folder))

window_size = 3
min_disparity = 0
max_disparity  = 16
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity , numDisparities=max_disparity , blockSize = window_size)
#stereo.setCameraParameters(left_matrix , right_matrix , distortion_left  ,distortion_right)


def depth_mapping(left_disparity_map, left_intrinsic, left_translation, right_translation, rectified=rectified_value):
    '''

    :params left_disparity_map: disparity map of left camera
    :params left_intrinsic: intrinsic matrix for left camera
    :params left_translation: translation vector for left camera
    :params right_translation: translation vector for right camera

    '''
    # Focal length of x axis for left camera

    focal_length = left_intrinsic[0][0]

    # Calculate baseline of stereo pair
    baseline = abs(right_translation[0] - left_translation[0]) 
    
    # Avoid instability and division by zero
    left_disparity_map[left_disparity_map == 0.0] = 0.1
    left_disparity_map[left_disparity_map == -1.0] = 0.1

    # depth_map = f * b/d
    depth_map = np.ones(left_disparity_map.shape)
    depth_map = (focal_length * baseline) / left_disparity_map

    return depth_map

rgb_value = True

def disparity_mapping(left_image, right_image, rgb=rgb_value):
    '''
    Takes a stereo pair of images from the sequence and
    computes the disparity map for the left image.

    :params left_image: image from left camera
    :params right_image: image from right camera

    '''

    if rgb:
        num_channels = 3
    else:
        num_channels = 1

    # Empirical values collected from a OpenCV website
    num_disparities = 6*16
    block_size = 3

    # Using SGBM matcher(Hirschmuller algorithm)
    matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                    minDisparity=0,
                                    blockSize=block_size,
                                    P1=8 * num_channels * block_size ** 2,
                                    P2=32 * num_channels * block_size ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY 
                                    )
    if rgb:
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Disparity map
    left_image_disparity_map = matcher.compute(
        left_image, right_image).astype(np.float32)/16

    return left_image_disparity_map

detector_name = 'sift'


def feature_extractor(image, detector=detector_name, mask=None):
    """
    provide keypoints and descriptors

    :params image: image from the dataset

    """
    if detector == 'sift':
        create_detector = cv2.SIFT_create()
    elif detector == 'orb':
        create_detector = cv2.ORB_create()

    keypoints, descriptors = create_detector.detectAndCompute(image, mask)

    return keypoints, descriptors

def feature_matching(first_descriptor, second_descriptor, detector=detector_name, k=2,  distance_threshold=1.0):
    """
    Match features between two images

    """

    if detector == 'sift':
        feature_matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
        feature_matcher = cv2.BFMatcher_create(
            cv2.NORM_L2, crossCheck=False)
    matches = feature_matcher.knnMatch(
        first_descriptor, second_descriptor, k=k)

    # Filtering out the weak features
    filtered_matches = []
    for match1, match2 in matches:
        if match1.distance <= distance_threshold * match2.distance:
            filtered_matches.append(match1)

    return filtered_matches
max_depth_value =  3000

def motion_estimation(matches, firstImage_keypoints, secondImage_keypoints, intrinsic_matrix, depth, max_depth=max_depth_value):
    """
    Estimating motion of the left camera from sequential imgaes 

    """
    rotation_matrix = np.eye(3)
    translation_vector = np.zeros((3, 1))

    # Only considering keypoints that are matched for two sequential frames
    image1_points = np.float32(
        [firstImage_keypoints[m.queryIdx].pt for m in matches])
    image2_points = np.float32(
        [secondImage_keypoints[m.trainIdx].pt for m in matches])

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    points_3D = np.zeros((0, 3))
    outliers = []

    # Extract depth information to build 3D positions
    for indices, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]

        # We will not consider depth greater than max_depth
        if z > max_depth:
            outliers.append(indices)
            continue

        # Using z we can find the x,y points in 3D coordinate using the formula
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy

        # Stacking all the 3D (x,y,z) points
        points_3D = np.vstack([points_3D, np.array([x, y, z])])

    # Deleting the false depth points
    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    # Apply Ransac Algorithm to remove outliers
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    rotation_matrix = cv2.Rodrigues(rvec)[0]

    return rotation_matrix, translation_vector, image1_points, image2_points

#using open3d to plot th e 3d pose of marker
tglobal = []
rglobal = []

def createcoordinateframe(position , rotation, size=5):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = size , origin = [ 0 , 0 ,0])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3:4] = position
    mesh_frame.transform(T)
    return mesh_frame


def draw_open3d():
    line_sets = o3d.geometry.LineSet()
    ll = []
    previous_pose = None
    for i, T_WC in enumerate(tglobal):
        T_WC=T_WC.T
        if previous_pose is not None:
            points = o3d.utility.Vector3dVector([previous_pose[0,:], T_WC[0,:]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets+=line
            ll.append(line)
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        
        previous_pose = T_WC
    frames = o3d.geometry.TriangleMesh()
    ff = []
    for rvec , tvec in zip(rglobal , tglobal):
        #R, _ = cv2.Rodrigues(rvec)
        frame = createcoordinateframe(tvec , rvec)
        frames+=frame
        ff.append(frames)

    
    #ok = o3d.visualization.Visualizer.create_window(window_name='Open3Dnew', width=1920, height=1080, left=50, top=50, visible=True)

    geometries = []
    geometries += ll
    geometries += ff
    o3d.io.write_triangle_mesh("open3d.ply", frames)
    o3d.io.write_line_set("open3d.ply" , line_sets , True , False)

    

    o3d.visualization.draw_geometries(geometries ,  window_name='Open3D', width=1000, height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

def press(event):
    #print('press', event.key)
    if event.key == 'p':
        draw_open3d()
pre = np.eye(4)
canvasH = 1200
canvasW = 1200
traj = np.zeros((canvasH,canvasW,3), dtype=np.uint8)
startFrame = int(1)
endFrame =  int(800)
for i in range(190):
    
    filename = image_files[i]
    if 'png' in filename:
        #print(filename)
        image02path = os.path.join(imagefile_folder , filename)
        image03path = os.path.join(datasetpath , 'image_3'  , filename)

        image_02 = cv2.imread(image02path)
        image_03 = cv2.imread(image03path)
        frm = image_02
        cv2.imshow('image', image_02)
        disparity_map = disparity_mapping(image_02 , image_03 , rgb=rgb_value)
        #print(disparity_map)
        depth_map = depth_mapping(disparity_map , left_intrinsic  , left_translation , right_translation , rectified_value)
        #print(depth_map)

        keypoints1 , descriptors1  = feature_extractor(image_02)
        filename = image_files[i+1]
        image02path = os.path.join(imagefile_folder , filename)
        image02_next = cv2.imread(image02path)
        keypoints2 , descriptors2  = feature_extractor(image02_next)
        matches = feature_matching(descriptors1 , descriptors2) 
        pose = np.eye(4)

        rotation_matrix, translation_vector, _, _ = motion_estimation(matches, keypoints1 , keypoints2, left_intrinsic , depth_map)
        pose[:3,:3] = rotation_matrix
        pose[:3,3:4] = translation_vector
        
        pose = np.dot(pre,np.linalg.inv(pose))
        #pose = np.dot(pre, pose)
       
        pre = pose
        eular_angles = tf.euler_from_matrix(pose , 'rxyz')
        eular_angles = np.degrees(eular_angles)
        print(eular_angles)
        
        print(pose)
        translation = pose[:3,3:4]
        rotation_matrix = pose[:3,:3]
        tglobal.append(translation)
        rglobal.append(rotation_matrix)
        #cv2.waitKey(1)
        canvasWCorr = 290
        canvasHCorr = 200
        draw_x, draw_y = int(translation[0])+canvasWCorr, int(translation[2])+canvasHCorr
        # grndPose = groundTruthTraj[frm].strip().split()
        # grndX = int(float(grndPose[3])) + canvasWCorr
        # grndY = int(float(grndPose[11])) + canvasHCorr

        #cv2.circle(traj, (grndX,grndY), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(translation[0],translation[1],translation[2])
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        cv2.circle(traj, (draw_x, draw_y), 1, (255, 255, 255), 1)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

        #draw_open3d()
        # plt.imshow(depth_map)
        # plt.colorbar()
        # plt.show()
        #draw_open3d()
        # depth_map = cv2.normalize(depth_map , None , alpha = 0 , beta = 255 , norm_type = cv2.NORM_MINMAX , dtype = cv2.CV_8U)
        # cv2.imshow('Disparity map' , depth_map)
        plt.gcf().canvas.mpl_connect('key_press_event', press)

        # if plt.waitforbuttonpress(0):
        #     key = plt.gcf().canvas.key_press_event
        #     if key == 'q':
        #         draw_open3d()
        # 
        #         break        

draw_open3d()
