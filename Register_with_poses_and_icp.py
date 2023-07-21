import cv2
import numpy as np
import open3d as o3d 
import transformations as tf
import open3d as o3d
import copy
import glob        




p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
arucoParams = cv2.aruco.DetectorParameters_create()


matrix = np.array([[1846.21240234375 , 0.00000000e+00  ,512.0] ,
    [0.00000000e+00 ,1846.21240234375 , 512.0] ,
    [0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00]])


distortion = np.array([[0.00000000e+00 , 0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 , 0.00000000e+00]])

pre = None
firstmarker = False
main_id = None
rel_posedict = {}
iter=1


def findRelPoses(ids,corners, markerSizeInMt):
    for i in range(len(corners)):
        for j in range(len(corners)):                                                                    #iterate through all the markers visible in the frame
                poseA = np.eye(4)
                cornersM = np.reshape(corners[i] , (1,1,4,2))
                rvecA , tvecA, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)     #estimate pose of A wrt Camera       
                RA, _ = cv2.Rodrigues(rvecA[0])
                tvecAT = tvecA[0].T                                                                                      #tvec  dimension is (1,1,3)
                poseA[:3,:3] = RA
                poseA[:3,3:4] = tvecAT
                poseB = np.eye(4)
                cornersM = np.reshape(corners[j] , (1,1,4,2))
                rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)       #estimate pose of B wrt Camera
                RB, _ = cv2.Rodrigues(rvecB[0])
                tvecBT = tvecB[0].T
                poseB[:3,:3] = RB
                poseB[:3,3:4] = tvecBT
                poseA_wrt_B = np.dot(np.linalg.inv(poseB) , poseA)                                  #pose of A wrt B is equal to dot product of ( inv of B wrt Camera and A wrt camera)
                rel_posedict[(ids[i][0],ids[j][0])] = poseA_wrt_B
                rel_posedict[(ids[j][0],ids[i][0])] = np.linalg.inv(poseA_wrt_B)                    #pose of A wrt B is inverse of pose of B wrt A
                
                

                if (ids[j][0], main_id) not in rel_posedict and (ids[i][0], main_id) in rel_posedict:
                    rel_posedict[(ids[j][0],main_id)] = np.dot(rel_posedict[(ids[i][0],main_id)] , rel_posedict[(ids[j][0] , ids[i][0]) ])          #finding relative pose wrt reference marker 
                    rel_posedict[(main_id , ids[j][0])] = np.linalg.inv(rel_posedict[(ids[j][0], main_id)])                                         
            
            


def create_point_cloud(rgb_file, depth_file , extrinsic ):

    depthimg = depth_file
    
    depth_map = depthimg.copy()

    d = np.float32(depth_map)*20/65535

    


    rgbImg = rgb_file

    rgbImg = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2BGR)





    color_ = o3d.geometry.Image(rgbImg)

    depth_ = o3d.geometry.Image(d)

    
    camera_matrix = o3d.camera.PinholeCameraIntrinsic(1024 , 1024 , 1846.21240234375,1846.21240234375,512.0,512.0) 

    

 

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_,depth_,depth_scale=1,

                                                                        depth_trunc=5,convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(

        rgbd_image, camera_matrix  , extrinsic  ) 

        # Flip it, otherwise the pointcloud will be upside down

    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd



def registericp(source , target ):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4) 
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation
    source_temp = copy.deepcopy(source)
    global combinedpcd
    source_temp.transform(result_icp.transformation)
    combinedpcd =  target + source_temp
    



def callback(rgb_image , depthimage ):
    
        global combinedpcd

        corners, ids, _ = cv2.aruco.detectMarkers(rgb_image , p_dict , parameters = arucoParams)

        global firstmarker
        global main_id
        global iter
        global pre
        global rel_posedict 
        
    
        markerSizeInMt = 0.08                                                                       #size of aruco marker in meters
        if len(corners)  > 0 :

            if firstmarker is False:
                firstmarker = True
                main_id = ids[0][0]                                                                 #assigning very first aruco as reference marker
                

            
            findRelPoses(ids=ids, corners=corners , markerSizeInMt=markerSizeInMt)                    #it finds relative poses between aruco markers
           
            
            poseB = np.eye(4)
            cornersM = np.reshape(corners[0] , (1,1,4,2))                                             #estimating pose of nearest marker
            rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)
            RB, _ = cv2.Rodrigues(rvecB[0])
            tvecBT = tvecB[0].T                                                                         #tvec  dimension is (1,1,3)
            poseB[:3,:3] = RB
            poseB[:3,3:4] = tvecBT
            if (main_id,ids[0][0]) not in rel_posedict:                                     #if relative pose wrt reference marker does not exist
                return
            pose1wrtC = np.dot(poseB,rel_posedict[(main_id,ids[0][0])])                 #POSE OF REFERENCE aruco wrt camera is equal to dot product of pose of current aruco wrt camera and relative pose of reference aruco wrt current one)
            poseCamerawrt1 = np.linalg.inv(pose1wrtC)                                   #camera pose is inverse of aruco pose                                             
            if(iter==1):
                pre=poseCamerawrt1                                                   #pre  = pose of first camera wrt world cordinate system
                
           
            relpose = np.dot( np.linalg.inv( poseCamerawrt1 ) , pre)                # pose of first camera  wrt camera i th camera          
           
            pcd = create_point_cloud(rgb_image , depthimage   , relpose)            #creating point cloud with relative pose of first camera wrt current oone as extrinsics
            if iter == 1:
                combinedpcd = pcd                                                   #for first pcd
            else:
                
                registericp(  combinedpcd , pcd )                       #Coloured ICP  for trajectory optimisation
           
            iter+=1

 
            
        
if __name__ == '__main__':


    dir1 = "/home/machine_visoin/codes/Sachin/Aruco-data/object_3d_sai/images/*.jpg"
    dir2 = "/home/machine_visoin/codes/Sachin/Aruco-data/object_3d_sai/depth/*.png"

    

    imagepaths1 = sorted(glob.glob(dir1) , key=lambda x: int(x.split('_')[-1].split('.')[0]))
    imagepaths2 = sorted(glob.glob(dir2) , key=lambda x: int(x.split('_')[-1].split('.')[0]))
    


    combinedpcd = o3d.geometry.PointCloud() 
    for filename1 , filename2  in zip(imagepaths1, imagepaths2 ):

        cv_image = cv2.imread(filename1)
        depth_image = cv2.imread(filename2,-1)[:,:,0]
        
        callback(cv_image , depth_image )
       
    o3d.io.write_point_cloud("MPC_withPoseAndICP.ply", combinedpcd)
    o3d.visualization.draw_geometries([combinedpcd])
