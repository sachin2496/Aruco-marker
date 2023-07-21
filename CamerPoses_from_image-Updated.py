import cv2
import numpy as np
import open3d as o3d 
import transformations as tf
import time
import csv
import plotly.graph_objects as go

import glob


p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
arucoParams = cv2.aruco.DetectorParameters_create()

csvfilename = 'Camera_extrinsics-Newcoke.csv'

with open(csvfilename, 'w' , newline='') as file:
    writer = csv.writer(file , delimiter='\t')
    writer.writerow(['Time' , 'X' , 'Y' , 'Z' , 'qx' , 'qy' , 'qz' , 'qw' ])


#camera matrix and distortion matrix


matrix = np.array([[1846.21240234375 , 0.00000000e+00  ,512.0] ,
    [0.00000000e+00 ,1846.21240234375 , 512.0] ,
    [0.00000000e+00 , 0.00000000e+00  ,1.00000000e+00]])



distortion = np.array([[0.00000000e+00 , 0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 , 0.00000000e+00]])




 
firstmarker = False
main_id = None
rel_posedict = {}
X = []
Y = []
Z = []

def findRelPoses(ids,corners, markerSizeInMt):
    for i in range(len(corners)):
        for j in range(len(corners)):                                                           #iterate through all the markers visible in the frame
                poseA = np.eye(4)
                cornersM = np.reshape(corners[i] , (1,1,4,2))
                rvecA , tvecA, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)            #estimate pose of A wrt Camera 
                RA, _ = cv2.Rodrigues(rvecA[0])
                tvecAT = tvecA[0].T                                                              #tvec  dimension is (1,1,3)
                poseA[:3,:3] = RA
                poseA[:3,3:4] = tvecAT
                poseB = np.eye(4)
                cornersM = np.reshape(corners[j] , (1,1,4,2))
                rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)                 #estimate pose of B wrt Camera
                RB, _ = cv2.Rodrigues(rvecB[0])
                tvecBT = tvecB[0].T
                poseB[:3,:3] = RB
                poseB[:3,3:4] = tvecBT
                poseA_wrt_B = np.dot(np.linalg.inv(poseB) , poseA)                                                          #pose of A wrt B is equal to dot product of ( inv of B wrt Camera and A wrt camera)
                rel_posedict[(ids[i][0],ids[j][0])] = poseA_wrt_B                                                       
                rel_posedict[(ids[j][0],ids[i][0])] = np.linalg.inv(poseA_wrt_B)                                           #pose of A wrt B is inverse of pose of B wrt A                                   
                
                

                if (ids[j][0], main_id) not in rel_posedict and (ids[i][0], main_id) in rel_posedict:
                    rel_posedict[(ids[j][0],main_id)] = np.dot(rel_posedict[(ids[i][0],main_id)] , rel_posedict[(ids[j][0] , ids[i][0]) ])          #finding relative pose wrt reference marker 
                    rel_posedict[(main_id , ids[j][0])] = np.linalg.inv(rel_posedict[(ids[j][0], main_id)]) 
            
      
                  
def createcoordinateframe(position , rotation, size=0.2 ):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = size , origin = [ 0 , 0 ,0])
    T = np.eye(4)
    T[:3, :3] = rotation
    position  = position.T
    T[:3, 3:4] = position
    mesh_frame.transform(T)
    return mesh_frame




def plot3d(rglobal , tglobal):
    line_sets = []
    previous_pose = None
    for i, T_WC in enumerate(tglobal):
        if previous_pose is not None:
            points = o3d.utility.Vector3dVector([previous_pose[0,0,:], T_WC[0,0,:]])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets.append(line)
        
        previous_pose = T_WC
    
    frames = []
    for rvec , tvec in zip(rglobal , tglobal):
        R, _ = cv2.Rodrigues(rvec)
        tvec = tvec[0]
        frame = createcoordinateframe(tvec , R)
        frames.append(frame)


    geometries = []
    geometries += line_sets
    geometries += frames 
    vis = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries(geometries)
    vis.destroy_window()

rglobal = []
tglobal = []
iter=1


def callback_function(cv_image):
       
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image , p_dict , parameters = arucoParams)
        
        global firstmarker
        global main_id
        global rel_posedict 
        global start_time
        global X
        global Y
        global Z
        global rglobal
        global tglobal
        global pre
        global iter

        markerSizeInMt = .08                                                                                                    #size of aruco marker in meters
        if len(corners)  > 0 :

            if firstmarker is False:
                firstmarker = True
                main_id = ids[0][0]                                                                                              #assigning very first aruco as reference marker
            
            
            findRelPoses(ids=ids, corners=corners , markerSizeInMt=markerSizeInMt)                                                   #it finds relative poses between aruco markers

            
            
            
            poseB = np.eye(4)
            cornersM = np.reshape(corners[0] , (1,1,4,2))                                                                           #estimating pose of nearest marker
            rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)
            RB, _ = cv2.Rodrigues(rvecB[0])
            tvecBT = tvecB[0].T                                                                                                 #tvec  dimension is (1,1,3)
            poseB[:3,:3] = RB
            poseB[:3,3:4] = tvecBT
            if (main_id,ids[0][0]) not in rel_posedict:                                                                        #if relative pose wrt reference marker does not exist
                cv2.imshow('markedframe' , cv_image)
                cv2.waitKey(1)
                return
            pose1wrtC = np.dot(poseB,rel_posedict[(main_id,ids[0][0])])                                                     #POSE OF REFERENCE aruco wrt camera is equal to dot product of pose of current aruco wrt camera and relative pose of reference aruco wrt current one)
            poseCamerawrt1 = np.linalg.inv(pose1wrtC)                                                                       #camera pose is inverse of aruco pose
            
            curtvec = poseCamerawrt1[:3 , 3:4]
            curtvec = curtvec.T
            curtvec = np.reshape(curtvec , (1,1,3))
            R = poseCamerawrt1[:3 , :3]
            
            ntvec = curtvec
            nrvec = poseCamerawrt1[:3 , :3]
            ntvec = np.reshape(ntvec , (1,1,3))
            nrvec , _ = cv2.Rodrigues(nrvec)
            tglobal.append(ntvec)
            rglobal.append(nrvec)
            x , y , z = curtvec[0][0]
            quatern = tf.quaternion_from_matrix(poseCamerawrt1) 
            qw , qx , qy , qz  = quatern
            X.append(x)
            Y.append(y)
            Z.append(z)
            eular_angles = tf.euler_from_matrix(poseCamerawrt1 , 'rxyz')
            eular_angles = np.degrees(eular_angles)
            print(eular_angles)
        
            elapsed_time = time.time() - start_time

            
            with open(csvfilename, 'a' , newline='') as file:
                writer = csv.writer(file , delimiter='\t')
                row = [elapsed_time , x , y , z , qx, qy , qz , qw ,eular_angles ]
                writer.writerow(row)
                
            imgmarked = cv2.aruco.drawDetectedMarkers(cv_image.copy() , corners , ids)
            cv2.imshow('markedframe' , imgmarked)
            cv2.waitKey(1)
        else :
            cv2.imshow('markedframe' , cv_image)
            cv2.waitKey(1)
        


if __name__ == '__main__':
    dir1 = "/home/machine_visoin/codes/Sachin/Aruco-data/object_3d_sai/images/*.jpg"
    

    imagepaths1 = sorted(glob.glob(dir1) , key=lambda x: int(x.split('_')[-1].split('.')[0]))
    

    start_time = time.time()
    for filename in imagepaths1:
        cv_image = cv2.imread(filename)
        callback_function(cv_image)
             
        
    f = input("enter 1 to print in plottly and 2 to print in open3d")
    if f == '1':
        fig = go.Figure(data=[go.Scatter3d(x=X, y=Y, z=Z, mode='lines')])


        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-2 , 2], autorange=False) ,
                yaxis=dict(range=[-2 , 2], autorange=False) ,
                zaxis=dict(range=[-2,  2], autorange=False) , 
                aspectmode='manual'  , 
                aspectratio=dict(x=1, y=1, z=1)

            )
        )

        fig.show() 
    else:
        plot3d(rglobal , tglobal)
