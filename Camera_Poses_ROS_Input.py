#!/usr/bin/env python3


import rospy as ros
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d 
import transformations as tf
import time
import csv
import plotly.graph_objects as go



bridge = CvBridge()

p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
arucoParams = cv2.aruco.DetectorParameters_create()

filename = '/home/machine_visoin/codes/rosws/src/runscript/src/newcsvN.csv'

with open(filename, 'w' , newline='') as file:
    writer = csv.writer(file , delimiter='\t')
    writer.writerow(['Time' , 'X' , 'Y' , 'Z' , 'qx' , 'qy' , 'qz' , 'qw' ])


#camera matrix and distortion matrix
matrix = np.array([[6.20044860e+02 ,  0.00000000e+00 , 3.21058197e+02] ,
   [0.00000000e+00    ,  6.19899047e+02  , 2.25138992e+02] , 
   [0.00000000e+00    ,  0.00000000e+00  , 1.00000000e+00]])

distortion = np.array([[0.00000000e+00 , 0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 , 0.00000000e+00]])




 
firstmarker = False
main_id = None
rel_posedict = {}
iter=1
X = []
Y = []
Z = []



def findRelPoses(ids,corners, markerSizeInMt):
    for i in range(len(corners)):
        for j in range(len(corners)):
                poseA = np.eye(4)
                cornersM = np.reshape(corners[i] , (1,1,4,2))
                rvecA , tvecA, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)
                RA, _ = cv2.Rodrigues(rvecA[0])
                tvecAT = tvecA[0].T
                poseA[:3,:3] = RA
                poseA[:3,3:4] = tvecAT
                poseB = np.eye(4)
                cornersM = np.reshape(corners[j] , (1,1,4,2))
                rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)
                RB, _ = cv2.Rodrigues(rvecB[0])
                tvecBT = tvecB[0].T
                poseB[:3,:3] = RB
                poseB[:3,3:4] = tvecBT
                poseA_wrt_B = np.dot(np.linalg.inv(poseB) , poseA)
                rel_posedict[(ids[i][0],ids[j][0])] = poseA_wrt_B
                rel_posedict[(ids[j][0],ids[i][0])] = np.linalg.inv(poseA_wrt_B)
                eular_angles = tf.euler_from_matrix(poseA_wrt_B, 'rxyz')
                eular_angles = np.degrees(eular_angles)
            
                if  (ids[i][0], main_id) in rel_posedict:
                    rel_posedict[(ids[j][0],main_id)] = np.dot(rel_posedict[(ids[i][0],main_id)] , rel_posedict[(ids[j][0] , ids[i][0]) ])
                    rel_posedict[(main_id , ids[j][0])] = np.linalg.inv(rel_posedict[(ids[j][0], main_id)]) 
            
            



Poseglobal = []

def callback_function(msg):
        cv_image = bridge.imgmsg_to_cv2(msg , desired_encoding="bgr8")
        
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image , p_dict , parameters = arucoParams)
        
        global firstmarker
        global main_id
        global rel_posedict 
        global start_time
        global iter
        global X
        global Y
        global Z
        global Poseglobal

        markerSizeInMt = .08
        if len(corners)  > 0 :

            if firstmarker is False:
                firstmarker = True
                main_id = ids[0][0]
            iter = iter + 1 

            findRelPoses(ids=ids, corners=corners , markerSizeInMt=markerSizeInMt)
            
            
            
            poseB = np.eye(4)
            cornersM = np.reshape(corners[0] , (1,1,4,2))
            rvecB , tvecB, _ = cv2.aruco.estimatePoseSingleMarkers(cornersM, markerSizeInMt, matrix, distortion)
            RB, _ = cv2.Rodrigues(rvecB[0])
            tvecBT = tvecB[0].T
            poseB[:3,:3] = RB
            poseB[:3,3:4] = tvecBT
            if (main_id,ids[0][0]) not in rel_posedict:
                cv2.imshow('markedframe' , cv_image)
                cv2.waitKey(1)
                return
            pose1wrtC = np.dot(poseB,rel_posedict[(main_id,ids[0][0])])
            poseCamerawrt1 = np.linalg.inv(pose1wrtC)
            curtvec = poseCamerawrt1[:3 , 3:4]
            curtvec = curtvec.T
            curtvec = np.reshape(curtvec , (1,1,3))
            X.append(curtvec[0,0,0])
            Y.append(curtvec[0,0,1])
            Z.append(curtvec[0,0,2])
            
            x , y , z = curtvec[0][0]
            quatern = tf.quaternion_from_matrix(poseCamerawrt1) 
            qw , qx , qy , qz  = quatern
            Poseglobal.append(poseCamerawrt1)
        
            elapsed_time = time.time() - start_time
            with open(filename, 'a' , newline='') as file:
                writer = csv.writer(file , delimiter='\t')
                row = [elapsed_time , x , y , z , qx, qy , qz , qw ]
                writer.writerow(row)
                

            imgmarked = cv2.aruco.drawDetectedMarkers(cv_image.copy() , corners , ids)
            cv2.imshow('markedframe' , imgmarked)
        else :
            cv2.imshow('markedframe' , cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


		
		    
	

if __name__ == '__main__':
    ros.init_node('python_node')
    ros.loginfo("This node has been started")
    rate = ros.Rate(10)

    while not ros.is_shutdown():
        ros.loginfo("Hello")
        rate.sleep()
        start_time = time.time()
        ros.Subscriber('/camera/out' , Image , callback_function)
        ros.spin()
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


       
