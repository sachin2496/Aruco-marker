#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d 
import csv
import transformations as tf
import time


bridge = CvBridge()

p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
arucoParams = cv2.aruco.DetectorParameters_create()

#camera device dependent parameters ie camera matrix and distortion matrix

#distortion = np.array([[-0.97189488  ,  4.9884679 ,   0.09231124 ,  0.0462571 , -15.31741642]])
# matrix = np.array([[9.79009121e+02 , 0.00000000e+00 ,  2.57722179e+02] , 
#  [0.00000000e+00  , 1.00508650e+03 ,  1.04894450e+02] , 
#  [0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])

matrix = np.array([[6.20044860e+02 ,  0.00000000e+00 , 3.21058197e+02] ,
   [0.00000000e+00    ,  6.19899047e+02  , 2.25138992e+02] , 
   [0.00000000e+00    ,  0.00000000e+00  , 1.00000000e+00]])


distortion = np.array([[0.00000000e+00 , 0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 , 0.00000000e+00]])



  
rglobal = []
tglobal = []
i=1
   

filename = '/home/machine_visoin/rosws/src/runscript/src/newcsv.csv'

with open(filename, 'w' , newline='') as file:
    writer = csv.writer(file , delimiter='\t')
    print("sac")
    writer.writerow(['Time' , 'X' , 'Y' , 'Z' , 'qx' , 'qy' , 'qz' , 'qw' ])


first_camera_pos = None



def createcoordinateframe(position , rotation, size=1 ):
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
            #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
        
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
    #print(frames)
    vis = o3d.visualization.Visualizer()
    o3d.visualization.draw_geometries(geometries)
    vis.destroy_window()



def callback_function(msg ):
        cv_image = bridge.imgmsg_to_cv2(msg , desired_encoding="bgr8")
        
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image , p_dict , parameters = arucoParams)
            
        markerSizeInCM = 8
        if len(corners)  > 0 :
            rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, matrix, distortion)
            print(rvec.shape)
            R , _ = cv2.Rodrigues(rvec[0])
            global first_camera_pos
            global pose_ref
            global start_time
            global rglobal
            global tglobal
            global i
            if first_camera_pos is None:
                first_camera_pos = rvec
                #R_marker_ref = tf.euler_matrix(alpha , beta , gamma , 'rxyz')
                pose_ref = np.eye(4)
                pose_ref[:3,:3] = R
                tvecc =  tvec[0].T
                pose_ref[:3,3:4 ] = tvecc

            tvecT = tvec[0].T
            i = i+1
            pose_i = np.eye(4)
            pose_i[:3,:3] = R
            pose_i[:3,3:4] = tvecT
    
            pose_i = np.linalg.inv(pose_i)
            pose_rel = np.dot(pose_ref, pose_i)

            x = pose_rel[0][3]
            y = pose_rel[1][3]
            z = pose_rel[2][3]
            print(pose_rel)

            quatern = tf.quaternion_from_matrix(pose_rel)
            qw , qx , qy , qz  = quatern
            eular_angles = tf.euler_from_matrix(pose_rel , 'rxyz')
            eular_angles = np.degrees(eular_angles)
            print(eular_angles)
        
            elapsed_time = time.time() - start_time
            with open(filename, 'a' , newline='') as file:
                writer = csv.writer(file , delimiter='\t')
                row = [elapsed_time , x , y , z , qx, qy , qz , qw ]
                writer.writerow(row)
                print("writing")
            nrvec = pose_rel[:3 , :3]
            ntvec = pose_rel[:3 , 3:4]
            ntvec = ntvec.T
            ntvec = np.reshape(ntvec , (1,1,3))
            nrvec , _ = cv2.Rodrigues(nrvec)
            #print(ntvec.shape)
            tglobal.append(ntvec)
            rglobal.append(nrvec)
            res = cv2.drawFrameAxes(cv_image, matrix, distortion , rvec, tvec, markerSizeInCM ,  3 )
            imgmarked = cv2.aruco.drawDetectedMarkers(cv_image.copy() , corners , ids)
            cv2.imshow('markedframe' , imgmarked)
        else :
            cv2.imshow('markedframe' , cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            plot3d(rglobal , tglobal)
            cv2.destroyAllWindows()
            return


		
		    
	

if __name__ == '__main__':
    rospy.init_node('python_node_executing')
    rospy.loginfo("This node has been started")
rate = rospy.Rate(10)

while not rospy.is_shutdown():
    rospy.loginfo("Hello")
    f = 0 
    rate.sleep()
    start_time = time.time()
    rospy.Subscriber('/camera/color/image_raw' , Image , callback_function)
    rospy.spin()
    cv2.destroyAllWindows()
