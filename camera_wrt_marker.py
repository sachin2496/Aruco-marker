#importing libraries

import cv2
import numpy as np
import open3d as o3d 


#defining dictionary which we will detect
p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

#camera device dependent parameters ie camera matrix and distortion matrix

distortion = np.array([[-0.97189488  ,  4.9884679 ,   0.09231124 ,  0.0462571 , -15.31741642]])
matrix = np.array([[9.79009121e+02 , 0.00000000e+00 ,  2.57722179e+02] , 
 [0.00000000e+00  , 1.00508650e+03 ,  1.04894450e+02] , 
 [0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00]])


#estimating rvec and tvec of camera wrt to marker
vid = cv2.VideoCapture(0)
p_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
rglobal = []
tglobal = []
arucoParams = cv2.aruco.DetectorParameters_create()


while(1):
    ret, frame = vid.read()
    if ret == 1:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame , p_dict , parameters = arucoParams)
        markerSizeInCM = .8
        objectpoints = np.array([[0 , 0 , 0] , [markerSizeInCM , 0 , 0] , [markerSizeInCM , markerSizeInCM , 0 ] , [0 , markerSizeInCM , 0 ]] , dtype=np.float32)
        if len(corners)  > 0 :
            object2d = corners[0]
            _ , rvec , tvec = cv2.solvePnP(objectpoints , object2d , matrix , distortion , flags=0)
            #print(tvec)
            rglobal.append(rvec)
            tglobal.append(tvec)
            res = cv2.drawFrameAxes(frame, matrix, distortion , rvec, tvec, markerSizeInCM ,  3 )
            imgmarked = cv2.aruco.drawDetectedMarkers(frame.copy() , corners , ids)
            cv2.imshow('markedframe' , imgmarked)
        else :
            cv2.imshow('markedframe' ,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

  

#using open3d to plot th e 3d pose of camera
line_sets = []
previous_pose = None
for i, T_WC in enumerate(tglobal):
    #print(T_WC.shape)
    T_WC = T_WC.T
    if previous_pose is not None:
        points = o3d.utility.Vector3dVector([previous_pose[0,:], T_WC[0,:]])
        lines = o3d.utility.Vector2iVector([[0, 1]])
        line = o3d.geometry.LineSet(points=points, lines=lines)
        line_sets.append(line)
    previous_pose = T_WC



def createcoordinateframe(position , rotation, size=0.5):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = size , origin = [ 0 , 0 ,0])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    mesh_frame.transform(T)
    return mesh_frame


frames = []
for rvec , tvec in zip(rglobal , tglobal):
    R, _ = cv2.Rodrigues(rvec)
    tvec =  np.reshape(tvec , (1, 1 , 3))
    
    frame = createcoordinateframe(tvec , R)
    frames.append(frame)




geometries = []
geometries += line_sets
geometries += frames


o3d.visualization.draw_geometries(geometries)
