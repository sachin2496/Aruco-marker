
import cv2
import numpy as np


vid = cv2.VideoCapture(0)


# Define the dimensions of checkerboard
CHECKERBOARD = (6, 8)


criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
					* CHECKERBOARD[1],
					3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
							0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None



_ , image = vid.read()
grayColor = image 



while(1): 
	ret, frame = vid.read()
	if ret == 1 :     
		grayColor = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(
						grayColor, CHECKERBOARD,
						cv2.CALIB_CB_ADAPTIVE_THRESH
						+ cv2.CALIB_CB_FAST_CHECK +
						cv2.CALIB_CB_NORMALIZE_IMAGE)
		if ret == True:
			threedpoints.append(objectp3d)

			# Refining pixel coordinates
			# for given 2d points.
			corners2 = cv2.cornerSubPix(
				grayColor, corners, (11, 11), (-1, -1), criteria)

			twodpoints.append(corners2)

			# Draw and display the corners
			image = cv2.drawChessboardCorners(frame ,
											CHECKERBOARD,
											corners2, ret)

			cv2.imshow('img', image)
		else :
			cv2.imshow('img' , frame)
		
	if cv2.waitKey(1) & 0xFF == ord('q') :
		break
vid.release()
cv2.destroyAllWindows()

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	threedpoints, twodpoints, grayColor.shape[::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)
