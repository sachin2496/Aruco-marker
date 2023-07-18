## Registering Objects using Camera sensors
We are registering objects using the camera sensors i.e, by using depth and rgb images .
<br>
## Camera Pose estimation using multiple Aruco Markers
We have a Rotating board in which 4 aruco marker are present in corners . Keep id of these markers different and dictionary same  and size being 8 cm .
<br>
<p align="center">
<img  width="300" height="200" src="/rgb_0_00000.jpg"  >
</p>


## Dataset Used
We have a synthetic dataset which is captured in blender in which depth and rgb images are present . To download this data  click 
<a href="https://drive.google.com/drive/folders/1cbPCJaJlYYIZCGvCbXDhjzNIAOOovA0G" > here </a>



## How to run this code 

### Installing dependencies
To install the required packages, run the following command:
```shell
$ pip install -r requirements.txt
```

### Setting Up the Environment
To set up your development environment, follow the steps below:
Clone the repository to your local machine:

```shell
$ git clone https://github.com/sachin2496/Registering-Objects-using-sensors.git
```
Navigate to the project directory:
```shell
$ cd Registering-Objects-using-sensors
```
Run this to find the camera caliberation matrix and distortion matrix , for this you need 6*8 checkeroard  
```shell
$ python caliberation.py
```
### Estimating Camera Poses from set of images having aruco markers and making a csv file for the poses
We are here estimating the camera poses with respect to the very first aruco marker . Assuming the markers are in rest position . 
```shell
$ python CamerPoses_from_image-Updated.py
```
#### Estimating Camera Poses in which input is coming from a  ROS topic of rgb images
We are here estimating Camera poses wrt very first aruco marker assuming aruco board in still position ,  subscribing the rgb image topic from the camera sensor . 
```shell
$ python Camera_Poses_ROS_Input.py
```

### Getting Point Cloud of synthetic dataset and merge them on the basis of estimated camera Poses
we can run this python file alone to estimate camera poses , make point clouds and merge them . We will estimate the relative poses of first camera position wrt to ith camera position and pass them as a extrinsic while forming the ith point clouds.
```shell
$ python Register_withPoses.py
```
<p align="center">
<img  width="300" height="200" src="/Image_with_poses.png"  >
</p>

### Use camera poses as a initial alignment and then register them with coloured ICP .
We are using camera poses with align the point clouds and the use icp to register them so them they converge more efficiently.

```shell
$ python Register_with_poses_and_icp.py
```

<p align="center">
<img  width="300" height="200" src="/Image_ColouredICP.png"  >
</p>











