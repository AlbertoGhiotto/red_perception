![ROS Distro](https://img.shields.io/badge/ROS-Melodic-Green)
![python](https://img.shields.io/badge/Python-2.7.17-blue)
![cv2](https://img.shields.io/badge/OpenCV-3.4.6.27-lightgrey)


# Red Perception

A simple node detecting the centrode and total area of a red region.

## Installation 

Install OpenCV in python 2 for ROS compatibility

``` pip2 install opencv-python==3.4.6.27 ```

Install numpy, matplotlib, cv_bridge and the other dependencies.


## Running 

Run the script with

``` rosrun robot_perception red_perception_node.py ```
