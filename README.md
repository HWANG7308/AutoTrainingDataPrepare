# AutoTrainingDataPrepare

## Project Description
Robot-assisted automated training data preparation for robot vision tasks

Uses the [UR - RTDE](https://www.universal-robots.com/how-tos-and-faqs/how-to/ur-how-tos/real-time-data-exchange-rtde-guide-22229/)- protocol to send continuous updates to a Robot for smooth continued motion.

heavily relies on this repository:
https://github.com/Mandelbr0t/UniversalRobot-RealtimeControl
which builds ontop of:
https://bitbucket.org/RopeRobotics/ur-interface/src/master/

The visualization of annotation labels is adapted from https://github.com/NVlabs/FoundationPose

## How to

Before running the program, you need to specify the following parameters.

1. The ip address of your robot to guarantee a successful connection to your robot.
2. The numbers of the step the robot should move regarding the polar angle and the azimuthal angle.

<summary>Image collection</summary>

```bash
python data_acquisition.py
```
<summary>Image annotation</summary>

```bash
python data_annotation.py
```


## TODO
High priority
- [x] Robot controller
- [x] Robot pose generation
- [x] Camera controller
- [x] Camera calibration
- [x] Image acquisition
- [x] Data annotation format confirmation (2D bounding box)
- [x] Data annotation format confirmation (6D pose)
- [x] 2D bounding box annotation
- [x] 6D pose annotation
- [ ] Training and testing (2D object detection)
- [ ] Training and testing (6D object pose estimation)

Low priority
- [ ] Robot path planning and optimization
- [ ] Data annotation format confirmation (3D bounding box)
- [x] 3D bounding box annotation
- [ ] Training and testing (3D object detection)
- [ ] More testing
- [ ] Demonstration video
- [ ] Explanation video
- [ ] License clarification
- [ ] Cleanup
- [ ] Refactoring
- [ ] Documentation
- [ ] Comments
- [ ] Dependencies