This is the implementation for the attack recovery algorithm desccribed in: 

> **Fast Attack Recovery for Stochastic Cyber-Physical Systems**
>
> Lin Zhang\*, Luis Burbano\*, Xin Chen, Alvaro A. Cardenas, Steve Drager, Matthew Anderson, and Fanxin Kong
>
> \* contributed equally. [[*RTAS 2024*]](https://2024.rtas.org/)

## News
- **`[2024/01/21]`** This paper has been accepted by [RTAS 2024](https://2024.rtas.org/accepted-papers/)!
- ** Moved. ** The code has been integrated into [CPSim](https://sim.cpsec.org/en/latest/), please refer to [the example document](https://sim.cpsec.org/en/latest/5_example.html#fast-attack-recovery-for-stochastic-cyber-physical-systems).

## Installation
Please refer to [CPSim documentation](https://sim.cpsec.org/en/latest/2_install.html#installation).

## Notes
Explanation for different branches:
|        | Git branch                                                              | Comments                                                                                                                                                                                                                                                          |
|--------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AirSim | [drone](https://github.com/CPSEC/probabilistic-recovery/tree/drone)     | This branch implements the recovery strategy in the drone. It requires Gazebo, and ROS.                                                                                                                                                                           |
| SVL    | [ros](https://github.com/CPSEC/probabilistic-recovery/tree/ros)         | This branch demonstrates how to deploy our algorithm on ROS 2, and further working with SVL simulator. Please refer to [ROS launch files](https://github.com/CPSEC/probabilistic-recovery/tree/ros/src/recovery/launch) for baselines and the proposed algorithm. |
| Robot  | [traxxas](https://github.com/CPSEC/probabilistic-recovery/tree/traxxas) | This experiment requires the robot vehicle and the OptiTrack cameras and software.                                                                                                                                                                                |