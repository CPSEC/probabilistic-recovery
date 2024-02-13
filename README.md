
# Fast attack recovery for the drone

This project builds upon the [repository](https://github.com/fdcl-gwu/uav_simulator.git). Please follow their instructions to install the necessary dependencies.

To run the code, please run Gazebo in one command prompt. In a second command prompt, make the `scripts` folder your working direction. Then, run the command
```
python main.py [strategy] [closed loop] [detection delay] [noise]
```
where `strategy` is a number between 1 and 3; 1: RTR-LQR, 2: VS, 3: OPR. `closed loop` is a binary: 0: does not use the closed loop, 1: use the close loop. Only implemented for the OPR strategy. `detection delay` is the time between detection and the recovery begins. `noise` is the noise level. 
