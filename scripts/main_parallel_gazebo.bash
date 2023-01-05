#!/bin/bash

# Parameters: port
# 	      strategy
export ROS_MASTER_URI=http://localhost:1135$1
export GAZEBO_MASTER_URI=http://localhost:1134$1

roslaunch uav_gazebo simple_world.launch & 
PID_GAZEBO=$!
sleep 10


strat=$1
noises=(0 0.001 0.002 0.003 0.004 0.005)
# noises=(0.004 0.005)
time=`date +%s`
for noise in ${noises[@]}; do
	./multiple_run.bash $strat ${noise} #> results/${main_f_name}
done
kill -9 ${PID_GAZEBO}
