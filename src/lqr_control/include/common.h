#include <fstream>
#include <iostream>
#include <string>

#include <lgsvl_msgs/VehicleControlData.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>

#include "map.h"
#include "reference_line.h"
#include "ros/ros.h"
#include "tf/tf.h"

struct VehicleState {
  double x;
  double y;
  double heading;   // 车辆朝向
  double kappa;     // 曲率(切线斜率)
  double velocity;    // 速度
  double angular_velocity;  // 角速度
  double acceleration;    // 加速度
  double beta;   // slip ange
  double steering_angle;

  const double wheelbase_ = 2.852;  // 左右轮的距离
  const double mass_fl = 520;                   // 左前悬的质量
  const double mass_fr = 520;                   // 右前悬的质量
  const double mass_rl = 520;                   // 左后悬的质量
  const double mass_rr = 520;                   // 右后悬的质量
  const double mass_front = mass_fl + mass_fr;  // 前悬质量
  const double mass_rear = mass_rl + mass_rr;   // 后悬质量
  const double mass_ = mass_front + mass_rear;
  const double lf_ = wheelbase_ * (1.0 - mass_front / mass_);
  const double lr_ = wheelbase_ * (1.0 - mass_rear / mass_);  // 汽车后轮到中心点的距离

  // 规划起点
  double planning_init_x; 
  double planning_init_y;

  double roll;  
  double pitch;
  double yaw;

  double target_curv;  // 期望点的曲率

  double vx;
  double vy;

  // added
  double start_point_x;
  double start_point_y;

  double relative_x = 0;
  double relative_y = 0;

  double relative_distance = 0;
};

struct TrajectoryPoint {
  double x;
  double y;
  double heading;
  double kappa;
  double v;
  double a;
};

// 轨迹
struct TrajectoryData {
  std::vector<TrajectoryPoint> trajectory_points;
};

struct LateralControlError {
  double lateral_error; // 横向误差
  double heading_error; // 转向误差
  double lateral_error_rate;  // 横向误差速率
  double heading_error_rate;  // 转向误差速度
};


struct ControlCmd {
  double steer_target;
  double acc;
};


struct EulerAngles {
    double roll, pitch, yaw;
};



typedef std::shared_ptr<LateralControlError> LateralControlErrorPtr;