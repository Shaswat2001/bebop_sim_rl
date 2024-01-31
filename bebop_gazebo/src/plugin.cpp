#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <rosgraph_msgs/Clock.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Float32.h>
#include <bebop_gazebo/RequestUavPose.h>

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
// #include <gazebo_ros/conversions/builtin_interfaces.hpp>
// #include <gazebo_ros/conversions/geometry_msgs.hpp>
// #include <gazebo_ros/node.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace std::chrono_literals;
using std::placeholders::_1;

namespace gazebo
{
    class ModelPush : public ModelPlugin
    {
    public:
        /// A pointer to the ROS node handle.
        ros::NodeHandle ros_node_;

        /// Subscriber to command velocities
        ros::Subscriber cmd_vel_sub_;

        /// Odometry publisher
        ros::Publisher odometry_pub_;

        ros::Subscriber clk_subscriber_;

        ros::Publisher publisher_;
        ros::Timer timer_;

        ros::ServiceServer service;

        ros::Subscriber pose_subscriber_;
        ros::Subscriber twst_subscriber_;

        ros::Publisher manip_ee_pub_;
        ros::Publisher manip_ee_diff_pub_;

        ros::Subscriber joint_subscription_;
        ros::Subscriber body_subscription_;

        void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf)
        {
            this->model = _model;
            // Initialize ROS node
            ROS_INFO("Loading Calmly (LOL)");

            clk_subscriber_ = this->ros_node_.subscribe<rosgraph_msgs::Clock>("clock", 10, &ModelPush::clk_update, this);
            service = this->ros_node_.advertiseService("get_uav_pose", &ModelPush::get_pose, this);

            this->update_connection_ = event::Events::ConnectWorldUpdateBegin(boost::bind(&ModelPush::OnUpdate, this));

            x_value_ = 0.3;
            y_value_ = 0.3;
            z_value_ = 0.50;
            r1_value_ = 0;
            r2_value_ = 0;
            r3_value_ = 0.0;
        }

        void OnUpdate()
        {
            this->model->SetWorldPose(ignition::math::Pose3d(x_value_, y_value_, z_value_, r1_value_, r2_value_, r3_value_));
        }

        void clk_update(const rosgraph_msgs::Clock::ConstPtr &time)
        {
            // TODO: Add implementation if needed
        }

        void pose_update(const geometry_msgs::PoseStamped::ConstPtr &pose)
        {
            // TODO: Add implementation if needed
        }

        void drone_body_string(const std_msgs::String &string_msg)
        {
            ROS_INFO("drone body message: %s", string_msg.data.c_str());

            std::cout << string_msg.data << std::endl;
            std::vector<std::string> stringVector;
            std::stringstream ss(string_msg.data);
            std::string token;

            while (std::getline(ss, token, ','))
            {
                stringVector.push_back(token);
            }

            double double_x_value_ = std::stod(stringVector[0]);
            double double_y_value_ = std::stod(stringVector[1]);
            double double_z_value_ = std::stod(stringVector[2]);
            double double_r1_value_ = std::stod(stringVector[3]);
            double double_r2_value_ = std::stod(stringVector[4]);
            double double_r3_value_ = std::stod(stringVector[5]);

            x_value_ = double_x_value_;
            y_value_ = double_y_value_;
            z_value_ = double_z_value_;
            r1_value_ = double_r1_value_;
            r2_value_ = double_r2_value_;
            r3_value_ = double_r3_value_;
        }

        bool get_pose(bebop_gazebo::RequestUavPose::Request &request,
                      bebop_gazebo::RequestUavPose::Response &response)
        {
            drone_body_string(request.uav_pose);

            int response_succ = 1;
            response.successful = response_succ;

            return true;
        }

    private:
        event::ConnectionPtr update_connection_;
        physics::ModelPtr model;
        double joint1_value_ = 0.;
        double joint2_value_ = 0.;
        double joint3_value_ = 0.;
        double joint4_value_ = 0.;
        double gripper_ = 0.;
        double x_value_ = 0.;
        double y_value_ = 0.;
        double z_value_ = 0.;
        double r1_value_ = 0.;
        double r2_value_ = 0.;
        double r3_value_ = 0.;
    };

    // Register this plugin with the simulator
    GZ_REGISTER_MODEL_PLUGIN(ModelPush)
}
