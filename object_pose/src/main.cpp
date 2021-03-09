#include <iostream>
#include <chrono>
#include <string>
#include <math.h>
//ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
//opencv
#include <opencv2/core/core.hpp>

//publish cvrect
#include <rect_msgs/cvRect.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

//Eigen
#include <Eigen/Eigen>
#include <Eigen/StdVector>

rect_msgs::cvRect cvRect_Now;

cv::Mat Depth_pic;
float dyaw =0; //takeoff yaw

double fx = 554.3826904296875;
double fy = 554.3826904296875;
double cx = 320;
double cy = 240;
#define PI 3.14159265

bool init_flag = false;
typedef std::chrono::duration<double,std::ratio<1,1000>> ms;


int track_cam =0;

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> SyncPolicyImagePose;
typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
Eigen::Vector3d camera_pos;
Eigen::Quaterniond camera_q;
Eigen::Quaterniond camera_q_tf;
Eigen::Vector3d quaternion_to_euler(const Eigen::Quaterniond &q)
{
    float quat[4];
    quat[0] = q.w();
    quat[1] = q.x();
    quat[2] = q.y();
    quat[3] = q.z();

    Eigen::Vector3d ans;
    ans[0] = atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]), 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]));
    ans[1] = asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]));
    ans[2] = atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]));
    return ans;
}
void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,const geometry_msgs::PoseStampedConstPtr& pose)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
    if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
        (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, 1000);
      }
    cv_ptr->image.copyTo(Depth_pic);
    if(Depth_pic.empty())
    {
        std::cout<<"depth is empty"<<std::endl;
    }
    camera_pos(0) = pose->pose.position.x;
    camera_pos(1) = pose->pose.position.y;
    camera_pos(2) = pose->pose.position.z;
    camera_q = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                         pose->pose.orientation.y, pose->pose.orientation.z);

    //std::cout<<"get depth"<<std::endl;
    //camera_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);



    //std::cout<<"recieved depth!"<<std::endl;

}
void rectCallback(const rect_msgs::cvRectConstPtr& msg)
{
    cvRect_Now = *msg;
}
nav_msgs::Odometry odom_;
double yaw_Now;
void OdomCallback(const nav_msgs::OdometryConstPtr& msg)
{
    odom_ = *msg;
    Eigen::Quaterniond q_drone(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d Euler_ = quaternion_to_euler(q_drone);
    //std::cout<<"yaw: "<<q_drone.w()<<"x: "<<q_drone.x()<<std::endl;
    yaw_Now = Euler_[2];
}
bool sign;
void signCallback(const std_msgs::BoolConstPtr& msg)
{
    sign = msg->data;
}
int main(int argc, char** argv) {
    ros::init(argc, argv, "object_pose");
    ros::NodeHandle nh("~");
    ros::Publisher object_pub = nh.advertise<geometry_msgs::PoseStamped>("/object/pose",1);
    ros::Subscriber rect_sub = nh.subscribe("/cvRect_track",1,rectCallback);
    ros::Subscriber odom_sub = nh.subscribe("/mavros/local_position/odom",1,OdomCallback);
    ros::Publisher dyaw_pub = nh.advertise<std_msgs::Float32>("/correct_dyaw",1);
    ros::Subscriber sign_sub =nh.subscribe("/go_sign",1,signCallback);
    image_transport::ImageTransport it(nh);

    /*
     * depth pose sub
     */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
    SynchronizerImagePose sync_image_pose_;

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/realsense_plugin/camera/depth/image_raw", 50));
    pose_sub_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh, "/orb_cam_align/camera_pose", 25));

    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&depthPoseCallback, _1, _2));



    ros::Rate loop_rate(30);
    camera_q_tf = Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5);
    std::cout<<"begin!"<<std::endl;
    //-----loop------//
    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();



        cv::Mat depth_frame = Depth_pic.clone();
        Eigen::Vector3d frame_pos;
        Eigen::Quaterniond frame_q,frame_tf;

        frame_pos = camera_pos;
        frame_q = camera_q;

        if(depth_frame.empty())
        {
            continue;
        }
        int u,v;
        u = cvRect_Now.x + cvRect_Now.width/2;
        v = cvRect_Now.y + cvRect_Now.height/2;
        std::cout<<"///"<<std::endl;
        std::cout<<"uv:"<<u<<" "<<v<<" "<<std::endl;


        double depth;
        depth = double(depth_frame.at<uint16_t>(v,u))/1000;
        std::cout<<"depth:"<<depth<<std::endl;

        Eigen::Matrix3d camera_r = frame_q.toRotationMatrix();
        Eigen::Matrix3d camera_tf = camera_q_tf.toRotationMatrix();
        Eigen::Vector3d pt_cur, pt_world;

        pt_cur(0) = (u - cx) * depth / fx;
        pt_cur(1) = (v - cy) * depth / fy;
        pt_cur(2) = depth;

        std::cout<<"pt cur:"<<pt_cur(0)<<" "<<pt_cur(1)<<" "<<pt_cur(2)<<" "<<std::endl;
        
        //compute goal yaw:



        pt_world = camera_r * pt_cur + frame_pos;
        geometry_msgs::PoseStamped object_pos;
        object_pos.header.stamp = ros::Time::now();
        object_pos.header.frame_id = "world";
        object_pos.pose.position.x = pt_world(0);
        object_pos.pose.position.y = pt_world(1);
        object_pos.pose.position.z = pt_world(2);

        if(!sign) continue;

        object_pub.publish(object_pos);

        /************** correct yaw *******************/
        /*
        float dy = (640/2 - u) * depth / fx;
        float dx = depth;
        float dyaw = atan2(dy,dx);
        */
        float error = (640/2 - u) /fx * 0.1; //比例控制
        dyaw += error;
        dyaw = dyaw > PI ? dyaw - 2 * PI : dyaw;
        dyaw = dyaw < -PI ? dyaw + 2 * PI : dyaw;
        std::cout<<"dyaw : "<<dyaw * 180 / PI<<std::endl;
        std::cout<<"error: "<<320 - u <<std::endl;
        std::cout<<"yaw_now: "<<yaw_Now * 180 /PI<<std::endl;
        std_msgs::Float32 dyaw_msg;
        dyaw_msg.data = dyaw; //+ yaw_Now;  //发布纠正后的偏航角 yaw
        dyaw_pub.publish(dyaw_msg);

    }

    return 0;
}
