#include <iostream>
#include <chrono>
#include <string>
//openvino
#include <inference_engine.hpp>
//ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
//opencv
#include <opencv2/core/core.hpp>

#include "security_barrier_camera/Detector.hpp"
#include "security_barrier_camera/VehicleAttributesClassifier.hpp"
#include "security_barrier_camera/Lpr.hpp"
//publish cvrect
#include <rect_msgs/cvRect.h>
#include <std_msgs/String.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

//Eigen
#include <Eigen/Eigen>
#include <Eigen/StdVector>

rect_msgs::cvRect cvRect_Now;

using namespace InferenceEngine;
cv::Mat Color_pic;
cv::Mat Depth_pic;

double fx = 384.0543518066406;
double fy = 384.0543518066406;
double cx = 322.71722412109375;
double cy = 241.58213806152344;

cv::Mat usbcam_pic1,usbcam_pic2;
bool init_flag = false;
typedef std::chrono::duration<double,std::ratio<1,1000>> ms;

/*****************************************/
enum MODE_STATE
{
    INIT,
    SEARCH,
    TRACK,
    LOST,
    STOP
};
MODE_STATE exec_state;
int track_cam =0;
/*****************************************/

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImagePtr cam_img;
   //std::cout<<"received pic!"<<std::endl;

   try {
       cam_img = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
   }catch(cv_bridge::Exception& e){
       ROS_ERROR("cv_bridge exception: %s",e.what());
       return;
   }
   if(cam_img)
   {
       Color_pic = cam_img->image.clone();
       //std::cout<<"color pic get"<<std::endl;
       cv::resize(Color_pic,Color_pic,cv::Size(640,480));

   }

}
/*
void usbcam_1_Callback(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImagePtr cam_img;
   //std::cout<<"received pic!"<<std::endl;

   try {
       cam_img = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
   }catch(cv_bridge::Exception& e){
       ROS_ERROR("cv_bridge exception: %s",e.what());
       return;
   }
   if(cam_img)
   {
       usbcam_pic1 = cam_img->image.clone();
       //cv::resize(Color_pic,Color_pic,cv::Size(640,480));

   }

}
void usbcam_2_Callback(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImagePtr cam_img;
   //std::cout<<"received pic!"<<std::endl;

   try {
       cam_img = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
   }catch(cv_bridge::Exception& e){
       ROS_ERROR("cv_bridge exception: %s",e.what());
       return;
   }
   if(cam_img)
   {
       usbcam_pic2 = cam_img->image.clone();
       //cv::resize(Color_pic,Color_pic,cv::Size(640,480));

   }

}
*/
bool lpr_string(const std::string& lprResult)
{
    if(lprResult == "<Beijing>Q60XZ8") return true;
    //if(lprResult == "<Beijing>NE91E9") return true;
    return false;
}
bool lpr_string_part(const std::string& lprResult)
{
    std::string delimiter =">";
    std::string back_plate = lprResult.substr(lprResult.find(delimiter)+1,2);
    std::string front_plate = lprResult.substr(0,lprResult.find(delimiter)+1);
    //std::cout<<front_plate + back_plate <<std::endl;
    if(back_plate == "AF") return true;
    else return false;
}
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped> SyncPolicyImagePose;
typedef std::shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
Eigen::Vector3d camera_pos;
Eigen::Quaterniond camera_q;
Eigen::Quaterniond camera_q_tf;
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

    std::cout<<"get depth"<<std::endl;
    //camera_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);



    //std::cout<<"recieved depth!"<<std::endl;

}
int main(int argc, char** argv) {
    ros::init(argc, argv, "security_barrier_camera");
    ros::NodeHandle nh("~");
    ros::Publisher cvRect_pub = nh.advertise<rect_msgs::cvRect>("/cvRect",10);
    ros::Publisher object_pub = nh.advertise<geometry_msgs::PoseStamped>("/object/pose",1);
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/realsense_plugin/camera/color/image_raw",1,imageCallback);
    //image_transport::Subscriber img_sub = it.subscribe("/camera/image_raw",1,imageCallback);


    //image_transport::Subscriber usbcam1_sub = it.subscribe("/video1/camera/image_raw",1,usbcam_1_Callback);
    //image_transport::Subscriber usbcam2_sub = it.subscribe("/video2/camera/image_raw",1,usbcam_2_Callback);

    /*
     * depth pose sub
     */
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
    std::shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
    SynchronizerImagePose sync_image_pose_;

    depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, "/realsense_plugin/camera/depth/image_raw", 50));
    pose_sub_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh, "/orb_cam_align/camera_pose", 25));
    //pose_sub_.reset(new message_filters::Subscriber<geometry_msgs::PoseStamped>(nh, "/RGBD/pose", 25));

    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&depthPoseCallback, _1, _2));

    //image_transport::Publisher img_pub = it.advertise("security_barrier_camera",1);

    //IE
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    InferenceEngine::Core ie;
    ie.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) }}, "CPU");
    ie.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS),CONFIG_VALUE(CPU_THROUGHPUT_AUTO)}}, "CPU");

    auto makeTagConfig = [&](const std::string &deviceName, const std::string &suffix) {
        std::map<std::string, std::string> config;
        return config;
    };

    Detector detector(ie, "CPU", "/home/amov/openvino_models/ir/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106.xml",
        {static_cast<float>(0.0), static_cast<float>(0.0)}, false, makeTagConfig("CPU", "Detect"));
    std::cout<<"1"<<std::endl;
    VehicleAttributesClassifier vehicleAttributesClassifier;
    vehicleAttributesClassifier = VehicleAttributesClassifier(ie, "CPU", "/home/amov/openvino_models/ir/intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml", false, makeTagConfig("CPU", "Attr"));
    std::cout<<"2"<<std::endl;
    Lpr lpr;
    lpr = Lpr(ie, "CPU", "/home/amov/openvino_models/ir/intel/license-plate-recognition-barrier-0001/FP16/license-plate-recognition-barrier-0001.xml",false, makeTagConfig("CPU", "LPR"));
    std::cout<<"3"<<std::endl;
    InferRequest detectorInferRequests = detector.createInferRequest();


    InferRequest attributesInferRequests = vehicleAttributesClassifier.createInferRequest();


    InferRequest lprInferRequests = lpr.createInferRequest();


    ros::Rate loop_rate(30);
    auto wallclock = std::chrono::high_resolution_clock::now();
    auto wallclock0 = wallclock;
    auto wallclock1 = wallclock;
    auto wallclock2 = wallclock;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;
    auto t2 = t0;
    auto t_c =t0;
    auto t_tracked = t0;
    exec_state = TRACK;
    camera_q_tf = Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5);
    std::cout<<"begin!"<<std::endl;
    //-----loop------//
    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
        if(Color_pic.empty()){
            std::cout<<"pic is empty!"<<std::endl;
            continue;
        }
        //auto t0 = std::chrono::high_resolution_clock::now();

        /*
        static int fsm_num =0;
        fsm_num++;
        if(fsm_num == 100)
        {
            std::cout<<"state: "<<exec_state<<std::endl;
            fsm_num =0;
        }
        */

        switch (exec_state)
        {
        case INIT:
        {
            std::cout<<"wait for goal!"<<std::endl;
            /*
             *  first on
             *  turn to SEARCH Mode
             */
            break;
        }
        case SEARCH:
        {
            /*
             *dont know where is car
             * use 3 cam to search car
             *
             */



            for(int iter_num=0;iter_num<3;iter_num++)
            {
                //std::cout<<iter_num<<std::endl;
                cv::Mat pic_copy = Color_pic.clone();
                detector.setImage(detectorInferRequests,pic_copy);
                //std::cout<<"he!"<<std::endl;
                detectorInferRequests.StartAsync();
                detectorInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); // break
                auto detector_results = detector.getResults(detectorInferRequests,pic_copy.size());
                //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
                std::list<cv::Rect> vehicleRects;
                std::list<cv::Rect> plateRects;
                std::pair<std::string, std::string> attriResults;
                std::string lprResults;
                cv::Mat frame = pic_copy.clone();
                //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
                for (Detector::Result result : detector_results)
                {
                    //std::cout<<"label: "<<result.label<<std::endl;
                    switch (result.label) {
                        case 1:
                        {
                            vehicleRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                            break;
                        }
                        case 2:
                        {
                            // expanding a bounding box a bit, better for the license plate recognition
                            result.location.x -= 5;
                            result.location.y -= 5;
                            result.location.width += 10;
                            result.location.height += 10;
                            plateRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                            break;
                        }
                        default:
                        {
                            //std::cout<<"label: "<<result.label<<std::endl;
                            throw std::exception();  // must never happen
                            break;
                        }
                    }

                }
                if(!(!vehicleRects.empty()||!plateRects.empty())) //当两种同时不存在
                //if(vehicleRects.empty())
                {


                    //std::cout<<"no Detect cars!"<<std::endl;
                    if(iter_num==0)
                    {
                        t0 = std::chrono::high_resolution_clock::now();
                        ms wall = std::chrono::duration_cast<ms>(t0 - wallclock0);
                        wallclock0 = t0;
                        std::ostringstream out;
                        out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";


                        cv::resize(frame,frame,cv::Size(640,480));
                        cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
                        std::string idx = std::to_string(iter_num);
                        cv::imshow("Detection Results:" + idx,frame);
                        cv::waitKey(3);

                    }
                    else if(iter_num ==1)
                    {
                        t1 = std::chrono::high_resolution_clock::now();
                        ms wall = std::chrono::duration_cast<ms>(t1 - wallclock1);
                        wallclock1 = t1;
                        std::ostringstream out;
                        out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";


                        cv::resize(frame,frame,cv::Size(640,480));
                        cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
                        std::string idx = std::to_string(iter_num);
                        cv::imshow("Detection Results:" + idx,frame);
                        cv::waitKey(3);

                    }
                    else if(iter_num ==2)
                    {
                        t2 = std::chrono::high_resolution_clock::now();
                        ms wall = std::chrono::duration_cast<ms>(t2 - wallclock2);
                        wallclock2 = t2;
                        std::ostringstream out;
                        out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";


                        cv::resize(frame,frame,cv::Size(640,480));
                        cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
                        std::string idx = std::to_string(iter_num);
                        cv::imshow("Detection Results:" + idx,frame);
                        cv::waitKey(3);

                    }


                    continue;//continue or ready 2 situations;

                }

                for (auto vehicleRectsIt = vehicleRects.begin(); vehicleRectsIt != vehicleRects.end();vehicleRectsIt++)
                {
                    const cv::Rect vehicleRect = *vehicleRectsIt;
                    vehicleAttributesClassifier.setImage(attributesInferRequests,pic_copy,vehicleRect);
                    attributesInferRequests.StartAsync();
                    attributesInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    attriResults = vehicleAttributesClassifier.getResults(attributesInferRequests);
                    //std::cout<<"attriResults: "<<attriResults.first<<" "<<attriResults.second<<std::endl;
                    cv::rectangle(frame, vehicleRect, {0, 255, 0},  4);
                    cv::putText(frame, attriResults.first+' '+attriResults.second,
                                cv::Point{vehicleRect.x, vehicleRect.y + 35},
                                cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 4);
                }
                vehicleRects.clear();

                for (auto plateRectsIt = plateRects.begin(); plateRectsIt != plateRects.end();plateRectsIt++)
                {
                    const cv::Rect plateRect = *plateRectsIt;
                    lpr.setImage(lprInferRequests,pic_copy,plateRect);
                    lprInferRequests.StartAsync();
                    lprInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                    lprResults = lpr.getResults(lprInferRequests);
                    //std::cout<<"lprResults: "<<lprResults<<std::endl;
                    cv::rectangle(frame, plateRect, {0, 0, 255},  4);
                    cv::putText(frame, lprResults,
                               cv::Point{plateRect.x, plateRect.y - 10},
                               cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
                    if(lpr_string(lprResults))
                    {
                        exec_state = TRACK;
                        track_cam = iter_num;
                    }
                    /*
                    if(lpr_string(lprResults))
                    {

                        cvRect_Now.x = (float)plateRect.x/frame.cols;
                        cvRect_Now.y = (float)plateRect.y/frame.rows;
                        cvRect_Now.width = (float)plateRect.width/frame.cols;
                        cvRect_Now.height = (float)plateRect.height/frame.rows;
                        cvRect_Now.result.data = lprResults;
                        cvRect_pub.publish(cvRect_Now);
                    }
                    */

                }
                plateRects.clear();
                if(iter_num == 0)
                {
                    t0 = std::chrono::high_resolution_clock::now();

                    //ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                    ms wall = std::chrono::duration_cast<ms>(t0 - wallclock0);

                    wallclock0 = t0;
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
                    cv::resize(frame,frame,cv::Size(640,480));
                    cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));

                    std::string idx = std::to_string(iter_num);
                    cv::imshow("Detection Results:"+idx,frame);
                    cv::waitKey(3);
                }
                else if(iter_num == 1)
                {
                    t1 = std::chrono::high_resolution_clock::now();

                    //ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                    ms wall = std::chrono::duration_cast<ms>(t1 - wallclock1);

                    wallclock1 = t1;
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
                    cv::resize(frame,frame,cv::Size(640,480));
                    cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));

                    std::string idx = std::to_string(iter_num);
                    cv::imshow("Detection Results:"+idx,frame);
                    cv::waitKey(3);
                }
                else if(iter_num == 2)
                {
                    t2 = std::chrono::high_resolution_clock::now();

                    //ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                    ms wall = std::chrono::duration_cast<ms>(t2 - wallclock2);

                    wallclock2 = t2;
                    std::ostringstream out;
                    out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
                    cv::resize(frame,frame,cv::Size(640,480));
                    cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));

                    std::string idx = std::to_string(iter_num);
                    cv::imshow("Detection Results:"+idx,frame);
                    cv::waitKey(3);

                }

            }
            break;
        }
        case TRACK:
        {
            /*find car
             * car is in sight, track it
             */
            cv::Mat pic_copy = Color_pic.clone();
            //std::cout<<"receieved!"<<std::endl;

            detector.setImage(detectorInferRequests,pic_copy);
            //std::cout<<"he!"<<std::endl;
            detectorInferRequests.StartAsync();
            detectorInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); // break
            auto detector_results = detector.getResults(detectorInferRequests,pic_copy.size());
            //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
            std::list<cv::Rect> vehicleRects;
            std::list<cv::Rect> plateRects;
            std::pair<std::string, std::string> attriResults;
            std::string lprResults;
            cv::Mat frame = pic_copy.clone();
            cv::Mat depth_frame = Depth_pic.clone();
            Eigen::Vector3d frame_pos;
            Eigen::Quaterniond frame_q,frame_tf;
            frame_pos = camera_pos;
            frame_q = camera_q;


            //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
            for (Detector::Result result : detector_results)
            {
                //std::cout<<"label: "<<result.label<<std::endl;
                switch (result.label) {
                    case 1:
                    {
                        vehicleRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                        break;
                    }
                    case 2:
                    {
                        // expanding a bounding box a bit, better for the license plate recognition
                        result.location.x -= 5;
                        result.location.y -= 5;
                        result.location.width += 10;
                        result.location.height += 10;
                        plateRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                        break;
                    }
                    default:
                    {
                        //std::cout<<"label: "<<result.label<<std::endl;
                        throw std::exception();  // must never happen
                        break;
                    }
                }

            }
            if(!(!vehicleRects.empty()||!plateRects.empty())) //当两种同时不存在
            //if(vehicleRects.empty())
            {


                //std::cout<<"no Detect cars!"<<std::endl;
                t_c = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t_c - wallclock);
                wallclock = t_c;
                std::ostringstream out;
                out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";


                cv::resize(frame,frame,cv::Size(640,480));
                cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
                cv::imshow("Tracking: Detection Results",frame);
                cv::waitKey(3);
                continue;//continue or ready 2 situations;

            }

            for (auto vehicleRectsIt = vehicleRects.begin(); vehicleRectsIt != vehicleRects.end();vehicleRectsIt++)
            {
                const cv::Rect vehicleRect = *vehicleRectsIt;
                vehicleAttributesClassifier.setImage(attributesInferRequests,pic_copy,vehicleRect);
                attributesInferRequests.StartAsync();
                attributesInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                attriResults = vehicleAttributesClassifier.getResults(attributesInferRequests);
                //std::cout<<"attriResults: "<<attriResults.first<<" "<<attriResults.second<<std::endl;
                cv::rectangle(frame, vehicleRect, {0, 255, 0},  4);
                cv::putText(frame, attriResults.first+' '+attriResults.second,
                            cv::Point{vehicleRect.x, vehicleRect.y + 35},
                            cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 4);
            }
            vehicleRects.clear();

            for (auto plateRectsIt = plateRects.begin(); plateRectsIt != plateRects.end();plateRectsIt++)
            {
                const cv::Rect plateRect = *plateRectsIt;
                lpr.setImage(lprInferRequests,pic_copy,plateRect);
                lprInferRequests.StartAsync();
                lprInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                lprResults = lpr.getResults(lprInferRequests);
                cv::rectangle(frame, plateRect, {0, 0, 255},  4);
                cv::putText(frame, lprResults,
                           cv::Point{plateRect.x, plateRect.y - 10},
                           cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
                if(lpr_string_part(lprResults))
                {
                    t_tracked = std::chrono::high_resolution_clock::now();
                    /*
                         * compute plate 3d point-d455
                         * publish goal, keep distance
                         */

                    /*
                    if(depth_frame.empty())
                    {

                        continue;
                    }
                    int u,v;
                    u = plateRect.x + plateRect.width/2;
                    v = plateRect.y + plateRect.height/2;

                    double depth;
                    depth = double(depth_frame.at<uint16_t>(v,u))/1000;

                    Eigen::Matrix3d camera_r = frame_q.toRotationMatrix();
                    Eigen::Matrix3d camera_tf = camera_q_tf.toRotationMatrix();
                    Eigen::Vector3d pt_cur, pt_world;

                    pt_cur(0) = (u - cx) * depth / fx;
                    pt_cur(1) = (v - cy) * depth / fy;
                    pt_cur(2) = depth;

                    //pt_cur = camera_tf * pt_cur;

                    pt_world = camera_r * pt_cur + frame_pos;
                    geometry_msgs::PoseStamped object_pos;
                    object_pos.header.stamp = ros::Time::now();
                    object_pos.header.frame_id = "world";
                    object_pos.pose.position.x = pt_world(0);
                    object_pos.pose.position.y = pt_world(1);
                    object_pos.pose.position.z = pt_world(2);

                    object_pub.publish(object_pos);
                    */



                    cvRect_Now.x = (float)plateRect.x;
                    cvRect_Now.y = (float)plateRect.y;
                    cvRect_Now.width = (float)plateRect.width;
                    cvRect_Now.height = (float)plateRect.height;
                    cvRect_Now.result.data = lprResults;
                    cvRect_pub.publish(cvRect_Now);


                }
                /*
                if(lpr_string(lprResults))
                {

                    cvRect_Now.x = (float)plateRect.x/frame.cols;
                    cvRect_Now.y = (float)plateRect.y/frame.rows;
                    cvRect_Now.width = (float)plateRect.width/frame.cols;
                    cvRect_Now.height = (float)plateRect.height/frame.rows;
                    cvRect_Now.result.data = lprResults;
                    cvRect_pub.publish(cvRect_Now);
                }
                */
            }
            plateRects.clear();
            /*
             * depth to goal
             *
             *
             */


            t_c = std::chrono::high_resolution_clock::now();
            ms last_tracked = std::chrono::duration_cast<ms>(t_c - t_tracked);
            if(last_tracked.count()/1000.0f > 1.5)
            {
                //exec_state = LOST;
                std::cout<<"LOST! try to search!"<<std::endl;
            }
            ms wall = std::chrono::duration_cast<ms>(t_c - wallclock);

            wallclock = t_c;
            std::ostringstream out;
            out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
            cv::resize(frame,frame,cv::Size(640,480));
            cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
            cv::imshow("Tracking: Detection Results",frame);
            cv::waitKey(3);



            break;
        }
        case LOST:
        {
            /*
             * have last goal, but exceed 10s cannot find out car
             * situation 1: turning
             * reach last goal and turn SEARCH
             *-------------------------------------------------
             * situation 2: move fast or cannot recognise plate
             * try to avoid this situation
             * use tracker?
             *
             */

            //try to find car, without plate
            cv::Mat pic_copy = Color_pic.clone();
            detector.setImage(detectorInferRequests,pic_copy);
            //std::cout<<"he!"<<std::endl;
            detectorInferRequests.StartAsync();
            detectorInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); // break
            auto detector_results = detector.getResults(detectorInferRequests,pic_copy.size());
            //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
            std::list<cv::Rect> vehicleRects;
            std::list<cv::Rect> plateRects;
            std::pair<std::string, std::string> attriResults;
            std::string lprResults;
            cv::Mat frame = pic_copy.clone();
            //std::cout<<"detect size: "<<detector_results.size()<<std::endl;
            for (Detector::Result result : detector_results)
            {
                //std::cout<<"label: "<<result.label<<std::endl;
                switch (result.label) {
                    case 1:
                    {
                        vehicleRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                        break;
                    }
                    case 2:
                    {
                        // expanding a bounding box a bit, better for the license plate recognition
                        result.location.x -= 5;
                        result.location.y -= 5;
                        result.location.width += 10;
                        result.location.height += 10;
                        plateRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), pic_copy.size()});
                        break;
                    }
                    default:
                    {
                        //std::cout<<"label: "<<result.label<<std::endl;
                        throw std::exception();  // must never happen
                        break;
                    }
                }

            }
            //if(!(!vehicleRects.empty()||!plateRects.empty())) //当两种同时不存在
            //if(vehicleRects.empty())
            if(vehicleRects.empty()&&plateRects.empty())
            {


                //std::cout<<"no Detect cars!"<<std::endl;
                t_c = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t_c - wallclock);
                wallclock = t_c;
                std::ostringstream out;
                out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps"<<" lost";


                cv::resize(frame,frame,cv::Size(640,480));
                cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
                cv::imshow("Lost: Detection Results",frame);
                cv::waitKey(3);
                continue;//continue or ready 2 situations;

            }

            for (auto vehicleRectsIt = vehicleRects.begin(); vehicleRectsIt != vehicleRects.end();vehicleRectsIt++)
            {
                const cv::Rect vehicleRect = *vehicleRectsIt;
                vehicleAttributesClassifier.setImage(attributesInferRequests,pic_copy,vehicleRect);
                attributesInferRequests.StartAsync();
                attributesInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
                attriResults = vehicleAttributesClassifier.getResults(attributesInferRequests);
                //std::cout<<"attriResults: "<<attriResults.first<<" "<<attriResults.second<<std::endl;
                cv::rectangle(frame, vehicleRect, {0, 255, 0},  4);
                cv::putText(frame, attriResults.first+' '+attriResults.second,
                            cv::Point{vehicleRect.x, vehicleRect.y + 35},
                            cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 4);
            }
            vehicleRects.clear();


            t_c = std::chrono::high_resolution_clock::now();
            ms wall = std::chrono::duration_cast<ms>(t_c - wallclock);

            wallclock = t_c;
            std::ostringstream out;
            out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
            cv::resize(frame,frame,cv::Size(640,480));
            cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));
            cv::imshow("Lost: Detection Results",frame);
            cv::waitKey(3);
            break;
        }
        case STOP:
        {
            break;
        }

        }
        /*
        detector.setImage(detectorInferRequests,Color_pic);
        detectorInferRequests.StartAsync();
        detectorInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY); // break
        auto detector_results = detector.getResults(detectorInferRequests,Color_pic.size());
        std::list<cv::Rect> vehicleRects;
        std::list<cv::Rect> plateRects;
        std::pair<std::string, std::string> attriResults;
        std::string lprResults;
        cv::Mat frame = Color_pic.clone();
        std::cout<<"detect size: "<<detector_results.size()<<std::endl;
        for (Detector::Result result : detector_results)
        {
            std::cout<<"label: "<<result.label<<std::endl;
            switch (result.label) {
                case 1:
                {
                    vehicleRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), Color_pic.size()});
                    break;
                }
                case 2:
                {
                    // expanding a bounding box a bit, better for the license plate recognition
                    result.location.x -= 5;
                    result.location.y -= 5;
                    result.location.width += 10;
                    result.location.height += 10;
                    plateRects.emplace_back(result.location & cv::Rect{cv::Point(0, 0), Color_pic.size()});
                    break;
                }
                default:
                {
                    std::cout<<"label: "<<result.label<<std::endl;
                    throw std::exception();  // must never happen
                    break;
                }
            }

        }
        if(!(!vehicleRects.empty()||!plateRects.empty())) //当两种同时不存在
        //if(vehicleRects.empty())
        {

            std::cout<<"no Detect cars!"<<std::endl;
            cv::resize(frame,frame,cv::Size(640,480));
            cv::imshow("Detection Results",frame);
            cv::waitKey(3);
            continue;//continue or ready 2 situations;

        }

        for (auto vehicleRectsIt = vehicleRects.begin(); vehicleRectsIt != vehicleRects.end();vehicleRectsIt++)
        {
            const cv::Rect vehicleRect = *vehicleRectsIt;
            vehicleAttributesClassifier.setImage(attributesInferRequests,Color_pic,vehicleRect);
            attributesInferRequests.StartAsync();
            attributesInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            attriResults = vehicleAttributesClassifier.getResults(attributesInferRequests);
            //std::cout<<"attriResults: "<<attriResults.first<<" "<<attriResults.second<<std::endl;
            cv::rectangle(frame, vehicleRect, {0, 255, 0},  4);
            cv::putText(frame, attriResults.first+' '+attriResults.second,
                        cv::Point{vehicleRect.x, vehicleRect.y + 35},
                        cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 4);
        }
        vehicleRects.clear();

        for (auto plateRectsIt = plateRects.begin(); plateRectsIt != plateRects.end();plateRectsIt++)
        {
            const cv::Rect plateRect = *plateRectsIt;
            lpr.setImage(lprInferRequests,Color_pic,plateRect);
            lprInferRequests.StartAsync();
            lprInferRequests.Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
            lprResults = lpr.getResults(lprInferRequests);
            //std::cout<<"lprResults: "<<lprResults<<std::endl;
            cv::rectangle(frame, plateRect, {0, 0, 255},  4);
            cv::putText(frame, lprResults,
                       cv::Point{plateRect.x, plateRect.y - 10},
                       cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
            if(lpr_string(lprResults))
            {

                cvRect_Now.x = (float)plateRect.x/frame.cols;
                cvRect_Now.y = (float)plateRect.y/frame.rows;
                cvRect_Now.width = (float)plateRect.width/frame.cols;
                cvRect_Now.height = (float)plateRect.height/frame.rows;
                cvRect_Now.result.data = lprResults;
                cvRect_pub.publish(cvRect_Now);
            }
        }
        plateRects.clear();
        t0 = std::chrono::high_resolution_clock::now();
        ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
        wallclock = t0;
        std::ostringstream out;
        out << std::fixed << std::setprecision(2) << wall.count() << "ms (" << 1000.f / wall.count() << "fps";
        cv::resize(frame,frame,cv::Size(640,480));
        cv::putText(frame,out.str(),cv::Point2f(0,50),cv::FONT_HERSHEY_TRIPLEX,1.3,cv::Scalar(0,0,255));

        cv::imshow("Detection Results",frame);
        cv::waitKey(3);
        */


    }
    //---------main loop end-----------

    return 0;
}
