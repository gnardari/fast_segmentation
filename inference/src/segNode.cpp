#include <segNode.h>

namespace trt_inference {
Segmentation::Segmentation(ros::NodeHandle& nh, EngConfig ec)
    : nh_(nh), it_(nh), InferenceEngineCPU(ec) {
  maskPubCloud_ =
      nh_.advertise<pcl::PointCloud<pcl::PointXYZI>>("maskCloud", 10);
  rawPubCloud_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZI>>("rawCloud", 10);
  rImgPub_ = it_.advertise("rangeImg", 10);
  rMaskPub_ = it_.advertise("maskImg", 10);
  // Subscriber
  sub_cloud_ = new pc_sub_type(nh_, "/velodyne/cloud", 5);
  sub_img_ = new image_sub_type(nh_, "/velodyne/image", 5);
  sync_ = new message_filters::Synchronizer<ImgCloudPolicy>(
      ImgCloudPolicy(5), *sub_img_, *sub_cloud_);
  sync_->registerCallback(
      boost::bind(&Segmentation::rangeCloudCb_, this, _1, _2));
}

// void Segmentation::maskCloud_(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                              // pcl::PointCloud<pcl::PointXYZI>::Ptr& outCloud,
                              // const cv::Mat mask) {
  // const float nan_coord = std::numeric_limits<float>::quiet_NaN();
  // pcl::PointCloud<pcl::PointXYZI>::Ptr auxCloud(
        // new pcl::PointCloud<pcl::PointXYZI>);
  // auxCloud->is_dense = false;
  // auxCloud->header = cloud->header;
  // size_t numPoints = 0;
//
  // for (size_t i = 0; i < cloud->height; i++) {
    // const unsigned char* row_p = mask.ptr<unsigned char>(i);
    // for (size_t j = 0; j < cloud->width; j++) {
      // if (j < width_ && row_p[j] == 255) {
        // cloud->at(j,i) = bg_point;
        // numPoints++;
        // auxCloud->push_back(cloud->at(j, i));
      // }
    // }
  // }
  // auxCloud->width = numPoints;
  // auxCloud->height = 1;
//
  // pcl::StatisticalOutlierRemoval<pcl::PointXYZI>sorfilter(true);
  // sorfilter.setInputCloud(auxCloud);
  // sorfilter.setMeanK(8);
  // sorfilter.setStddevMulThresh(1.0);
  // sorfilter.filter(*outCloud);
// }
void Segmentation::maskCloud_(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr& outCloud,
                              cv::Mat mask) {
  // const float nan_coord = std::numeric_limits<float>::quiet_NaN();
  outCloud->is_dense = false;
  outCloud->header = cloud->header;
  size_t numPoints = 0;

  for (size_t i = 0; i < cloud->height; i++) {
    unsigned char* row_p = mask.ptr<unsigned char>(i);
    for (size_t j = 0; j < cloud->width; j++) {
      if (j < width_ && row_p[j] > 0) {
        // cloud->at(j,i) = bg_point;
        row_p[j] = 255;
        numPoints++;
        outCloud->push_back(cloud->at(j, i));
      }
    }
  }
  outCloud->width = numPoints;
  outCloud->height = 1;
}

void Segmentation::preProcessRange_(cv::Mat& rImg, float maxDist) {
  cv::extractChannel(rImg, rImg, 0);
  rImg.convertTo(rImg, CV_32F);
  for (size_t i = 0; i < rImg.rows; i++) {
    float* row_p = rImg.ptr<float>(i);
    for (size_t j = 0; j < rImg.cols; j++) {
      if (row_p[j] > maxDist * 500.0) {
        row_p[j] = 0;
      } else {
        row_p[j] /= 500.0;
      }
    }
  }
}


void Segmentation::extractSamples(cv::Mat& rImg, std::vector<cv::Mat>& samples) {
  // double rangeSplits[4][2] = {{0.0,10.0},{10.0,20.0},{20.0,30.0},{30.0,40.0}};
  double rangeSplits[3][2] = {{0.0,20.0},{10.0,30.0},{20.0,40.0}};
  for(const auto& radiusRng : rangeSplits){
    cv::Mat rngMask(rImg.rows, rImg.cols, CV_8U);
    cv::Mat samp;

    inRange(rImg, radiusRng[0], radiusRng[1], rngMask);
    rImg.copyTo(samp, rngMask);
    
    // std::cout << samp.rows << std::endl;
    // std::cout << samp.cols << std::endl;
    // double min, max;
    // cv::minMaxLoc(samp, &min, &max);
    // std::cout << min << std::endl;
    // std::cout << max << std::endl;

    // samp = (samp - radiusRng[0]) / (radiusRng[1] - radiusRng[0]);
    samp /= radiusRng[1];

    // std::cout << "vs" << std::endl;
    // cv::minMaxLoc(samp, &min, &max);
    // std::cout << min << std::endl;
    // std::cout << max << std::endl;
    samples.push_back(samp.clone());
  }
}

void Segmentation::rangeCloudCb_(
    const sensor_msgs::ImageConstPtr& imgMsg,
    const sensor_msgs::PointCloud2ConstPtr& cloudMsg) {
  ROS_INFO("Got msg");
  try {
    cv::Mat rawRangeImg = cv_bridge::toCvShare(imgMsg)->image;
    preProcessRange_(rawRangeImg, 40.0);
    cv::Mat std_rimg;
    if (rawRangeImg.cols > width_) {
      // crop
      cv::Rect roI(0, 0, width_, height_);
      std_rimg = rawRangeImg(roI);
    } else if (rawRangeImg.cols < width_) {
      // pad
      std_rimg.create(height_, width_, rawRangeImg.type());
      std_rimg.setTo(cv::Scalar::all(0));
      rawRangeImg.copyTo(
          std_rimg(cv::Rect(0, 0, rawRangeImg.cols, rawRangeImg.rows)));

    } else {
      std_rimg = rawRangeImg;
    }

    cv::Mat rImg = std_rimg.clone();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr segCloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloudMsg, *cloud);
    segCloud->header = cloud->header;

    std::vector<cv::Mat> samples;
    extractSamples(rImg, samples);
    cv::Mat mask = cv::Mat::zeros(height_, width_, CV_8U);
    // get nn output
    for(auto srImg : samples){
      cv::Mat sMask(height_, width_, CV_8U);
      run(srImg, sMask);

      // std::cout << "mask" << std::endl;
      double min, max;
      cv::minMaxLoc(sMask, &min, &max);
      // std::cout << min << std::endl;
      // std::cout << max << std::endl;
      mask += sMask;
    }
    // project output to cloud and publish
    // std::cout << "final mask" << std::endl;
    // double min, max;
    // cv::minMaxLoc(mask, &min, &max);
    // std::cout << min << std::endl;
    // std::cout << max << std::endl;
    maskCloud_(cloud, segCloud, mask);

    sensor_msgs::PointCloud2 rawCloudMsg;
    sensor_msgs::PointCloud2 segCloudMsg;
    pcl::toROSMsg(*cloud.get(), rawCloudMsg);
    pcl::toROSMsg(*segCloud.get(), segCloudMsg);
    rawCloudMsg.header = imgMsg->header;
    segCloudMsg.header = imgMsg->header;

    // publish result and forward synced raw data
    maskPubCloud_.publish(segCloudMsg);
    rawPubCloud_.publish(rawCloudMsg);
    rMaskPub_.publish(
        cv_bridge::CvImage(imgMsg->header, imgEnc::MONO8, mask).toImageMsg());
    rImgPub_.publish(
        cv_bridge::CvImage(imgMsg->header, imgEnc::TYPE_32FC1, rImg)
            .toImageMsg());
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'mono16'.",
              imgMsg->encoding.c_str());
  }
}
}  // namespace trt_inference
/*
int main(int argc, char **argv){

    EngConfig ec;
    ec.width = 3600;
    ec.height = 16;
    ec.numClasses = 2;
    ec.planPath =
"/home/jetson/Documents/realtime_segmentation/models/simple_erfnet.plan";
    ec.inputTensorName = "inputs/X";
    ec.outputTensorName = "up23/BiasAdd";

    //InferenceEngine infEng(planPath, input_tensor, output_tensor,
    //        width, height, num_classes);
    ROS_INFO("Created Engine");

    ros::init(argc, argv, "segmentation");
    ros::NodeHandle nh;
    trt_inference::Segmentation non(nh, ec);

    ROS_INFO("Created Node");
    ros::spin();
}*/
