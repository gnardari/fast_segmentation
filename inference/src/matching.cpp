#include "matching.hpp"

namespace fs = boost::filesystem;
using namespace std;
using namespace pcl;

#define MAXBUFSIZE  ((int) 1e6)

// double calculate_distance(PointXYZD p1, PointXYZD p2){
//   Eigen::VectorXf d(8);
//   for (size_t i = 0; i < 8; ++i){
//     d[i] = (p1.distances[i] - p2.distances[i]);
//   }
//   return d.squaredNorm();
// }
//
// int match_clouds(pcl::PointCloud<PointXYZD>::Ptr A, pcl::PointCloud<PointXYZD>::Ptr B){
//   int score = -1;
//   vector<Match> point_matches; 
//   for (size_t i = 0; i < A->points.size(); ++i){
//     for (size_t j = 0; j < B->points.size(); ++j){
//       float dist = calculate_distance(A->points[i],
//                                       B->points[j]);
//       if (dist < 0.5){
//         Match m;
//         m.a = A->points[i];
//         m.b = B->points[i];
//         point_matches.push_back(m);
//       }
//     }
//   }
//
//   //cout << point_matches.size() << endl;
//
//   return point_matches.size();
// }
//
// pcl::PointCloud<PointXYZD>::Ptr readCloud(string filename){
//   pcl::PointCloud<PointXYZD>::Ptr cloud (new PointCloud<PointXYZD>);
//
//   if (pcl::io::loadPCDFile<PointXYZD> (filename, *cloud) == -1) /#<{(| load the file
//   {
//     PCL_ERROR ("Couldn't read file\n");
//     return (cloud);
//   }
// //  std::cout << "Loaded "
// //            << cloud->width * cloud->height
// //            << " data points from .pcd"
// //            << std::endl;
//
//   // Populate distance vector for each tree
//   for (size_t i = 0; i < cloud->points.size (); ++i){
//     for (size_t j = 0; j < cloud->points.size (); ++j){
//       if(i == j) continue;
//
//       float dist = pcl::geometry::distance(cloud->points[i],cloud->points[j]);
//       //cout << dist << endl;
//       float x[8];
//       
//       copy(begin(cloud->points[i].distances),
//            end(cloud->points[i].distances), begin(x));
//
//       int largest = std::distance(x, std::max_element(begin(x), end(x)));
//       //cout << largest << endl;
//
//       if (dist < cloud->points[i].distances[largest]){
//           cloud->points[i].distances[largest] = dist;
//       }
//       
//     }
//
//     sort(begin(cloud->points[i].distances), end(cloud->points[i].distances));
//   }
//   return cloud;
// }

std::vector<std::string> get_directories(const std::string& s)
{
    vector<string> r;
    for(auto& p : fs::recursive_directory_iterator(s))
        if(p.extension() == ".pcd" && p.path().filename().rfind(cloudName, 0) != 0)
            r.push_back(p.path().string());
    return r;
}

void readFullCloud(string basePath, string cloudName,
    pcl::PointCloud<PointXYZD>::Ptr& fullMap){
    vector<string> dirs = get_directories(basePath);
    for (auto d : dirs){
        cout << d << endl;
    }
}

int main(int argc, char** argv){
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    string basePath = "/root/bags/cloud17-19_tree_detection_results/";

    pcl::PointCloud<PointXYZD>::Ptr fullMap;
    readFullCloud(basePath, "cloud19", fullMap);
    // pcl::PointCloud<PointXYZD>::Ptr scan = readCloud(string("/root/bags/submaps/area_")+
    //                                                to_string(j)+"/"+to_string(k)+".pcd");
    // int num_matches = match_clouds(Ma, Mb);

    return 0;
}
