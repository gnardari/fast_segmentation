#include <iostream>
#include <random>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

using namespace std;

void graphToCloud(graph_t g, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud){
    size_t numPoints = 1;
    vector<double> color_values(num_vertices(g));
    iota(std::begin(color_values), std::end(color_values), 1);
    random_device rd;
    mt19937 gen(rd());
    shuffle(color_values.begin(), color_values.end(), gen);

    for(auto vd : make_iterator_range(vertices(g))){
        if(g[vd].treeId == -1) continue;
        for(auto coord : g[vd].points){
            pcl::PointXYZRGB pclP;
            pclP.x = coord[0];
            pclP.y = coord[1];
            pclP.z = coord[2];
            pclP.intensity = color_values[g[vd].treeId];
            cloud.push_back(pclP);
            numPoints++;
        }
    }
    cloud.width = numPoints;
    cloud.height = 1;
}

/*
void graphToMarkers(graph_t g, visualization_msgs::MarkerArray& markers){
    for(size_t i=0;i<g.nodes();i++){
    
    
    }
} 
*/
/*
int main(int argc, char** argv){

  // string base_path = "/media/gnardari/DATA/bags/vr_labels/test_cloud5/output/single/";
  string base_path = "/root/bags/all_labels_cloud1-17/";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::io::loadPCDFile(base_path+string("test_cloud5_0_tree0.pcd"), *cloud);

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud (cloud);
  while (!viewer.wasStopped ())
   {
   }

}*/
