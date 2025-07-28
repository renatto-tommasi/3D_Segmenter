#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>
#include <pcl/console/print.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <memory>

#include "..//inc/PartLoader.h"

// Type definitions for clarity
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT,PointNT,FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;



class Segmenter {
public:
    Segmenter();

    void loadPointCloud(const std::string& filename);
    
    void visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

    void segment(); 

    void segmentPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

    std::unique_ptr<Loader> part_loader_;


private:

    void changeColorToWhite();

    void colorPointsByCluster(const std::vector<pcl::PointIndices>& cluster_indices);

    void voxelizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float leaf_size);
    
    void visualizeAlignment(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scene_cloud,
                           const pcl::PointCloud<pcl::PointXYZRGB>& aligned_part,
                           const std::string& part_name,
                           double fitness_score);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterOutTable(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_in);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropInnerRectangle(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud_in,
	double cut_min_x_mm,
	double cut_max_x_mm,
	double cut_min_y_mm,
	double cut_max_y_mm);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr working_cloud_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> working_clouds_;
    int cloud_size_;
};