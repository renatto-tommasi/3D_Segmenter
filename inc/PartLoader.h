#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/search/kdtree.h>
#include <filesystem>

// Type definitions for consistency
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

struct Part {
    std::string name;
    PointCloudNT::Ptr cloud;  // Changed to PointNormal
    float thickness;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features; 
};

class Loader {
public:
    Loader();

    int loadParts(const std::string& folder_path);

    void visualizeParts(int part_index);

    void setPartThicknesses(const std::vector<float>& thicknesses) {
        part_thicknesses_ = thicknesses;
    }
    std::vector<float> getPartThicknesses() const {
        return part_thicknesses_;
    }

    std::vector<Part> getParts() const {
        return parts_;
    }

    std::size_t getTotalParts() const {
        return total_parts_;
    }
private:
    int loadPart(const std::string& filename);

    int createVolume(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, PointCloudNT::Ptr& volume_cloud, float thickness = 1.0f);

    int getFeatures(PointCloudNT::Ptr& cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr& features, pcl::VoxelGrid<PointNT>& voxel_filter);
    std::vector<Part> parts_;
    std::size_t total_parts_;
    std::vector<float> part_thicknesses_;
};
