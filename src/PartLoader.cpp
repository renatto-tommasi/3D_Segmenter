#include "../inc/PartLoader.h"

// Use the same typedefs as defined in PartLoader.h
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudNT;

Loader::Loader() {
    // Constructor implementation
    total_parts_ = 0;
    part_thicknesses_ = {1}; // Example thicknesses in mm
}

int Loader::loadParts(const std::string& folder_path) {
        // Load part implementation
        for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
            // the entry is a folder
            if (entry.is_directory()){
                std::cout << "Loading parts from folder: " << entry.path() << std::endl;
                for (const auto& part_entry : std::filesystem::directory_iterator(entry.path())) {
                    // Only load surface.pcd files
                    if (part_entry.is_regular_file() && part_entry.path().filename() == "surface.pcd") {
                        loadPart(part_entry.path().string());
                    }
                }
            } 
        }
        std::cout << "Total parts loaded: " << total_parts_ << std::endl;
        return 0; // Success
    }


int Loader::loadPart(const std::string& filename) {
    // Load part implementation
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return -1; // Error loading part
    }

    // if no rgb data is present, set it to white
    if (cloud->points[0].r == 0 && cloud->points[0].g == 0 && cloud->points[0].b == 0) {
        for (auto& point : cloud->points) {
            point.r = static_cast<uint8_t>(255);
            point.g = static_cast<uint8_t>(255);
            point.b = static_cast<uint8_t>(255);
        }
        std::cout << "No RGB data found in " << filename << ". Setting all points to white." << std::endl;
    }
    std::cout << "Loaded " << cloud->size() << " points from " << filename << std::endl;

    // Extract part name from the folder name
    std::string part_name = std::filesystem::path(filename).parent_path().filename().string();

    // Create volume from the loaded cloud
    PointCloudNT::Ptr volume_cloud(new PointCloudNT);
    pcl::VoxelGrid<PointNT> voxel_filter;
    for (const float thickness : part_thicknesses_) {
        if (createVolume(cloud, volume_cloud, thickness) != 0) {
            std::cerr << "Error creating volume for part: " << filename << std::endl;
            return -1; // Error creating volume
        }
        // Get features
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>);
        if (getFeatures(volume_cloud, features, voxel_filter) != 0) {
            std::cerr << "Error getting features for part: " << filename << std::endl;
            return -1; // Error getting features
        }

        Part new_part;
        new_part.name = part_name+ "_" + std::to_string(static_cast<int>(thickness)) + "mm";
        new_part.cloud = volume_cloud;
        new_part.thickness = thickness;
        new_part.features = features; // Store features

        parts_.push_back(new_part);
        total_parts_++;

    }
    

    return 0; // Success
}

int Loader::createVolume(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
    PointCloudNT::Ptr& volume_cloud, 
    float thickness) {
    // Create volume implementation
    if (cloud->empty()) {
        std::cerr << "Error: Cloud is empty." << std::endl;
        return -1; // Error creating volume
    }
    
    volume_cloud->points.clear();
    
    // First, estimate normals for the input RGB cloud
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(10.0f); // 10mm radius for normal estimation
    normal_estimation.compute(*normals);
    
    // Takes a planar pointcloud and gives it a thickness, converting to PointNormal
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
        const auto& normal = normals->points[i];
        
        for (float j = 0; j < thickness; j += 1.0f) {
            PointNT new_point;
            new_point.x = point.x;
            new_point.y = point.y;
            new_point.z = point.z + j; // Apply thickness
            new_point.normal_x = normal.normal_x;
            new_point.normal_y = normal.normal_y;
            new_point.normal_z = normal.normal_z;
            new_point.curvature = normal.curvature;
            volume_cloud->points.push_back(new_point);
        }
    }
    volume_cloud->width = static_cast<uint32_t>(volume_cloud->points.size());
    volume_cloud->height = 1;
    volume_cloud->is_dense = true;

    return 0; // Success
}

int Loader::getFeatures(PointCloudNT::Ptr& cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr& features, pcl::VoxelGrid<PointNT>& voxel_filter) 
{
    features->clear();
    if (cloud->empty()) {
        std::cerr << "Error: Cloud is empty." << std::endl;
        return -1; // Error getting features
    }
    
    // Create a copy for downsampling
    int original_size = cloud->size();
    // Set appropriate voxel size based on cloud size (data is in mm)
    float leaf_size = 10.0f; // Default 10mm

    // Apply voxel grid filtering with appropriate leaf size
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.setInputCloud(cloud);
    voxel_filter.filter(*cloud);

    std::cout << "Downsampled from " << original_size << " to " << cloud->size()
              << " points (leaf size: " << leaf_size << "mm)" << std::endl;
    
    // Check if downsampling was effective
    if (cloud->empty()) {
        std::cerr << "Warning: Downsampled cloud is empty, using original cloud." << std::endl;
        cloud = cloud;
    }

    // Compute FPFH features using the PointNormal cloud directly
    pcl::FPFHEstimationOMP<PointNT, PointNT, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setInputCloud(cloud);
    fpfh_estimation.setInputNormals(cloud); // PointNormal clouds contain their own normals
    pcl::search::KdTree<PointNT>::Ptr tree(new pcl::search::KdTree<PointNT>);
    fpfh_estimation.setSearchMethod(tree);
    fpfh_estimation.setRadiusSearch(leaf_size * 5.0f); // Scale FPFH radius with leaf size
    fpfh_estimation.compute(*features);
    
    if (features->empty()) {
        std::cerr << "Error: No features computed." << std::endl;
        return -1; // Error getting features
    }
    
    std::cout << "Computed " << features->size() << " FPFH features." << std::endl;
    return 0; // Success
}

void Loader::visualizeParts(int part_index) {
    if (part_index < 0 || part_index >= parts_.size()) {
        std::cerr << "Invalid part index: " << part_index << std::endl;
        return;
    }
    std::cout << "Visualizing part: " << parts_[part_index].name << std::endl;

    // Convert PointNormal cloud to RGB cloud for visualization
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    rgb_cloud->points.resize(parts_[part_index].cloud->size());
    for (size_t i = 0; i < parts_[part_index].cloud->size(); ++i) {
        const auto& pt = parts_[part_index].cloud->points[i];
        rgb_cloud->points[i].x = pt.x;
        rgb_cloud->points[i].y = pt.y;
        rgb_cloud->points[i].z = pt.z;
        rgb_cloud->points[i].r = 255; // White color for visualization
        rgb_cloud->points[i].g = 255;
        rgb_cloud->points[i].b = 255;
    }
    rgb_cloud->width = parts_[part_index].cloud->width;
    rgb_cloud->height = parts_[part_index].cloud->height;
    rgb_cloud->is_dense = parts_[part_index].cloud->is_dense;

    pcl::visualization::CloudViewer viewer("Part Viewer");
    viewer.showCloud(rgb_cloud);
    while (!viewer.wasStopped()) {
        // Keep the viewer open until closed by the user
    }
}

