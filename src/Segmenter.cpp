#include "../inc/Segmenter.h"
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/common/time.h>




Segmenter::Segmenter() {
    // Constructor implementation
    working_clouds_ = std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>();
    working_cloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_size_ = 0;
    part_loader_ = std::make_unique<Loader>();
}

void Segmenter::loadPointCloud(const std::string& filename) {

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *working_cloud_) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return;
    }
    std::cout << "Loaded " << working_cloud_->size() << " points from " << filename << std::endl;


    cloud_size_ = working_cloud_->size();

    if (cloud_size_ >= 1000000) {
        std::cout << "Point cloud is large enough for voxelization." << std::endl;
        voxelizePointCloud(working_cloud_, 5.0f);
        std::cout << "Filtered cloud from " << cloud_size_ << " to " << working_cloud_->size() << " points." << std::endl;
        cloud_size_ = working_cloud_->size();
        // Save the filtered cloud if needed
        std::string new_filename = filename.substr(0, filename.find_last_of('.')) + "_filtered.pcd";
        pcl::io::savePCDFile(new_filename, *working_cloud_);
        std::cout << "Filtered cloud saved to " << new_filename << std::endl;

    } else {
        std::cout << "Point cloud has probably already been filtered." << std::endl;
    }

    visualizePointCloud(working_cloud_);


    working_cloud_ = cropInnerRectangle(working_cloud_, 530, 800, 300, 800);

    working_cloud_ = filterOutTable(working_cloud_);

    // Cluster extraction
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(working_cloud_);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(10.0); // 10mm
    ec.setMinClusterSize(500); // Minimum cluster size
    ec.setSearchMethod(tree);
    ec.setInputCloud(working_cloud_);
    ec.extract(cluster_indices);

    std::cout << "Extracted " << cluster_indices.size() << " clusters." << std::endl;

    // Color points based on cluster assignments
    colorPointsByCluster(cluster_indices);

    // Visualize the working cloud
    visualizePointCloud(working_cloud_);

    // Store the each of the clusters in working_clouds_
    working_clouds_.clear();
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (const auto& index : indices.indices) {
            cluster->points.push_back(working_cloud_->points[index]);
        }
        working_clouds_.push_back(cluster);
    }
    std::cout << "Stored " << working_clouds_.size() << " clusters in working_clouds_." << std::endl;
}

void Segmenter::visualizePointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    std::cout << "Visualizing point cloud with " << cloud->size() << " points." << std::endl;
    // Removed changeColorToWhite() to preserve cluster colors
    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {
        // Keep the viewer open until closed by the user
    }
}

void Segmenter::segment(){

    for (const auto& cloud : working_clouds_) {
        segmentPointCloud(cloud);
    }

}

void Segmenter::segmentPointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) 
{

    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {
        // Keep the viewer open until closed by the user
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> part_clouds;
    part_clouds.reserve(part_loader_->getTotalParts());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloudNT::Ptr scene_cloud(new PointCloudNT);
    PointCloudNT::Ptr object_aligned (new PointCloudNT);


    // DownSample
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    const float leaf_size = 10.0f; 
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.setInputCloud(cloud);
    voxel_filter.filter(*scene_cloud_rgb);

    std::cout << "Downsampled scene cloud from " << cloud->size() << " to " << scene_cloud_rgb->size() << " points." << std::endl;

    // Estimate normals for the downsampled scene
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(scene_cloud_rgb);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    normal_estimation.setSearchMethod(tree);
    normal_estimation.setRadiusSearch(leaf_size * 3.0f); // Scale normal estimation radius with leaf size
    normal_estimation.compute(*normals);

    // Convert RGB cloud to PointNormal cloud for registration
    scene_cloud->points.resize(scene_cloud_rgb->size());
    for (size_t i = 0; i < scene_cloud_rgb->size(); ++i) {
        scene_cloud->points[i].x = scene_cloud_rgb->points[i].x;
        scene_cloud->points[i].y = scene_cloud_rgb->points[i].y;
        scene_cloud->points[i].z = scene_cloud_rgb->points[i].z;
        scene_cloud->points[i].normal_x = normals->points[i].normal_x;
        scene_cloud->points[i].normal_y = normals->points[i].normal_y;
        scene_cloud->points[i].normal_z = normals->points[i].normal_z;
        scene_cloud->points[i].curvature = normals->points[i].curvature;
    }
    scene_cloud->width = scene_cloud_rgb->width;
    scene_cloud->height = scene_cloud_rgb->height;
    scene_cloud->is_dense = scene_cloud_rgb->is_dense;

    // Feature estimation
    pcl::PointCloud<FeatureT>::Ptr features(new pcl::PointCloud<FeatureT>);
    
    // Compute FPFH features using the PointNormal cloud
    pcl::FPFHEstimationOMP<PointNT, PointNT, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setInputCloud(scene_cloud);
    fpfh_estimation.setInputNormals(scene_cloud);
    pcl::search::KdTree<PointNT>::Ptr tree_nt(new pcl::search::KdTree<PointNT>);
    fpfh_estimation.setSearchMethod(tree_nt);
    fpfh_estimation.setRadiusSearch(leaf_size * 5.0f); // Scale FPFH radius with leaf size
    fpfh_estimation.compute(*features);

    std::cout << "Computed FPFH features for scene cloud." << std::endl;

    for (const auto& object : part_loader_->getParts()) {
        pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
        align.setInputSource(object.cloud);
        align.setSourceFeatures(object.features);
        align.setInputTarget(scene_cloud);
        align.setTargetFeatures(features);
        align.setMaximumIterations (50000); // Number of RANSAC iterations
        align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
        align.setCorrespondenceRandomness (5); // Number of nearest features to use
        align.setSimilarityThreshold (0.95f); // Polygonal edge length similarity threshold
        align.setMaxCorrespondenceDistance (2.5f * leaf_size); // Inlier threshold
        align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
        {
            pcl::ScopeTime t("Alignment");
            align.align (*object_aligned);
        }
        if (align.hasConverged ())
        {
            // Print results
            printf ("\n");
            Eigen::Matrix4f transformation = align.getFinalTransformation ();
            pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
            pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
            pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
            pcl::console::print_info ("\n");
            pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
            pcl::console::print_info ("\n");
            pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object.cloud->size ());

            // Convert PointNormal aligned object to RGB for visualization
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
            aligned_rgb->points.resize(object_aligned->size());
            for (size_t i = 0; i < object_aligned->size(); ++i) {
                aligned_rgb->points[i].x = object_aligned->points[i].x;
                aligned_rgb->points[i].y = object_aligned->points[i].y;
                aligned_rgb->points[i].z = object_aligned->points[i].z;
                aligned_rgb->points[i].r = 255;
                aligned_rgb->points[i].g = 0;
                aligned_rgb->points[i].b = 0;
            }
            aligned_rgb->width = object_aligned->width;
            aligned_rgb->height = object_aligned->height;
            aligned_rgb->is_dense = object_aligned->is_dense;

            visualizeAlignment(scene_cloud_rgb, *aligned_rgb, object.name, align.getFitnessScore());
        }
    }
    

}

void Segmenter::changeColorToWhite() {
    for (auto& point : working_cloud_->points) {
        point.r = static_cast<uint8_t>(255);
        point.g = static_cast<uint8_t>(255);
        point.b = static_cast<uint8_t>(255);
    }
}

void Segmenter::colorPointsByCluster(const std::vector<pcl::PointIndices>& cluster_indices) {
    // First, color all points black (background)
    for (auto& point : working_cloud_->points) {
        point.r = 0;
        point.g = 0;
        point.b = 0;
    }
    
    // Define a set of distinct colors for clusters
    std::vector<std::array<uint8_t, 3>> colors = {
        {255, 0, 0},     // Red
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 255, 0},   // Yellow
        {255, 0, 255},   // Magenta
        {0, 255, 255},   // Cyan
        {255, 128, 0},   // Orange
        {128, 0, 255},   // Purple
        {255, 192, 203}, // Pink
        {0, 128, 128},   // Teal
        {128, 128, 0},   // Olive
        {128, 0, 128},   // Maroon
        {255, 165, 0},   // Dark Orange
        {173, 255, 47},  // Green Yellow
        {70, 130, 180},  // Steel Blue
        {255, 20, 147},  // Deep Pink
        {0, 191, 255},   // Deep Sky Blue
        {255, 215, 0},   // Gold
        {220, 20, 60},   // Crimson
        {75, 0, 130}     // Indigo
    };
    
    std::cout << "Coloring " << cluster_indices.size() << " clusters..." << std::endl;
    
    // Color each cluster with a different color
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        const auto& cluster = cluster_indices[i];
        const auto& color = colors[i % colors.size()]; // Cycle through colors if more clusters than colors
        
        std::cout << "Cluster " << i << ": " << cluster.indices.size() << " points - "
                  << "RGB(" << static_cast<int>(color[0]) << "," 
                  << static_cast<int>(color[1]) << "," 
                  << static_cast<int>(color[2]) << ")" << std::endl;
        
        for (const auto& index : cluster.indices) {
            working_cloud_->points[index].r = color[0];
            working_cloud_->points[index].g = color[1];
            working_cloud_->points[index].b = color[2];
        }
    }
}

void Segmenter::voxelizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float leaf_size) {
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    voxel_filter.filter(*filtered_cloud);
    std::cout << "Filtered cloud size: " << filtered_cloud->size() << std::endl;
    working_cloud_ = filtered_cloud;
}

void Segmenter::visualizeAlignment(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& scene_cloud,
                                  const pcl::PointCloud<pcl::PointXYZRGB>& aligned_part,
                                  const std::string& part_name,
                                  double fitness_score) {
    // Create a copy of the aligned part as a pointer
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_part_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(aligned_part));
    
    // Color the scene cloud in white/gray
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_scene(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*scene_cloud, *colored_scene);
    for (auto& point : colored_scene->points) {
        point.r = 200;  // Light gray
        point.g = 200;
        point.b = 200;
    }
    
    // Color the aligned part in bright red for visibility
    for (auto& point : aligned_part_ptr->points) {
        point.r = 255;  // Bright red
        point.g = 0;
        point.b = 0;
    }
    
    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Alignment Visualization"));
    viewer->setBackgroundColor(0, 0, 0);
    
    // Add scene cloud in gray
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_scene, "scene");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "scene");
    
    // Add aligned part in red
    viewer->addPointCloud<pcl::PointXYZRGB>(aligned_part_ptr, "aligned_part");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_part");
    
    // Add text with part info
    std::string info_text = "Part: " + part_name + "\nFitness Score: " + std::to_string(fitness_score);
    viewer->addText(info_text, 10, 10, "info_text");
    
    // Add coordinate system
    viewer->addCoordinateSystem(10.0);
    viewer->initCameraParameters();
    
    std::cout << "Press 'q' to close the visualization and continue processing..." << std::endl;
    
    // Show until user closes
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Segmenter::cropInnerRectangle(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud_in,
	double cut_min_x_mm,
	double cut_max_x_mm,
	double cut_min_y_mm,
	double cut_max_y_mm)
{
	using CloudT = pcl::PointCloud<pcl::PointXYZRGB>;
	auto cloud_out = CloudT::Ptr(new CloudT); // Copy full structure

	if (!cloud_in || cloud_in->empty()) return cloud_out;

	// Get bounds
	float min_x = std::numeric_limits<float>::max();
	float max_x = -std::numeric_limits<float>::max();
	float min_y = std::numeric_limits<float>::max();
	float max_y = -std::numeric_limits<float>::max();

	for (const auto& p : cloud_in->points)
	{
		if (!pcl::isFinite(p)) continue;
		min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
		min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
	}

	// Apply margins
	min_x += static_cast<float>(cut_min_x_mm);
	max_x -= static_cast<float>(cut_max_x_mm);
	min_y += static_cast<float>(cut_min_y_mm);
	max_y -= static_cast<float>(cut_max_y_mm);

	if (min_x >= max_x || min_y >= max_y)
		return cloud_out;

	// Iterate and only add points within the rectangle
    cloud_out->points.clear();
	for (const auto& p : cloud_in->points)
	{
		if (!pcl::isFinite(p)) continue;
		if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y)
		{
			continue; // Skip points outside the rectangle
		}
		else
		{
			cloud_out->points.push_back(p);
		}
	}
		

	cloud_out->is_dense = false;
	return cloud_out;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Segmenter::filterOutTable(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud_in) {
    using CloudT = pcl::PointCloud<pcl::PointXYZRGB>;
    auto cloud_out = CloudT::Ptr(new CloudT);
    
    if (!cloud_in || cloud_in->empty()) {
        std::cout << "Input cloud is empty for table filtering." << std::endl;
        return cloud_out;
    }
    
    std::cout << "Starting table filtering on " << cloud_in->size() << " points..." << std::endl;
    
    // Create segmentation object for plane detection
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    
    // Configure plane segmentation
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(5.0f); // 5mm tolerance for plane detection

    // Find the largest plane (presumably the table)
    seg.setInputCloud(cloud_in);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.size() == 0) {
        std::cout << "No plane detected. Returning original cloud." << std::endl;
        pcl::copyPointCloud(*cloud_in, *cloud_out);
        return cloud_out;
    }
    
    std::cout << "Detected plane with " << inliers->indices.size() << " inliers." << std::endl;
    std::cout << "Plane equation: " << coefficients->values[0] << "x + " 
              << coefficients->values[1] << "y + " 
              << coefficients->values[2] << "z + " 
              << coefficients->values[3] << " = 0" << std::endl;
    
    // Extract points that are NOT on the table plane
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers);
    extract.setNegative(true); // Extract everything EXCEPT the plane
    extract.filter(*cloud_out);
    
    // Additional filtering: remove points that are too close to the plane
    // This helps remove small objects sitting directly on the table
    CloudT::Ptr final_cloud(new CloudT);
    final_cloud->points.clear();

    float min_height_above_plane = 0.0f; // Minimum 1mm above the plane

    for (const auto& point : cloud_out->points) {
        if (!pcl::isFinite(point)) continue;
        
        // Calculate distance from point to plane
        float distance = -(coefficients->values[0] * point.x + 
                                 coefficients->values[1] * point.y + 
                                 coefficients->values[2] * point.z + 
                                 coefficients->values[3]);
        
        // Keep points that are sufficiently far from the plane
        if (distance > min_height_above_plane) {
            final_cloud->points.push_back(point);
        }
    }
    
    final_cloud->width = static_cast<uint32_t>(final_cloud->points.size());
    final_cloud->height = 1;
    final_cloud->is_dense = false;
    
    std::cout << "Table filtering complete. Reduced from " << cloud_in->size() 
              << " to " << final_cloud->size() << " points." << std::endl;
    
    return final_cloud;
}

int main() {
    std::cout << "Segmenter application started" << std::endl;
    
    // Create a Segmenter instance
    Segmenter segmenter;

    // Load a point cloud file
    std::string filename = "../test_11571.pcd"; // Replace with your PCD file path
    segmenter.loadPointCloud(filename);

    segmenter.part_loader_->loadParts("../Projects/0003"); // Load parts from a Folder

    segmenter.segment();
    return 0;
}