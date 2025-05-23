// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "KissICP.hpp"

#include <Eigen/Core>
#include <vector>
#include <iostream>

#include "kiss_icp/core/Preprocessing.hpp"
#include "kiss_icp/core/Registration.hpp"
#include "kiss_icp/core/VoxelHashMap.hpp"

namespace kiss_icp::pipeline {

KissICP::Vector3dVectorTuple KissICP::RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                                                    const std::vector<double> &timestamps, 
                                                    const Sophus::SE3d &external_guess) {

    if (is_first_frame_) {
        last_pose_ = external_guess;
        is_first_frame_ = false;
        // std::cout << "[KissICP] Initialized last_pose_ from external guess." << std::endl;
    }

    // Preprocess the input cloud
    const auto &preprocessed_frame = preprocessor_.Preprocess(frame, timestamps, last_delta_);

    // Voxelize
    const auto &[source, frame_downsample] = Voxelize(preprocessed_frame);

    // Get adaptive_threshold
    const double sigma = adaptive_threshold_.ComputeThreshold();

    // Compute initial_guess for ICP
    // const auto initial_guess = last_pose_ * last_delta_;

    // std::cout << "New iteration" << std::endl;

    // std::cout << "Sigma: " << sigma << std::endl;
    // std::cout << "External guess: " << std::endl;
    // std::cout << external_guess.matrix() << std::endl;

    // Run ICP
    const auto new_pose = registration_.AlignPointsToMap(source,         // frame
                                                         local_map_,     // voxel_map
                                                         external_guess,  // external_guess
                                                         1.0 * sigma,    // max_correspondence_dist
                                                         sigma);         // kernel
    
    // std::cout << "new_pose: " << std::endl;
    // std::cout << new_pose.matrix() << std::endl;

    // Compute the difference between the prediction and the actual estimate
    const auto model_deviation = external_guess.inverse() * new_pose;
    // std::cout << "Model deviation: " << std::endl;
    // std::cout << model_deviation.matrix() << std::endl;

    // std::cout << "last_pose_: " << std::endl;
    // std::cout << last_pose_.matrix() << std::endl;

    // Update step: threshold, local map, delta, and the last pose
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    local_map_.Update(frame_downsample, new_pose);
    last_delta_ = last_pose_.inverse() * new_pose;
    last_pose_ = new_pose;

    // std::cout << "last_delta_: " << std::endl;
    // std::cout << last_delta_.matrix() << std::endl;
    
    // const double sigma2 = adaptive_threshold_.ComputeThreshold();
    // std::cout << "Sigma2: " << sigma2 << std::endl;

    // Return the (deskew) input raw scan (preprocessed_frame) and the points used for registration
    // (source)
    return {preprocessed_frame, source};
}

KissICP::Vector3dVectorTuple KissICP::Voxelize(const std::vector<Eigen::Vector3d> &frame) const {
    const auto voxel_size = config_.voxel_size;
    const auto frame_downsample = kiss_icp::VoxelDownsample(frame, voxel_size * 0.05);
    const auto source = kiss_icp::VoxelDownsample(frame_downsample, voxel_size * 0.6);
    return {source, frame_downsample};
}

}  // namespace kiss_icp::pipeline
