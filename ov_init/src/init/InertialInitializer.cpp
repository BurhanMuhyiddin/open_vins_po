/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "InertialInitializer.h"

#ifndef __ANDROID__
#include "dynamic/DynamicInitializer.h"
#endif
#include "static/StaticInitializer.h"

#include "feat/FeatureHelper.h"
#include "types/Type.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"
#include "utils/helper.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_init;

InertialInitializer::InertialInitializer(InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db)
    : params(params_), _db(db) {

  // Vector of our IMU data
  imu_data = std::make_shared<std::vector<ov_core::ImuData>>();

  // Create initializers
  init_static = std::make_shared<StaticInitializer>(params, _db, imu_data);
#ifndef __ANDROID__
  init_dynamic = std::make_shared<DynamicInitializer>(params, _db, imu_data);
#else
  init_dynamic = nullptr;
#endif
  Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
  Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
  Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
  pDrtVioInit = std::make_shared<DRT::drtLooselyCoupled>(R_ItoC.transpose(), -1.0 * R_ItoC.transpose() * p_IinC);
}

void InertialInitializer::feed_imu(const ov_core::ImuData &message, double oldest_time) {

  // Append it to our vector
  imu_data->emplace_back(message);

  // Sort our imu data (handles any out of order measurements)
  // std::sort(imu_data->begin(), imu_data->end(), [](const IMUDATA i, const IMUDATA j) {
  //    return i.timestamp < j.timestamp;
  //});

  // Loop through and delete imu messages that are older than our requested time
  // std::cout << "INIT: imu_data.size() " << imu_data->size() << std::endl;
  if (oldest_time != -1) {
    auto it0 = imu_data->begin();
    while (it0 != imu_data->end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data->erase(it0);
      } else {
        it0++;
      }
    }
  }
}

bool InertialInitializer::initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                                     std::shared_ptr<ov_type::IMU> t_imu, bool wait_for_jerk) {
  // DRT: check parallax and add feature
  double newest_cam_time = -1;
  for (auto const &feat : _db->get_internal_data()) {
    for (auto const &camtimepair : feat.second->timestamps) {
      for (auto const &time : camtimepair.second) {
        newest_cam_time = std::max(newest_cam_time, time);
      }
    }
  }

  const double sf = sqrt(200);
  
  if (init_feature)
  {
    init_feature = false;
    return false;
  }

  if (is_resetted)
  {
    for (const auto& k_timestamp: keyframe_timestamps)
    {
      Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1 >> >>
                    image;
      for (const auto &feat : _db->get_internal_data()) {
        if (feat.second->timestamps[0].size() > 1) {
          auto feat_index = std::find(feat.second->timestamps[0].begin(), feat.second->timestamps[0].end(), k_timestamp);
          if (feat_index == feat.second->timestamps[0].end())
          {
            continue;
          }
          unsigned int i = std::distance(feat.second->timestamps[0].begin(), feat_index);
          int feature_id = feat.second->featid;
          int camera_id = 0;
          Eigen::Matrix< double, 3, 1 > f_Ci;
          f_Ci << feat.second->uvs_norm[0][i](0), feat.second->uvs_norm[0][i](1), 1;
          f_Ci /= f_Ci.norm();
          f_Ci /= f_Ci(2);
          double x = f_Ci(0);
          double y = f_Ci(1);
          double z = 1;
          double p_u = feat.second->uvs[0][i].x();
          double p_v = feat.second->uvs[0][i].y();
          double velocity_x = 0;
          double velocity_y = 0;
          Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
          xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
          image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
      }

      pDrtVioInit->addFeatureCheckParallax(k_timestamp, image, 0.0, true);
    }

    is_resetted = false;
  }

  Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1 >> >>
                    image;
  for (const auto &feat : _db->get_internal_data()) {
      if (feat.second->timestamps[0].size() > 1) {
        auto feat_index = std::find(feat.second->timestamps[0].begin(), feat.second->timestamps[0].end(), newest_cam_time);
        if (feat_index == feat.second->timestamps[0].end())
        {
          continue;
        }
        unsigned int i = std::distance(feat.second->timestamps[0].begin(), feat_index);
        int feature_id = feat.second->featid;
        int camera_id = 0;
        Eigen::Matrix< double, 3, 1 > f_Ci;
        f_Ci << feat.second->uvs_norm[0][i](0), feat.second->uvs_norm[0][i](1), 1;
        f_Ci /= f_Ci.norm();
        f_Ci /= f_Ci(2);
        double x = f_Ci(0);
        double y = f_Ci(1);
        double z = 1;
        double p_u = feat.second->uvs[0][i].x();
        double p_v = feat.second->uvs[0][i].y();
        double velocity_x = 0;
        double velocity_y = 0;
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
      }
  }

  if (pDrtVioInit->addFeatureCheckParallax(newest_cam_time, image, 0.0))
  {
    keyframe_timestamps.push_back(newest_cam_time);
  }

  if (keyframe_timestamps.size() < 10)
  {
    return false;
  }

  for (int j = 1; j < keyframe_timestamps.size(); j++)
  {
    double last_keyframe_timestamp_in_imu = keyframe_timestamps[j-1] + params.calib_camimu_dt;
    double current_keyframe_timestamp_in_imu = keyframe_timestamps[j] + params.calib_camimu_dt;
    std::vector<ov_core::ImuData> imu_segment =
        InitializerHelper::select_imu_readings(*imu_data, last_keyframe_timestamp_in_imu, current_keyframe_timestamp_in_imu);

    Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
    Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
    Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
    vio::IMUBias bias;
    vio::IMUCalibParam
            imu_calib(R_ItoC, p_IinC, params.sigma_w * sf, params.sigma_a * sf, params.sigma_wb / sf, params.sigma_ab / sf);
    vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_keyframe_timestamp_in_imu, current_keyframe_timestamp_in_imu);

    int n = imu_segment.size() - 1;

    for (int i = 0; i < n; i++) {
      double dt;
      Eigen::Vector3d gyro;
      Eigen::Vector3d acc;

      acc = imu_segment[i].am;
      gyro = imu_segment[i].wm;

      if (i == 0 && i < (n - 1))               // [start_time, imu[0].time]
      {
        dt = imu_segment[i + 1].timestamp - last_keyframe_timestamp_in_imu;
      } else if (i < (n - 1))      // [imu[i].time, imu[i+1].time]
      {
        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
      } else if (i > 0 && i == n - 1) {
        dt = current_keyframe_timestamp_in_imu - imu_segment[i].timestamp;
      } else if (i == 0 && i == (n - 1)) {
        dt = current_keyframe_timestamp_in_imu - last_keyframe_timestamp_in_imu;
      }
      CHECK(dt >= 0);
      imu_preint.integrate_new_measurement(gyro, acc, dt);
    }

    pDrtVioInit->addImuMeasure(imu_preint);
  }

  bool is_good = pDrtVioInit->checkAccError();

  if (!is_good)
  {
    std::cout << "(checkAccError): couldn't initialize... resetting everything...\n";
    is_resetted = true;
    // remove imu measurements and features before the oldest timestamp
    double oldest_keyframe_timestamp = keyframe_timestamps[0];
    keyframe_timestamps.erase(keyframe_timestamps.begin());

    _db->cleanup_measurements(oldest_keyframe_timestamp);
    auto it_imu = imu_data->begin();
    while (it_imu != imu_data->end() && it_imu->timestamp < oldest_keyframe_timestamp + params.calib_camimu_dt) {
      it_imu = imu_data->erase(it_imu);
    }

    // reset initialzier
    Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
    Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
    Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
    pDrtVioInit = std::make_shared<DRT::drtLooselyCoupled>(R_ItoC.transpose(), -1.0 * R_ItoC.transpose() * p_IinC);

    return false;
  }

  if ( !pDrtVioInit->process())
  {
    std::cout << "(process): couldn't initialize... resetting everything...\n";
    is_resetted = true;
    // remove imu measurements and features before the oldest timestamp
    double oldest_keyframe_timestamp = keyframe_timestamps[0];
    keyframe_timestamps.erase(keyframe_timestamps.begin());

    _db->cleanup_measurements(oldest_keyframe_timestamp);
    auto it_imu = imu_data->begin();
    while (it_imu != imu_data->end() && it_imu->timestamp < oldest_keyframe_timestamp + params.calib_camimu_dt) {
      it_imu = imu_data->erase(it_imu);
    }

    // reset initialzier
    Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
    Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
    Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);
    pDrtVioInit = std::make_shared<DRT::drtLooselyCoupled>(R_ItoC.transpose(), -1.0 * R_ItoC.transpose() * p_IinC);

    return false;
  }

  std::cout << "I initialized at time: " << std::fixed << newest_cam_time << "\n";
  std::cout << "gravity direction: " << pDrtVioInit->gravity.transpose() << "\n";
  std::cout << "initialized bias: " << pDrtVioInit->biasg.transpose() << "\n";
  std::cout << "initialized rot: " << pDrtVioInit->rotation.back() << "\n";
  std::cout << "initialized pos: " << pDrtVioInit->position.back().transpose() << "\n";
  std::cout << "initialized vel: " << pDrtVioInit->velocity.back().transpose() << "\n";
  // std::cout << "intiialzied velocity to:\n";
  // for (const auto velocity : pDrtVioInit->velocity)
  // {
  //   std::cout << "norm: " << velocity.norm() << ", vector: " << velocity.transpose() << "\n";
  // }

  // Get rotation with z axis aligned with -g (z_in_G=0,0,1)
  // Eigen::Vector3d z_axis = a_avg_2to1 / a_avg_2to1.norm();
  Eigen::Vector4d q_ItoC = params.camera_extrinsics.at(0).block(0, 0, 4, 1);
  Eigen::Vector3d p_IinC = params.camera_extrinsics.at(0).block(4, 0, 3, 1);
  Eigen::Matrix3d R_ItoC = quat_2_Rot(q_ItoC);

  Eigen::Matrix3d Ro;
  InitializerHelper::gram_schmidt(pDrtVioInit->gravity, Ro);
  // Eigen::Vector4d q_GtoI = rot_2_quat(Ro.transpose());
  // std::cout << "efheshfkjsfkjas: " << q_GtoI.transpose() << "\n";

  // std::cout << "dskjfhkjsf: " << q_GtoI.transpose() << "\n";
  // std::cout << "gravity direction after: " << (Ro * pDrtVioInit->gravity).transpose() << "\n";

  // // Set our biases equal to our noise (subtract our gravity from accelerometer bias)
  // Eigen::Vector3d gravity_inG;
  // gravity_inG << 0.0, 0.0, params.gravity_mag;

  // Eigen::Vector3d g0_body = pDrtVioInit->gravity;
  // Eigen::Vector3d z = g0_body.normalized();
  // Eigen::Vector3d x = Eigen::Vector3d(1,0,0) - z * z.dot(Eigen::Vector3d(1,0,0));
  // x.normalize();
  // Eigen::Vector3d y = z.cross(x);
  // y.normalize();

  Eigen::Matrix3d R_aligning;
  // R_aligning.row(0) = x.transpose();
  // R_aligning.row(1) = y.transpose();
  // R_aligning.row(2) = z.transpose();

  R_aligning = Ro;

  // std::cout << "aligning rot:\n" << R_aligning << "\n";
  // std::cout << "gravity direction before: " << pDrtVioInit->gravity.transpose() << "\n";
  // std::cout << "gravity direction after: " << (R_aligning * pDrtVioInit->gravity).transpose() << "\n";

  Eigen::Vector3d bg = pDrtVioInit->biasg;
  Eigen::Vector3d ba = pDrtVioInit->biasa; //- quat_2_Rot(q_GtoI) * gravity_inG;

  // Set our state variables
  timestamp = keyframe_timestamps.back();
  Eigen::VectorXd imu_state = Eigen::VectorXd::Zero(16);
  // imu_state.block(0, 0, 4, 1) = rot_2_quat(pDrtVioInit->rotation.back().transpose() * R_aligning);
  Eigen::Matrix3d R_GtoI = pDrtVioInit->rotation.back().transpose() * R_aligning; 
  imu_state.block(0, 0, 4, 1) = rot_2_quat(R_GtoI);
  // imu_state.block(4, 0, 3, 1) = R_aligning.transpose() * pDrtVioInit->position.back();
  imu_state.block(7, 0, 3, 1) = R_aligning.transpose() * pDrtVioInit->velocity.back();
  imu_state.block(10, 0, 3, 1) = bg;
  imu_state.block(13, 0, 3, 1) = ba;
  assert(t_imu != nullptr);
  t_imu->set_value(imu_state);
  t_imu->set_fej(imu_state);

  // Create base covariance and its covariance ordering
  order.clear();
  order.push_back(t_imu);
  covariance = Eigen::MatrixXd::Identity(t_imu->size(), t_imu->size());
  covariance.block(0, 0, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity(); // q
  covariance.block(3, 3, 3, 3) = std::pow(0.10, 2) * Eigen::Matrix3d::Identity(); // p
  covariance.block(6, 6, 3, 3) = std::pow(0.20, 2) * Eigen::Matrix3d::Identity(); // v (static)
  covariance.block(9, 9, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();
  covariance.block(12, 12, 3, 3) = std::pow(0.30, 2) * Eigen::Matrix3d::Identity();

  timestamp += params.calib_camimu_dt;

  return true;

//   // Get the newest and oldest timestamps we will try to initialize between!
//   double newest_cam_time = -1;
//   for (auto const &feat : _db->get_internal_data()) {
//     for (auto const &camtimepair : feat.second->timestamps) {
//       for (auto const &time : camtimepair.second) {
//         newest_cam_time = std::max(newest_cam_time, time);
//       }
//     }
//   }
//   double oldest_time = newest_cam_time - params.init_window_time - 0.10;
//   if (newest_cam_time < 0 || oldest_time < 0) {
//     return false;
//   }

//   // Remove all measurements that are older then our initialization window
//   // Then we will try to use all features that are in the feature database!
//   _db->cleanup_measurements(oldest_time);
//   auto it_imu = imu_data->begin();
//   while (it_imu != imu_data->end() && it_imu->timestamp < oldest_time + params.calib_camimu_dt) {
//     it_imu = imu_data->erase(it_imu);
//   }

//   // Compute the disparity of the system at the current timestep
//   // If disparity is zero or negative we will always use the static initializer
//   bool disparity_detected_moving_1to0 = false;
//   bool disparity_detected_moving_2to1 = false;
//   if (params.init_max_disparity > 0) {

//     // Get the disparity statistics from this image to the previous
//     // Only compute the disparity for the oldest half of the initialization period
//     double newest_time_allowed = newest_cam_time - 0.5 * params.init_window_time;
//     int num_features0 = 0;
//     int num_features1 = 0;
//     double avg_disp0, avg_disp1;
//     double var_disp0, var_disp1;
//     FeatureHelper::compute_disparity(_db, avg_disp0, var_disp0, num_features0, newest_time_allowed);
//     FeatureHelper::compute_disparity(_db, avg_disp1, var_disp1, num_features1, newest_cam_time, newest_time_allowed);

//     // Return if we can't compute the disparity
//     int feat_thresh = 15;
//     if (num_features0 < feat_thresh || num_features1 < feat_thresh) {
//       PRINT_WARNING(YELLOW "[init]: not enough feats to compute disp: %d,%d < %d\n" RESET, num_features0, num_features1, feat_thresh);
//       return false;
//     }

//     // Check if it passed our check!
//     PRINT_INFO(YELLOW "[init]: disparity is %.3f,%.3f (%.2f thresh)\n" RESET, avg_disp0, avg_disp1, params.init_max_disparity);
//     disparity_detected_moving_1to0 = (avg_disp0 > params.init_max_disparity);
//     disparity_detected_moving_2to1 = (avg_disp1 > params.init_max_disparity);
//   }

//   // Use our static initializer!
//   // CASE1: if our disparity says we were static in last window and have moved in the newest, we have a jerk
//   // CASE2: if both disparities are below the threshold, then the platform has been stationary during both periods
//   bool has_jerk = (!disparity_detected_moving_1to0 && disparity_detected_moving_2to1);
//   bool is_still = (!disparity_detected_moving_1to0 && !disparity_detected_moving_2to1);
//   if (((has_jerk && wait_for_jerk) || (is_still && !wait_for_jerk)) && params.init_imu_thresh > 0.0) {
//     PRINT_DEBUG(GREEN "[init]: USING STATIC INITIALIZER METHOD!\n" RESET);
//     return init_static->initialize(timestamp, covariance, order, t_imu, wait_for_jerk);
//   } else if (params.init_dyn_use && !is_still) {
// #ifndef __ANDROID__
//     PRINT_DEBUG(GREEN "[init]: USING DYNAMIC INITIALIZER METHOD!\n" RESET);
//     std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;
//     std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;
//     if (init_dynamic) {
//       return init_dynamic->initialize(timestamp, covariance, order, t_imu, _clones_IMU, _features_SLAM);
//     }
// #else
//     PRINT_ERROR(RED "[init]: DYNAMIC INITIALIZER not available on Android (Ceres Solver not included)\n" RESET);
// #endif
//   } else {
//     std::string msg = (has_jerk) ? "" : "no accel jerk detected";
//     msg += (has_jerk || is_still) ? "" : ", ";
//     msg += (is_still) ? "" : "platform moving too much";
//     PRINT_INFO(YELLOW "[init]: failed static init: %s\n" RESET, msg.c_str());
//   }
//   return false;
}
