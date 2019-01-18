#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

Tools::Tools() {
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
		const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size

	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		return rmse;
	}

  for (int i = 0; i < estimations.size(); i++) {
    VectorXd temp = estimations[i] - ground_truth[i];
    temp = temp.array() * temp.array();
    rmse += temp;
  }

  // calculate the mean
  rmse /= estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();
  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = c2 * c1;
  // check division by zero

  // compute the Jacobian matrix

  Hj << px / c2, py / c2, 0, 0,
      -py / c1, px / c1, 0, 0,
      py * (vx * py - vy * px) / c3, px * (vy * px - vx * py) / c3, px / c2, py / c2;

  return Hj;
}
