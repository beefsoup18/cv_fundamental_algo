#pragma once

#include <vector>
#include <Eigen/Dense>


std::vector<int> normalized_cut(std::vector<std::vector<double>> adjacency_matrix, int k);

Eigen::MatrixXd calculate_squared_euclidean_distance(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd calculate_similarity_matrix(const Eigen::MatrixXd& dists, double sigma);
Eigen::VectorXd calculate_degree_matrix(const Eigen::MatrixXd& W);
Eigen::MatrixXd calculate_laplacian_matrix(const Eigen::MatrixXd& W, const Eigen::VectorXd& D);
Eigen::MatrixXd solve_generalized_eigenproblem(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
Eigen::MatrixXd solve_specialized_eigenproblem(const Eigen::MatrixXd& A);