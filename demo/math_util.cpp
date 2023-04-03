#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <Eigen/Dense>

using namespace std;
//using namespace Eigen;

Eigen::MatrixXd calculate_squared_euclidean_distance(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    // 计算欧氏距离的平方
    VectorXd normsA = A.rowwise().squaredNorm();
    VectorXd normsB = B.rowwise().squaredNorm();
    MatrixXd dists = normsA.replicate(1, B.rows()) + normsB.transpose().replicate(A.rows(), 1) - 2 * A * B.transpose();
    return dists;
}

Eigen::MatrixXd calculate_similarity_matrix(const Eigen::MatrixXd& dists, double sigma) {
    // 构建相似矩阵W
    Eigen::MatrixXd W = exp(-dists.array() / pow(sigma, 2));
    W.diagonal().setZero();
    return W;
}

Eigen::VectorXd calculate_degree_matrix(const Eigen::MatrixXd& W) {
    // 计算度矩阵D
    Eigen::VectorXd D = W.rowwise().sum();
    return D;
}

Eigen::MatrixXd calculate_laplacian_matrix(const Eigen::MatrixXd& W, const Eigen::VectorXd& D) {
    // 计算拉普拉斯矩阵L=D-W
    MatrixXd L = D.asDiagonal() - W;
    return L;
}

Eigen::MatrixXd solve_generalized_eigenproblem(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    // 解广义特征值问题 A*x = lambda*B*x，返回特征向量矩阵
    SelfAdjointEigenSolver<Eigen::MatrixXd> Eigen::eigensolver(A, B);
    Eigen::MatrixXd vecs = Eigen::eigensolver.eigenvectors();
    return vecs;
}


Eigen::MatrixXd solve_specialized_eigenproblem(const Eigen::MatrixXd& A) {
    // 解狭义特征值问题 A*x = lambda*B*x，返回特征向量矩阵
    SelfAdjointEigenSolver<Eigen::MatrixXd> Eigen::eigensolver(A);
    Eigen::VectorXd vecs = Eigen::eigensolver.eigenvectors();
    return vecs;
}

