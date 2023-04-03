#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
// using namespace Eigen;


// 计算欧式距离
double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (int i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

// 计算高斯核函数值
double gaussian_kernel(double x, double y, double sigma) {
    return exp(-pow(x - y, 2) / (2 * pow(sigma, 2)));
}

// 计算相似度矩阵
Eigen::MatrixXd compute_similarity_matrix(const Eigen::MatrixXd& data_matrix, double sigma) {
    // 获取样本数
    int n_sample = data_matrix.rows();

    // 初始化相似度矩阵
    Eigen::MatrixXd similarity_matrix(n_sample, n_sample);

    // 计算相似度矩阵
    for (int i = 0; i < n_sample; ++i) {
        for (int j = i + 1; j < n_sample; ++j) {
            // double similarity_ij = euclidean_distance(data_matrix.row(i).column())
            double similarity_ij = gaussian_kernel(data_matrix.row(i).norm(), data_matrix.row(j).norm(), sigma);
            similarity_matrix(i * n_rows + j, j) = similarity_ij;
            similarity_matrix(j, i) = similarity_ij;
        }
    }

    return similarity_matrix;
}

// 计算归一化拉普拉斯矩阵
Eigen::MatrixXd compute_normalized_laplacian(const Eigen::MatrixXd& similarity_matrix) {
    // 计算度矩阵
    Eigen::VectorXd degree_vector = similarity_matrix.colwise().sum();
    Eigen::MatrixXd degree_matrix = degree_vector.asDiagonal();

    // 计算拉普拉斯矩阵
    Eigen::MatrixXd laplacian_matrix = degree_matrix - similarity_matrix;

    // 计算归一化拉普拉斯矩阵
    Eigen::MatrixXd normalized_laplacian_matrix = degree_matrix.inverse().sqrt() * laplacian_matrix * degree_matrix.inverse().sqrt();

    return normalized_laplacian_matrix;
}


Eigen::MatrixXd vec2mat(const std::vector<std::vector<std::vector<char>>>& vec3d) {
    Eigen::MatrixXd matrix(vec3d.size(), vec3d[0].size());
    for (int i = 0; i < vec3d.size(); i++) {
        for (int j = 0; j < vec3d[0].size(); j++) {
            matrix(i, j) = vec3d[i][j];
        }
    }
    return matrix;
}


vec<double> Vec2vec(const Eigen::VectorXd& V) {
    vec<double> v;
    for (int i = 0; i < V.size(); i++) {
        v.push_back(V.row(i));
    }
    return v;
}


// 谱聚类函数
vector<vector<double>> spectral_clustering(const MatrixXd& data_matrix, int n_clusters, double sigma) {
    // 计算相似度矩阵
    Eigen::MatrixXd similarity_matrix = compute_similarity_matrix(data_matrix, sigma);

    // 计算归一化拉普拉斯矩阵
    Eigen::MatrixXd normalized_laplacian_matrix = compute_normalized_laplacian(similarity_matrix);

    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(normalized_laplacian_matrix);
    Eigen::VectorXd eigen_values = eigen_solver.eigenvalues();
    Eigen::MatrixXd eigen_vectors = eigen_solver.eigenvectors();

    // 获取特征向量的前n_clusters个分量
    Eigen::MatrixXd cluster_centers = eigen_vectors.block(0, 0, n_clusters, n_samples).transpose();

    // 构建邻接矩阵
    vector<vector<double>> adjacency_matrix(n_samples, vector<double>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            double similarity_ij = gaussian_kernel((cluster_centers.row(i) - cluster_centers.row(j)).norm(), 0, sigma);
            adjacency_matrix[i][j] = similarity_ij;
            adjacency_matrix[j][i] = similarity_ij;
        }
    }

    return adjacency_matrix;
}


int test() {
    // 示例数据矩阵
    vector<vector<vector<int>>> data {{{54, 57, 232}, {53, 47, 230}}, 
                                      {{53, 75, 199}, {36, 241, 24}},
                                      {{145, 42, 241}, {41, 192, 242}}};

    // 调用谱聚类
    vector<vector<double>> adjacency_matrix = spectral_clustering(vec2mat(data), 2, 1);

    // 输出邻接矩阵
    for (int i = 0; i < adjacency_matrix.size(); ++i) {
        for (int j = 0; j < adjacency_matrix[i].size(); ++j) {
            cout << adjacency_matrix[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
