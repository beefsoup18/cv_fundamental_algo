#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "headdefs.h"

using namespace std;

// Function to calculate the Normalized Cut of a graph
// Input: adjacency matrix of the graph, number of clusters k
// Output: vector containing the cluster assignments of each node
vector<int> normalized_cut(vector<vector<double>> adjacency_matrix, int k) {
    
    // Get the number of nodes in the graph
    int n = adjacency_matrix.size();
    
    // Initialize the cluster assignments randomly
    vector<int> cluster_assignments(n);
    for (int i = 0; i < n; ++i) {
        cluster_assignments[i] = rand() % k;
    }
    
    // // Calculate the degree matrix
    // vector<double> degree(n);
    // for (int i = 0; i < n; ++i) {
    //     double sum = 0.0;
    //     for (int j = 0; j < n; ++j) {
    //         sum += adjacency_matrix[i][j];
    //     }
    //     degree[i] = sum;
    // }

    Eigen::VectorXd degree = calculate_degree_matrix(vec2mat(adjacency_matrix));
    
    // // Calculate the Laplacian matrix
    // vector<vector<double>> laplacian(n, vector<double>(n));
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         if (i == j) {
    //             laplacian[i][j] = degree[i] - adjacency_matrix[i][j];
    //         } else {
    //             laplacian[i][j] = -adjacency_matrix[i][j];
    //         }
    //     }
    // }

    Eigen::MatrixXd laplacian = calculate_laplacian_matrix(adjacency_matrix, degree);
    
    // // Calculate the eigenvectors and eigenvalues of the Laplacian matrix
    // vector<double> eigenvalues(n);
    // vector<vector<double>> eigenvectors(n, vector<double>(n));
    // for (int i = 0; i < n; ++i) {
    //     vector<double> x(n, 1.0);
    //     double norm = 0.0;
    //     for (int j = 0; j < n; ++j) {
    //         norm += x[j] * x[j];
    //     }
    //     norm = sqrt(norm);
    //     for (int j = 0; j < n; ++j) {
    //         x[j] /= norm;
    //     }
    //     for (int j = 0; j < 100; ++j) {
    //         vector<double> y(n);
    //         for (int k = 0; k < n; ++k) {
    //             for (int l = 0; l < n; ++l) {
    //                 y[k] += laplacian[k][l] * x[l];
    //             }
    //         }
    //         double lambda = 0.0;
    //         for (int k = 0; k < n; ++k) {
    //             lambda += x[k] * y[k];
    //         }
    //         eigenvalues[i] = lambda;
    //         double norm = 0.0;
    //         for (int k = 0; k < n; ++k) {
    //             x[k] = y[k] - lambda * x[k];
    //             norm += x[k] * x[k];
    //         }
    //         norm = sqrt(norm);
    //         for (int k = 0; k < n; ++k) {
    //             x[k] /= norm;
    //         }
    //     }
    //     for (int j = 0; j < n; ++j) {
    //         eigenvectors[j][i] = x[j];
    //     }
    // }
    
    // // Sort the eigenvectors by their corresponding eigenvalues
    // vector<int> indices(n);
    // for (int i = 0; i < n; ++i) {
    //     indices[i] = i;
    // }
    // sort(indices.begin(), indices.end(), [&](int i, int j) { return eigenvalues[i] < eigenvalues[j]; });
    // reverse(indices.begin(), indices.end());

    // eigenvectors
    vector<int> indices = Vec2vec(solve_specialized_eigenproblem(laplacian));
    
    // Assign nodes to clusters based on the sign of the eigenvectors
    vector<int> cluster_sizes(k);
    for (int i = 0; i < n; ++i) {
        int node_index = indices[i];
        int cluster_index = -1;
        double max_value = -1.0;
        for (int j = 0; j < k; ++j) {
            double value = fabs(eigenvectors[node_index][j]);
            if (value > max_value) {
                max_value = value;
                cluster_index = j;
            }
        }
        cluster_sizes[cluster_index]++;
        cluster_assignments[node_index] = cluster_index;
    }
    
    // If any clusters are empty, reassign nodes
    bool empty_cluster = true;
    while (empty_cluster) {
        empty_cluster = false;
        for (int i = 0; i < k; ++i) {
            if (cluster_sizes[i] == 0) {
                empty_cluster = true;
                int node_index = -1;
                double max_degree = -1.0;
                for (int j = 0; j < n; ++j) {
                    if (cluster_assignments[j] != i) {
                        continue;
                    }
                    if (degree[j] > max_degree) {
                        max_degree = degree[j];
                        node_index = j;
                    }
                }
                cluster_assignments[node_index] = rand() % k;
                cluster_sizes[i]--;
                cluster_sizes[cluster_assignments[node_index]]++;
            }
        }
    }
    
    // Return the cluster assignments
    return cluster_assignments;
}


int test() {
    // Define the adjacency matrix
    vector<vector<double>> adjacency_matrix = {{0, 1, 1, 0},
                                                          {1, 0, 1, 0},
                                                          {1, 1, 0, 1},
                                                          {0, 0, 1, 0}};
    
    // Perform normalized cut clustering with k=2
    int k = 2;
    vector<int> cluster_assignments = normalized_cut(adjacency_matrix, k);
    
    // Print the results
    for (int i = 0; i < cluster_assignments.size(); ++i) {
        cout << "Node " << i << " belongs to cluster " << cluster_assignments[i] << endl;
    }
    
    return 0;
}
