#include <vector>
#include <iostream>
#include <cmath>

using namespace std;
double dot(const vector<double>& vec1, const vector<double>& vec2) {
    double product = 0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        product += vec1[i] * vec2[i];
    }
    return product;
}
double norm(const vector<double>& vec) {
    double sum = 0;
    for (double element : vec) {
        sum += element * element;
    }
    return sqrt(sum);
}

pair<vector<vector<double>>, vector<vector<double>>> classicalGramSchmidt(const vector<vector<double>>& A) {
    size_t M = A.size();    // Number of rows
    size_t N = A[0].size(); // Number of columns

    vector<vector<double>> Q = A;
    vector<vector<double>> R(N, vector<double>(N, 0));

    for (size_t k = 0; k < N; ++k) {
        // Creating a vector for the k-th column of A/Q
        vector<double> q_k(M);
        for (size_t i = 0; i < M; ++i) {
            q_k[i] = A[i][k];
        }

        for (size_t j = 0; j < k; ++j) {
            // Creating a vector for the j-th column of Q
            vector<double> q_j(M);
            for (size_t i = 0; i < M; ++i) {
                q_j[i] = Q[i][j];
            }

            R[j][k] = dot(q_j, q_k);
            for (size_t i = 0; i < M; ++i) {
                q_k[i] -= R[j][k] * Q[i][j];
            }
        }

        // Calculate the norm of the vector q_k
        R[k][k] = norm(q_k);
        if (R[k][k] != 0) {
            for (size_t i = 0; i < M; ++i) {
                Q[i][k] = q_k[i] / R[k][k];
            }
        }
    }

    return {Q, R};
}


pair<vector<vector<double>>, vector<vector<double>>> modifiedGramSchmidt(const vector<vector<double>>& A) {
    size_t M = A.size();    // Number of rows
    size_t N = A[0].size(); // Number of columns

    vector<vector<double>> Q = A; // Start with Q = A
    vector<vector<double>> R(N, vector<double>(N, 0));

    for (size_t k = 0; k < N; ++k) {
        // Extract the k-th column of Q into a vector
        vector<double> q_k(M);
        for (size_t i = 0; i < M; ++i) {
            q_k[i] = Q[i][k];
        }

        // Compute the norm of the k-th column of Q
        R[k][k] = norm(q_k);

        // Normalize the k-th column of Q
        if (R[k][k] != 0) {
            for (size_t i = 0; i < M; ++i) {
                Q[i][k] /= R[k][k];
            }
        }

        for (size_t j = k + 1; j < N; ++j) {
            // Compute dot product of k-th and j-th columns
            R[k][j] = 0;
            for (size_t i = 0; i < M; ++i) {
                R[k][j] += Q[i][k] * Q[i][j];
            }

            // Update the j-th column of Q
            for (size_t i = 0; i < M; ++i) {
                Q[i][j] -= R[k][j] * Q[i][k];
            }
        }
    }

    return {Q, R};
}






void printVector(const vector<double>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

// Function to print a matrix
void printMatrix(const vector<vector<double>>& m) {
    for (const auto& row : m) {
        printVector(row);
    }
}
// Function to perform matrix-vector multiplication
vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<double> result(rows, 0);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// Function to perform backward substitution
vector<double> backwardSubstitution(const vector<vector<double>>& R, const vector<double>& y) {
    size_t N = R.size();
    vector<double> x(N, 0);

    for (int i = N - 1; i >= 0; --i) {
        x[i] = y[i];
        for (size_t j = i + 1; j < N; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        if (R[i][i] != 0) {  // Check for non-zero diagonal element
            x[i] /= R[i][i];
        }
    }
    return  x;
}

// Function to transpose a matrix
vector<vector<double>> transposeMatrix(const vector<vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    vector<vector<double>> trans(cols, vector<double>(rows, 0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            trans[j][i] = matrix[i][j];
        }
    }
    return trans;
}



int main() {
    double epsilon1 = 1e-10;
    vector<vector<double>> A1 = { {1, 1, 1}, {1, 1 + epsilon1, 1}, {1, 1, 1 + epsilon1} };
    cout << "Test Case 1:" << endl;
    auto result1 = classicalGramSchmidt(A1);
    cout << "Classical Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result1.first);
    cout << "Classical Gram-Schmidt R Matrix:" << endl;
    printMatrix(result1.second);

    auto result1m = modifiedGramSchmidt(A1);
    cout << "Modified Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result1m.first);
    cout << "Modified Gram-Schmidt R Matrix:" << endl;
    printMatrix(result1m.second);
    cout << endl;

    double epsilon2 = 1e-6;
    size_t size2 = 10; 
    vector<vector<double>> A2(size2, vector<double>(size2, epsilon2));
    for (size_t i = 0; i < size2; ++i) {
        A2[i][i] = 1;
    }
    cout << "Test Case 2:" << endl;
    auto result2 = classicalGramSchmidt(A2);
    cout << "Classical Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result2.first);
    cout << "Classical Gram-Schmidt R Matrix:" << endl;
    printMatrix(result2.second);

    auto result2m = modifiedGramSchmidt(A2);
    cout << "Modified Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result2m.first);
    cout << "Modified Gram-Schmidt R Matrix:" << endl;
    printMatrix(result2m.second);
    cout << endl;


    vector<vector<double>> A3 = { {1, 0.1, 0.01}, {0.1, 0.01, 0.001}, {0.01, 0.001, 0.0001} };
    cout << "Test Case 3:" << endl;
    auto result3 = classicalGramSchmidt(A3);
    cout << "Classical Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result3.first);
    cout << "Classical Gram-Schmidt R Matrix:" << endl;
    printMatrix(result3.second);

    auto result3m = modifiedGramSchmidt(A3);
    cout << "Modified Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result3m.first);
    cout << "Modified Gram-Schmidt R Matrix:" << endl;
    printMatrix(result3m.second);
    cout << endl;

 
    size_t size4 = 5; 
    vector<vector<double>> A4(size4, vector<double>(size4, 1));
    A4[2][3] = 1e-10; 
    cout << "Test Case 4:" << endl;
    auto result4 = classicalGramSchmidt(A4);
    cout << "Classical Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result4.first);
    cout << "Classical Gram-Schmidt R Matrix:" << endl;
    printMatrix(result4.second);

    auto result4m = modifiedGramSchmidt(A4);
    cout << "Modified Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result4m.first);
    cout << "Modified Gram-Schmidt R Matrix:" << endl;
    printMatrix(result4m.second);
    cout << endl;

  
    size_t size5 = 20; 
    vector<vector<double>> A5(size5, vector<double>(size5, 0));
    for (size_t i = 0; i < size5; ++i) {
        for (size_t j = 0; j < size5; ++j) {
            A5[i][j] = 1 + static_cast<double>(rand()) / RAND_MAX * 0.01; // Slight variations
        }
    }
    cout << "Test Case 5:" << endl;
    auto result5 = classicalGramSchmidt(A5);
    cout << "Classical Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result5.first);
    cout << "Classical Gram-Schmidt R Matrix:" << endl;
    printMatrix(result5.second);

    auto result5m = modifiedGramSchmidt(A5);
    cout << "Modified Gram-Schmidt Q Matrix:" << endl;
    printMatrix(result5m.first);
    cout << "Modified Gram-Schmidt R Matrix:" << endl;
    printMatrix(result5m.second);
    cout << endl;

    return 0;
}
