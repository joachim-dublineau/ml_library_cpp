#include "utils.h"
#include "decision_tree_classifier.h" 
#include "decision_tree_classifier.cpp" // important to avoid linking issues
#include <iostream>

using namespace std;

int main(){
    // Data Creation
    int input_shape = 4;
    int num_examples = 8;
    

    int matrix[num_examples][input_shape] = {{1,1,0,1},
    {2,1,1,0},
    {0,2,1,0},
    {0,1,1,1},
    {1,0,2,1},
    {2,2,2,1},
    {1,2,2,1},
    {0,1,1,0}};

    vector<int> labels {1,0,0,1,1,0,0,0};

    vector<vector<int>> input_matrix;
    for (int i = 0; i < num_examples; i++){
        vector<int> row;
        for (int j = 0; j < input_shape; j++){
            row.push_back(matrix[i][j]);
        }
        input_matrix.push_back(row);
    }

    float matrix_2[num_examples][input_shape] = {{1.0,1.0,0.,1.},
    {2.,1.,1.,0.},
    {0.,2.,1.,0.},
    {0.,1.,1.,1.},
    {1.,0.,2.,1.},
    {2.,2.,2.,1.},
    {1.,2.,2.,1.},
    {0.,1.,1.,0.}};

    vector<vector<float>> input_matrix_2;
    for (int i = 0; i < num_examples; i++){
        vector<float> row;
        for (int j = 0; j < input_shape; j++){
            row.push_back(matrix_2[i][j]);
        }
        input_matrix_2.push_back(row);
    }

    int matrix_test[4][input_shape] = {{1,1,0,1},
    {2,1,1,0},
    {0,1,1,0},
    {1,2,1,1}};

    vector<int> labels_test {1,0,0,1};

    vector<vector<int>> test_matrix;
    for (int i = 0; i < 4; i++){
        vector<int> row;
        for (int j = 0; j < input_shape; j++){
            row.push_back(matrix_test[i][j]);
        }
        test_matrix.push_back(row);
    }
    
    //Model Creation

    DecisionTreeClassifier<int> tree(input_shape);
    DecisionTreeClassifier<float> tree_2(input_shape);

    tree.build_tree(input_matrix, labels);
    cout << "Tree created" << endl;
    cout << tree << endl;

    tree_2.build_tree(input_matrix_2, labels);
    cout << "Tree created" << endl;
    cout << tree_2 << endl;

    cout << "Depth of the tree: " << tree.compute_depth() << endl;

    vector<int> predicted_labels = tree.predict(test_matrix);
    for (int elem : predicted_labels){
        cout << elem << endl;
    }

    // TODO: ADD proportion to leafs as proba. NLL error
}

