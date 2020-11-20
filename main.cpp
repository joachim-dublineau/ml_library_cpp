#include "decision_tree.h"
#include "decision_tree.cpp" // important to avoid linking issues
#include "condition.h"
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

    vector<vector<int>> input_matrix;
    for (int i = 0; i < num_examples; i++){
        vector<int> row;
        for (int j = 0; j < input_shape; j++){
            row.push_back(matrix[i][j]);
        }
        input_matrix.push_back(row);
    }

    vector<int> labels {1,0,0,1,1,0,0,0};
    
    //Model Creation

    DecisionTreeClassifier<int> tree(input_shape);

    tree.build_tree(input_matrix, labels);
    cout << "Tree created" << endl;
    cout << tree << endl;
}

