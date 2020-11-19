#include "decision_tree.h"
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

    DecisionTreeClassifier tree = DecisionTreeClassifier(input_shape);

    // Test of Gini Index:
    // vector<int> group1 {0,0,1,1,1,1};
    // vector<int> group2 {1,2,2,2};
    // vector<int> classes {0,1,2};
    // vector<vector<int>> groups {group1, group2};
    // cout << tree.gini_index_2(groups, classes) << endl;

    // Test of eval_split and do_split
    // Condition condition;
    // condition.label = 0;
    // condition.value = 0;
    // condition.type = 0;
    // cout << tree.eval_split(input_matrix, labels, condition) << endl;
    // Split split = tree.do_split(input_matrix, labels, condition);
    // for (vector<int> elem: split.mat1){
    //     cout << elem.at(0) << elem.at(1) << elem.at(2) << elem.at(3) << endl;
    // }
    // for (int elem: split.lab1){
    //     cout << elem << endl;
    // }
    // for (vector<int> elem: split.mat2){
    //     cout << elem.at(0) << elem.at(1) << elem.at(2) << elem.at(3) << endl;
    // }
    
    tree.build_tree(input_matrix, labels);
    cout << "Tree created" << endl;
    cout << tree << endl;
}

