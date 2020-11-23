#include <cmath>
#include "decision_tree_classifier.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h> 
#include <type_traits>

using namespace std;


template<typename T>
DecisionTreeClassifier<T>::DecisionTreeClassifier(int input_shape_, int num_values_to_examine){
    input_shape = input_shape_;
    number_values_to_test = num_values_to_examine;
};

template<typename T>
DecisionTreeClassifier<T>::~DecisionTreeClassifier(){
};

template<typename T>
void DecisionTreeClassifier<T>::build_tree(vector<vector<T>> matrix, vector<int> labels){
    // This function builds the tree.

    //stop when all elements are of the same class or if matrix is empty.
    bool continue_build = false;
    if (labels.size() > 0){
        int class_ = labels.at(0);
        for (int elem : labels){
            if (elem != class_){
                continue_build = true;
                break;
            }
        } 
    }

    if (continue_build){
        int num_rows = labels.size();
        int num_columns = matrix.at(0).size();
        vector<double> ginis;
        unordered_map<int, Condition<T>> conditions_loc;
        int index = 0;

        // if they are some elements lefts
        for (int i = 0; i < num_columns; i++){
            vector<T> column;
            for (int j = 0; j < num_rows; j ++){
                column.push_back(matrix.at(j).at(i));
            }

            T value_min = *min_element(column.begin(), column.end());
            T value_max = *max_element(column.begin(), column.end());

            // create range of values to examine
            T step;
            
            if (typeid(T).name() == typeid(int).name()){
                step = 1;
            }
            else{
                step = (value_max - value_min)/(1.0 * number_values_to_test);
            }
            
            // if value_max - value_min -1 >= 0
            for (T k = value_min + step; k <= value_max; k += step){
                Condition<T> condition;
                condition.value = k;
                condition.type = 1;
                condition.label = i;
                double gini_value = this->eval_split(matrix, labels, condition);
                ginis.push_back(gini_value);
                conditions_loc[index] = condition;
                index += 1;
            }

            // else: they are equal which means that this is not a good discriminator
            // --> changing column 
        }

        int best_gini_index = min_element(ginis.begin(), ginis.end()) - ginis.begin();
        Condition<T> best_condition = conditions_loc[best_gini_index];

        // find 2 indexes that aren't already attributed
        int  max_key;
        if (keys.size()>0){
            max_key = *max_element(keys.begin(), keys.end());
        }
        else{
            max_key = 0;
            keys.push_back(0);
        }
        keys.push_back(max_key + 1);
        keys.push_back(max_key + 2);
        
        // updating conditions
        conditions[max_key] = best_condition;    

        // updating nodes
        vector<int> connections;
        connections.push_back(max_key+1);
        connections.push_back(max_key+2);
        nodes[max_key] = connections;
        
        // do_split and execute build tree on left node and right node
        Split<T> split = this->do_split(matrix, labels, best_condition);
        this->build_tree(split.mat1, split.lab1);
        this->build_tree(split.mat2, split.lab2);
    }
    // else{
    //     //do nothing
    // }
};

template<typename T>
ostream& operator<<(std::ostream &strm, const DecisionTreeClassifier<T> &tree) {
    // Overloading operator << 
    unordered_map<int, vector<int>>::iterator it;
    unordered_map<int, vector<int>> nodes = tree.nodes;
    vector<int> keys = tree.keys;
    for (int index: keys){
        // check if there is condition on this node and display it
        if (tree.conditions.find(index) != tree.conditions.end()){
            strm << "Node " << index << ":" << endl;
            strm << "Condition: ";
            Condition<T> condition = tree.conditions.find(index)->second;
            if (condition.type == 0){
                strm << "column" << condition.label << " = " << condition.value << endl;
            }
            if (condition.type == 1){
                strm << "column" << condition.label << " < " << condition.value << endl;
            }
            strm << "Connected to: "; 
            for (it = nodes.begin(); it != nodes.end(); it++){
                if (it->first == index){
                    for (int elem : it->second){
                        strm << elem << ", ";
                    }
                }
            }
            cout << endl;
        }
        else {
            strm << "Leaf " << index << endl;
        }
    }

    return strm;
};

template<typename T>
double DecisionTreeClassifier<T>::gini_index(vector<vector<int>> groups, vector<int> classes){
    // This function computes the gini index, given a set of groups and their
    // possible classes.
    // Test: 
    // vector<int> group1 {0,0,1,1,1,1};
    // vector<int> group2 {1,2,2,2};
    // vector<int> classes {0,1,2};
    // vector<vector<int>> groups {group1, group2};
    // cout << tree.gini_index(groups, classes) << endl;
    // Expected output: 0.41666
    int size_total = 0;
    for (vector<int> group: groups){
        size_total += group.size();
    }
    double sum = 0.;
    for (vector<int> group: groups){
        double score = 0.;
        for (int class_: classes){
            int num_elements = 0;
            // count the number of elements from the group that belong to the class
            for (int value_class: group){
                if (value_class == class_){ 
                    num_elements += 1;
                }
            }
            score += num_elements*num_elements/(1.0 * group.size() * group.size());
        }
        sum += (1 - score) * group.size()/(1.0*size_total);
    }
    return sum;
};

template<typename T>
double DecisionTreeClassifier<T>::eval_split(vector<vector<T>> matrix, vector<int> labels, Condition<T> condition){
    // This function will use the condition to split the data from matrix and labels
    // and compute the gini_index.
    // Test:
    // Condition condition;
    // condition.label = 0;
    // condition.value = 0;
    // condition.type = 0;
    // cout << tree.eval_split(matrix, labels, condition) << endl;

    int column_index = condition.label;
    T criteria_value = condition.value;
    int criteria = condition.type;
    vector<int> group1;
    vector<int> group2;
    vector<vector<int>> groups;
    groups.push_back(group1);
    groups.push_back(group2);

    vector<int> classes;
    for (int i = 0; i < labels.size(); i++){
        vector<T> row = matrix.at(i);
        T value = row.at(column_index);
        int class_ = labels.at(i);

        // if a new class is discovered, it is added
        if (!(count(classes.begin(), classes.end(), class_))){
            classes.push_back(class_);
        }

        //if it respects the condition group1, else group 0
        if (criteria == 0){
            if (value == criteria_value){
                groups.at(1).push_back(class_);
            }
            else {
                groups.at(0).push_back(class_);
            }
        }

        if (criteria == 1){
            if (value < criteria_value){
                groups.at(1).push_back(class_);
            }
            else {
                groups.at(0).push_back(class_);
            }
        }
    }
    double gini = this->gini_index(groups, classes);
    return gini;
};

template<typename T>
Split<T> DecisionTreeClassifier<T>::do_split(vector<vector<T>> matrix, vector<int> labels, Condition<T> condition){
    // This function will return the Split object given an input matrix
    // a label vector and a condition.
    int column_index = condition.label;
    T criteria_value = condition.value;
    int criteria = condition.type;
    vector<vector<T>> mat1;
    vector<vector<T>> mat2;
    vector<int> lab1;
    vector<int> lab2;

    for (int i = 0; i < labels.size(); i++){
        vector<T> row = matrix.at(i);
        T value = row.at(column_index);
        int class_ = labels.at(i);

        //if it respects the condition group1, else group 0
        if (criteria == 0){
            if (value == criteria_value){
                mat2.push_back(row);
                lab2.push_back(class_);
            }
            else {
                mat1.push_back(row);
                lab1.push_back(class_);
            }
        }

        if (criteria == 1){
            if (value < criteria_value){
                mat2.push_back(row);
                lab2.push_back(class_);
            }
            else {
                mat1.push_back(row);
                lab1.push_back(class_);
            }
        }
    }
    Split<T> split;
    split.mat1 = mat1;
    split.mat2 = mat2;
    split.lab1 = lab1;
    split.lab2 = lab2;
    return split;
}