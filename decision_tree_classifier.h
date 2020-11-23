#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <unordered_map>
#include <iostream>
#include "utils.h"

using namespace std;

template<typename T>
class DecisionTreeClassifier{

public:
    DecisionTreeClassifier(int, int max_depth = 5, int num_values_to_examine = 100);
    ~DecisionTreeClassifier();
    
    void build_tree(vector<vector<T>>, vector<int>, int = 0);
    vector<int> predict(vector<vector<T>>);
    double error(vector<int>, vector<int>);
    int compute_depth(int = 0, int = 0);

    // marked as friend to access private element
    template<typename L>
    friend ostream& operator<<(ostream& strm, const DecisionTreeClassifier<L>& tree);

private:
    vector<int> keys;
    int input_shape;
    int max_depth;
    int number_values_to_test;
    unordered_map<int, Condition<T>> conditions;
    unordered_map<int, vector<int>> nodes;
    unordered_map<int, int> leafs;
    double gini_index(vector<vector<int>>, vector<int>);
    double eval_split(vector<vector<T>>, vector<int>, Condition<T>);
    Split<T> do_split(vector<vector<T>>, vector<int>, Condition<T>);
    
};

#endif