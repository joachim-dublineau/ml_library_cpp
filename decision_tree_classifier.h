#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <unordered_map>
#include <iostream>
#include "utils.h"

using namespace std;

template<typename T>
class DecisionTreeClassifier{

public:
    DecisionTreeClassifier(int, int num_values_to_examine = 100);
    ~DecisionTreeClassifier();
    
    void build_tree(vector<vector<T>>, vector<int>);

    // marked as friend to access private element
    template<typename L>
    friend ostream& operator<<(ostream& strm, const DecisionTreeClassifier<L>& tree);

private:
    vector<int> keys;
    int input_shape;
    int number_values_to_test;
    unordered_map<int, Condition<T>> conditions;
    unordered_map<int, vector<int>> nodes;
    double gini_index(vector<vector<int>>, vector<int>);
    double eval_split(vector<vector<T>>, vector<int>, Condition<T>);
    Split<T> do_split(vector<vector<T>>, vector<int>, Condition<T>);
};

#endif