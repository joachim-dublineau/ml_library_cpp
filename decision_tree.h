#ifndef DECISIONTREECLASSIFIER_H
#define DECISIONTREECLASSIFIER_H

#include <unordered_map>
#include <vector>
#include "condition.h"

using namespace std;

class DecisionTreeClassifier{

public:
    DecisionTreeClassifier(int);
    ~DecisionTreeClassifier();
    
    void build_tree(vector<vector<int>>, vector<int>);
    // marked as friend to access private element
    friend ostream& operator<<(ostream& strm, const DecisionTreeClassifier& tree);

private:
    vector<int> keys;
    int input_shape;
    unordered_map<int, Condition> conditions;
    unordered_map<int, vector<int>> nodes;
    double gini_index(vector<vector<int>>, vector<int>);
    double eval_split(vector<vector<int>>, vector<int>, Condition);
    Split do_split(vector<vector<int>>, vector<int>, Condition);
};

#endif