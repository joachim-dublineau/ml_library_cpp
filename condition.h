#ifndef CONDITION_H
#define CONDITION_H

#include <vector>
#include <iostream>

using namespace std;

struct Condition{
    int value;
    int label;
    int type; // 0 =, 1 <, 2 >, 3 <=, 4 >=
};

struct Split{
    vector<vector<int>> mat1;
    vector<int> lab1;
    vector<vector<int>> mat2;
    vector<int> lab2;
};

#endif