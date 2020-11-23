#ifndef CONDITION_H
#define CONDITION_H

#include <vector>
#include <iostream>

using namespace std;

template<typename T>
struct Condition{
    T value;
    int label;
    int type; // 0 =, 1 <, 2 >, 3 <=, 4 >=
};

template<typename T>
struct Split{
    vector<vector<T>> mat1;
    vector<int> lab1;
    vector<vector<T>> mat2;
    vector<int> lab2;
};

#endif