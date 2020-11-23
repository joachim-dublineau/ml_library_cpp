#ifndef UTILS_H
#define UTILS_H

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

struct TreeNotBuiltException : public exception {
   const char * what () const throw () {
      return "The tree must be built before any prediction.";
   }
};

int max_(int a, int b);
#endif