#pragma once
#include <chrono>
#include <cstdio>
#include <iostream>

//#define debuging
#ifdef debuging
#define myDebug(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__);
#else
#define myDebug(fmt, ...) ;
#endif

extern std::chrono::time_point<std::chrono::steady_clock> start, stop;

#define recordtime
#ifdef recordtime
#define recordStart() start = std::chrono::high_resolution_clock::now();
#define recordStop(x) stop = std::chrono::high_resolution_clock::now(); fprintf(stderr, "%s %dms\n", (x), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
#else
#define recordStart() ;
#define recordStop(x) ;
#endif

// Function for exchanging two indices
inline void swap(int& x, int& y) { int z = x; x = y, y = z; };

// Minimum of two double variables
inline double min(double x, double y) { return x < y ? x : y; };
