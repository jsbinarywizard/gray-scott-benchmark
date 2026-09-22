#pragma once 

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

struct timer{
    timer() : start(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }

    private:
        std::chrono::high_resolution_clock::time_point start;
};