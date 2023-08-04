#pragma once

#include <chrono>
#include <iostream>

class Timer {
public:
  void start() { start_ = std::chrono::steady_clock::now(); }

  void stop() { elapsed_ = std::chrono::steady_clock::now() - start_; }

  double elapsed_ms() const { return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_).count(); }

  double elapsed() const { return elapsed_.count(); }

private:
  std::chrono::steady_clock::time_point start_;
  std::chrono::duration<double> elapsed_;
};