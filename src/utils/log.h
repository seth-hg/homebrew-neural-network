#pragma once

#include <iostream>
#include <ostream>
#include <string_view>

template <typename T> void _LOG(std::ostream &os, const T &arg) {
  os << arg;
}

template <typename T, typename... Args> void _LOG(std::ostream &os, const T &first, const Args &...rest) {
  os << first << " ";
  _LOG(os, rest...);
}

template <typename T, typename... Args> void LOG_ERROR(const T &first, const Args &...rest) {
  std::cerr << "\e[31m"
            << "[E] ";
  _LOG(std::cerr, first, rest...);
  std::cerr << "\e[0m" << std::endl;
}

template <typename T, typename... Args> void LOG_INFO(const T &first, const Args &...rest) {
  std::cout << "[I] ";
  _LOG(std::cout, first, rest...);
  std::cout << "\e[0m" << std::endl;
}

template <typename T, typename... Args> void LOG_DEBUG(const T &first, const Args &...rest) {
  std::cout << "\e[36m"
            << "[D] ";
  _LOG(std::cout, first, rest...);
  std::cout << "\e[0m" << std::endl;
}
