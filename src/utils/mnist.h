#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// File format explained here:
//   https://rstudio-pubs-static.s3.amazonaws.com/465274_cd4d339c79aa4d94a0140b63aaf065e0.html
template <typename T> class MNIST {
public:
  MNIST() : n_features_(784) {}

  bool load(const std::string &data_path) {
    if (!loadImageFile(data_path + "/train-images-idx3-ubyte", &train_data_)) {
      return false;
    }
    if (!loadLabelFile(data_path + "/train-labels-idx1-ubyte", &train_labels_)) {
      return false;
    }
    if (!loadImageFile(data_path + "/t10k-images-idx3-ubyte", &test_data_)) {
      return false;
    }
    if (!loadLabelFile(data_path + "/t10k-labels-idx1-ubyte", &test_labels_)) {
      return false;
    }
    return true;
  }

  const std::vector<T> &train_data() const { return train_data_; }

  const std::vector<T> &test_data() const { return test_data_; }
  const std::vector<uint8_t> &test_label() const { return test_labels_; }

  size_t n_features() const { return n_features_; }
  size_t n_train() const { return train_data_.size() / n_features_; }
  size_t n_test() const { return test_data_.size() / n_features_; }

private:
  bool loadImageFile(const std::string &path, std::vector<T> *data) {
    std::ifstream ifs(path, std::ios::binary);
    uint32_t magic, samples, rows, cols;
    // 4 byte for magic
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = ntohl(magic);
    if (magic != IMAGE_FILE_MAGIC) {
      return false;
    }
    // 4 byte for num of images
    ifs.read(reinterpret_cast<char *>(&samples), sizeof(samples));
    samples = ntohl(samples);
    // 4 byte for height of each image, should be 28
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    rows = ntohl(rows);
    // 4 byte for width of each image, should be 28
    ifs.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    cols = ntohl(cols);
    n_features_ = rows * cols;
    // images, (rows * cols) bytes each
    std::vector<uint8_t> raw(n_features_ * samples);
    ifs.read(reinterpret_cast<char *>(raw.data()), raw.size());
    data->resize(n_features_ * samples);
    float x = 0.0;
    for (size_t i = 0; i < data->size(); ++i) {
      (*data)[i] = static_cast<T>(raw[i]) / 255;
      x += (*data)[i];
    }
    return true;
  }

  bool loadLabelFile(const std::string &path, std::vector<uint8_t> *data) {
    std::ifstream ifs(path, std::ios::binary);
    uint32_t magic, samples;
    // 4 byte for magic
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    magic = ntohl(magic);
    if (magic != LABEL_FILE_MAGIC) {
      return false;
    }
    // 4 byte for num of labels
    ifs.read(reinterpret_cast<char *>(&samples), sizeof(samples));
    samples = ntohl(samples);
    // labels, 1 byte each
    data->resize(samples);
    ifs.read(reinterpret_cast<char *>(data->data()), data->size());
    return true;
  }

private:
  size_t n_features_;
  std::vector<T> train_data_;
  std::vector<uint8_t> train_labels_;
  std::vector<T> test_data_;
  std::vector<uint8_t> test_labels_;

  static constexpr uint32_t IMAGE_FILE_MAGIC = 2051;
  static constexpr uint32_t LABEL_FILE_MAGIC = 2049;
};