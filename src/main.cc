#include "mlp.h"
#include "utils/log.h"
#include "utils/mnist.h"
#include "utils/timer.h"

#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

bool parseArgs(int argc, char **argv, std::string *model_path, std::string *data_path) {
  for (int i = 1; i < argc; ++i) {
    if (std::string("--model") == argv[i]) {
      i += 1;
      model_path->assign(argv[i]);
    } else if (std::string("--data") == argv[i]) {
      i += 1;
      data_path->assign(argv[i]);
    } else {
      return false;
    }
  }
  return true;
}

void printUsage(const char *cmd) {
  static const char *usage = " --model MODEL_PATH --data DATA_PATH";
  std::cout << cmd << usage << std::endl;
}

extern size_t argmax(size_t n, const float *values);

int main(int argc, char **argv) {
  std::string model_path("../scripts/mnist_mlp.bin"), data_path("../data/MNIST/raw");
  if (!parseArgs(argc, argv, &model_path, &data_path) || model_path.empty() || data_path.empty()) {
    printUsage(argv[0]);
    return 0;
  }

  MNIST<float> dataset;
  if (!dataset.load(data_path)) {
    LOG_ERROR("failed loading MNIST dataset");
    return 0;
  }

  MLP mlp({dataset.n_features(), 128, 10});
  if (!mlp.load(model_path)) {
    LOG_ERROR("failed loading model");
    return 0;
  }

  Timer timer;
  std::vector<float> out(dataset.n_test() * 10);
  timer.start();
  mlp.forwardBatch(dataset.n_test(), dataset.test_data(), &out);
  timer.stop();

  size_t correct = 0;
  std::vector<int> classes(dataset.n_test());
#pragma omp parallel for
  for (size_t i = 0; i < dataset.n_test(); ++i) {
    size_t idx = argmax(10, out.data() + 10 * i);
    classes[i] = idx;
  }

  for (size_t i = 0; i < classes.size(); ++i) {
    size_t target = (size_t)dataset.test_label()[i];
    if (classes[i] == target) {
      correct += 1;
    }
  }

  LOG_INFO("Elapsed:", timer.elapsed_ms(), "ms, Accuracy:", correct / (float)dataset.n_test());
  return 0;
}
