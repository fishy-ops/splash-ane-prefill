// Neural Engine time for one projection run the way Splash runs the FFN's
// share (runtime/ops/AneFfn.cpp): int8 activations in 2560-channel segments,
// int8 weights, a matmul per segment summed in fp16 and scaled per output
// channel. Times evaluations at several chunk rows, so a projection's ANE
// time can be set against the GPU's on the same chunk.
//
//   V=engine; xcrun clang++ -std=c++20 -O2 -fobjc-arc -mmacosx-version-min=26.4 \
//     -I$V/runtime -I$V/dev bench/ane/ane_projection.mm $V/build/engine/libsplash.a \
//     -framework Foundation -framework Metal -framework IOKit -framework IOSurface -o build/ane_projection
//   build/ane_projection METALLIB N K [ROWS...]

#include "AwakeClock.hpp"
#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace splash;
using Element = ane::Surface::Element;

namespace {

static const uint32_t kSegment = std::getenv("SEGMENT") ? static_cast<uint32_t>(std::atoi(std::getenv("SEGMENT"))) : 2560;
static const uint32_t kMaximumRows = std::getenv("MAXROWS") ? static_cast<uint32_t>(std::atoi(std::getenv("MAXROWS"))) : 2048;

std::string dims(uint64_t rows, uint64_t width) {
  return "[1, 1, " + std::to_string(rows) + ", " + std::to_string(width) + "]";
}
std::string tensor(const char *type, uint64_t rows, uint64_t width) {
  return std::string("tensor<") + type + ", " + dims(rows, width) + ">";
}

struct Bench final {
  uint32_t n, k, segments;
  std::vector<uint32_t> widths;  // each segment's input channels, the last possibly short
  std::vector<ane::Surface> x, w;
  ane::Surface scale, out, outT;
};
const bool fp16Weights = std::getenv("VARIANT") && std::string(std::getenv("VARIANT")) == "fp16w";
const bool transposed = std::getenv("VARIANT") && std::string(std::getenv("VARIANT")).find("T") != std::string::npos;
const bool fp16Inputs = std::getenv("VARIANT") && std::string(std::getenv("VARIANT")).find("x16") != std::string::npos;

std::string function(const Bench &b, uint32_t rows) {
  std::string params, body;
  const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
  for (uint32_t s = 0; s < b.segments; ++s) {
    const std::string i = std::to_string(s);
    const uint32_t width = b.widths[s];
    params += (params.empty() ? "" : ", ") + b.x[s].bufferType(width, rows) + " x" + i;
    params += ", " + b.w[s].bufferType(b.n, width) + " w" + i;
    line(tensor(fp16Inputs ? "fp16" : "int8", width, rows) + " x" + i + "_t = tensor_buffer_to_tensor<ios17>(input = x" + i + ")");
    line(tensor(fp16Weights ? "fp16" : "int8", b.n, width) + " w" + i + "_t = tensor_buffer_to_tensor<ios17>(input = w" + i + ")");
    if (fp16Inputs) line(tensor("fp16", width, rows) + " x" + i + "_d = identity(x = x" + i + "_t)");
    else line(tensor("fp16", width, rows) + " x" + i + "_d = dequantize(input = x" + i + "_t, scale = fp16(0x1p-7))");
    if (fp16Weights) line(tensor("fp16", b.n, width) + " w" + i + "_d = identity(x = w" + i + "_t)");
    else line(tensor("fp16", b.n, width) + " w" + i + "_d = dequantize(input = w" + i + "_t, scale = fp16(0x1p-7))");
    if (transposed)
      line(tensor("fp16", rows, b.n) + " m" + i +
           " = matmul(transpose_x = bool(true), transpose_y = bool(true), x = x" + i + "_d, y = w" + i + "_d)");
    else
      line(tensor("fp16", b.n, rows) + " m" + i +
           " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + i + "_d, y = x" + i + "_d)");
  }
  std::string total = "m0";
  for (uint32_t s = 1; s < b.segments; ++s) {
    const std::string next = "sum" + std::to_string(s);
    line((transposed ? tensor("fp16", rows, b.n) : tensor("fp16", b.n, rows)) + " " + next + " = add(x = " + total + ", y = m" + std::to_string(s) + ")");
    total = next;
  }
  if (transposed) {
    line(tensor("fp16", rows, b.n) + " ys = mul(x = " + total + ", y = fp16(0.5))");
    line(b.outT.bufferType(rows, b.n) +
         " y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
         "strides = tensor<int64, [4]>(" + b.outT.strides(rows) + "))");
    return "    func p" + std::to_string(rows) + "<ios18>(" + params + ") {\n" + body + "    } -> (y);\n";
  }
  params += ", " + b.scale.bufferType(b.n, 1) + " s";
  line(tensor("fp16", b.n, 1) + " s_t = tensor_buffer_to_tensor<ios17>(input = s)");
  line(tensor("fp16", b.n, rows) + " ys = mul(x = " + total + ", y = s_t)");
  line(b.out.bufferType(b.n, rows) +
       " y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
       "strides = tensor<int64, [4]>(" + b.out.strides(b.n) + "))");
  return "    func p" + std::to_string(rows) + "<ios18>(" + params + ") {\n" + body + "    } -> (y);\n";
}

void fill(const ane::Surface &surface, uint32_t rows, uint32_t width, std::mt19937 &random) {
  auto *bytes = static_cast<uint8_t *>(surface.buffer.contents());
  std::uniform_int_distribution<int> value(-127, 127);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t column = 0; column < width; ++column)
      bytes[uint64_t(row) * surface.strideBytes + column] = static_cast<uint8_t>(value(random));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: ane_projection METALLIB N K [ROWS...]\n");
    return 2;
  }
  if (const auto why = ane::unavailable()) {
    std::fprintf(stderr, "Neural Engine unavailable: %s\n", why->c_str());
    return 1;
  }
  metal::MetalBackend backend(argv[1]);
  Bench b{static_cast<uint32_t>(std::atoi(argv[2])), static_cast<uint32_t>(std::atoi(argv[3])), 0, {}, {}, {}, {}, {}};
  if (b.k % 128) {
    std::fprintf(stderr, "K must be a multiple of 128\n");
    return 2;
  }
  for (uint32_t done = 0; done < b.k; done += kSegment) b.widths.push_back(std::min(kSegment, b.k - done));
  b.segments = static_cast<uint32_t>(b.widths.size());
  std::vector<uint32_t> rowsList;
  for (int i = 4; i < argc; ++i) rowsList.push_back(static_cast<uint32_t>(std::atoi(argv[i])));
  if (rowsList.empty()) rowsList = {512, 1024, 2048};

  std::mt19937 random(7);
  for (uint32_t s = 0; s < b.segments; ++s) {
    b.x.push_back(ane::Surface::create(backend, b.widths[s], kMaximumRows, fp16Inputs ? Element::Float16 : Element::Int8));
    b.w.push_back(ane::Surface::create(backend, b.n, b.widths[s], fp16Weights ? Element::Float16 : Element::Int8));
    if (!fp16Inputs) fill(b.x.back(), b.widths[s], kMaximumRows, random);
    if (!fp16Weights) fill(b.w.back(), b.n, b.widths[s], random);
  }
  b.scale = ane::Surface::create(backend, b.n, 1, Element::Float16);
  for (uint32_t row = 0; row < b.n; ++row)
    *reinterpret_cast<_Float16 *>(static_cast<uint8_t *>(b.scale.buffer.contents()) + uint64_t(row) * b.scale.strideBytes) =
        _Float16(1.0f / 64.0f);
  b.out = ane::Surface::create(backend, b.n, kMaximumRows, Element::Float16);
  b.outT = ane::Surface::create(backend, kMaximumRows, b.n, Element::Float16);

  std::string mil = "program(1.3)\n{\n";
  for (uint32_t rows : rowsList) mil += function(b, rows);
  mil += "}\n";
  const ane::Program::Limits limits{std::chrono::seconds(300), {}};
  const auto compileStart = AwakeClock::now();
  ane::Program program(mil, {}, limits);
  std::printf("N=%u K=%u: compiled/loaded in %.1f s\n", b.n, b.k,
              std::chrono::duration<double>(AwakeClock::now() - compileStart).count());

  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t value = 0;
  for (uint32_t rows : rowsList) {
    const uint32_t procedure = program.procedure("p" + std::to_string(rows));
    std::vector<ane::Surface> inputs;
    for (const std::string &name : program.inputs(procedure)) {
      const uint32_t index = name.size() > 1 ? static_cast<uint32_t>(std::atoi(name.c_str() + 1)) : 0;
      if (name[0] == 'x') inputs.push_back(b.x.at(index));
      else if (name[0] == 'w') inputs.push_back(b.w.at(index));
      else if (name == "s") inputs.push_back(b.scale);
      else throw std::runtime_error("unexpected input " + name);
    }
    const ane::Program::Binding binding = program.bind(procedure, inputs, transposed ? b.outT : b.out);
    std::vector<double> samples;
    for (int rep = 0; rep < 13; ++rep) {
      struct Report final {
        std::mutex mutex;
        std::condition_variable changed;
        std::optional<bool> result;
      };
      auto report = std::make_shared<Report>();
      const uint64_t wait = ++value, signal = ++value;
      const auto start = AwakeClock::now();
      event.signal(wait);
      program.enqueue(binding, event, wait, signal, [report](bool ok) {
        std::lock_guard lock(report->mutex);
        report->result = ok;
        report->changed.notify_all();
      });
      std::unique_lock lock(report->mutex);
      report->changed.wait(lock, [&] { return report->result.has_value(); });
      const double ms = std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count();
      if (!*report->result) throw std::runtime_error("evaluation failed");
      if (rep < 3) std::printf("  rows=%u eval %d: %.2f ms\n", rows, rep, ms);
      if (rep >= 3) samples.push_back(ms);  // the first evaluations warm the service
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double ops = 2.0 * b.n * b.k * rows;
    std::printf("N=%u K=%u rows=%u: median %.2f ms (min %.2f)  %.1f TOPS\n", b.n, b.k, rows, median, samples.front(),
                ops / (median * 1e-3) / 1e12);
  }
  return 0;
}
