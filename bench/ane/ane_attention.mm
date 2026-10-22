// Neural Engine time for prefill attention of one chunk over its context, as a
// split would run its share of KV heads: per KV head, the chunk's GQA query
// heads stacked as rows (fp16) against int8 K and V of the context with
// per-token scales, softmax(Q K^T + causal mask) V. Times evaluations at
// several context lengths.
//
//   V=engine; xcrun clang++ -std=c++20 -O2 -fobjc-arc -mmacosx-version-min=26.4 \
//     -I$V/runtime -I$V/dev bench/ane/ane_attention.mm $V/build/engine/libsplash.a \
//     -framework Foundation -framework Metal -framework IOKit -framework IOSurface -o build/ane_attention
//   build/ane_attention METALLIB KV_HEADS [CONTEXT...]

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

constexpr uint32_t kRows = 2048, kGroup = 6, kHead = 256;

std::string tensor(const char *type, std::initializer_list<uint64_t> shape) {
  std::string text = std::string("tensor<") + type + ", [";
  bool first = true;
  for (uint64_t d : shape) {
    text += (first ? "" : ", ") + std::to_string(d);
    first = false;
  }
  return text + "]>";
}

void fillInt8(const ane::Surface &surface, uint32_t rows, uint32_t width, std::mt19937 &random) {
  auto *bytes = static_cast<int8_t *>(surface.buffer.contents());
  std::uniform_int_distribution<int> value(-127, 127);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t column = 0; column < width; ++column)
      bytes[uint64_t(row) * surface.strideBytes + column] = static_cast<int8_t>(value(random));
}

void fillHalf(const ane::Surface &surface, uint32_t rows, uint32_t width, float spread, std::mt19937 &random) {
  std::uniform_real_distribution<float> value(-spread, spread);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t column = 0; column < width; ++column)
      reinterpret_cast<_Float16 *>(static_cast<uint8_t *>(surface.buffer.contents()) +
                                   uint64_t(row) * surface.strideBytes)[column] = _Float16(value(random));
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ane_attention METALLIB KV_HEADS [CONTEXT...]\n");
    return 2;
  }
  if (const auto why = ane::unavailable()) {
    std::fprintf(stderr, "Neural Engine unavailable: %s\n", why->c_str());
    return 1;
  }
  metal::MetalBackend backend(argv[1]);
  const uint32_t heads = static_cast<uint32_t>(std::atoi(argv[2]));
  std::vector<uint32_t> contexts;
  for (int i = 3; i < argc; ++i) contexts.push_back(static_cast<uint32_t>(std::atoi(argv[i])));
  if (contexts.empty()) contexts = {2048, 4096, 6144};
  const uint32_t maxContext = *std::max_element(contexts.begin(), contexts.end());

  std::mt19937 random(11);
  // Per KV head: Q [kGroup * kRows, 256] fp16, K and V [context, 256] int8 with
  // per-token fp16 scales; the causal mask [kRows, kRows] for the chunk's own
  // columns is shared; the output stacks every head's [kGroup * kRows, 256].
  std::vector<ane::Surface> q, k, v, ks, vs;
  for (uint32_t h = 0; h < heads; ++h) {
    q.push_back(ane::Surface::create(backend, kGroup * kRows, kHead, Element::Float16));
    k.push_back(ane::Surface::create(backend, maxContext, kHead, Element::Int8));
    v.push_back(ane::Surface::create(backend, maxContext, kHead, Element::Int8));
    ks.push_back(ane::Surface::create(backend, 1, maxContext, Element::Float16));
    vs.push_back(ane::Surface::create(backend, maxContext, 1, Element::Float16));
    fillHalf(q.back(), kGroup * kRows, kHead, 0.06f, random);
    fillInt8(k.back(), maxContext, kHead, random);
    fillInt8(v.back(), maxContext, kHead, random);
    fillHalf(ks.back(), 1, maxContext, 0.01f, random);
    fillHalf(vs.back(), maxContext, 1, 0.01f, random);
  }
  ane::Surface mask = ane::Surface::create(backend, kRows, kRows, Element::Float16);
  for (uint32_t row = 0; row < kRows; ++row)
    for (uint32_t column = 0; column < kRows; ++column)
      reinterpret_cast<_Float16 *>(static_cast<uint8_t *>(mask.buffer.contents()) + uint64_t(row) * mask.strideBytes)
          [column] = column <= row ? _Float16(0.0f) : _Float16(-30000.0f);
  ane::Surface out = ane::Surface::create(backend, heads * kGroup * kRows, kHead, Element::Float16);

  std::string mil = "program(1.3)\n{\n";
  for (uint32_t context : contexts) {
    const uint32_t history = context - kRows;
    std::string params, body, outputs;
    const auto line = [&](const std::string &text) { body += "        " + text + ";\n"; };
    params = mask.bufferType(kRows, kRows) + " m";
    line(tensor("fp16", {1, 1, kRows, kRows}) + " m_t = tensor_buffer_to_tensor<ios17>(input = m)");
    const std::string g = std::to_string(kGroup), r = std::to_string(kRows), c = std::to_string(context);
    for (uint32_t h = 0; h < heads; ++h) {
      const std::string i = std::to_string(h);
      params += ", " + q[h].bufferType(kGroup * kRows, kHead) + " q" + i + ", " + k[h].bufferType(context, kHead) +
                " k" + i + ", " + v[h].bufferType(context, kHead) + " v" + i + ", " + ks[h].bufferType(1, context) +
                " ks" + i + ", " + vs[h].bufferType(context, 1) + " vs" + i;
      line(tensor("fp16", {1, 1, kGroup * kRows, kHead}) + " q" + i + "_t = tensor_buffer_to_tensor<ios17>(input = q" + i + ")");
      line(tensor("int8", {1, 1, context, kHead}) + " k" + i + "_t = tensor_buffer_to_tensor<ios17>(input = k" + i + ")");
      line(tensor("int8", {1, 1, context, kHead}) + " v" + i + "_t = tensor_buffer_to_tensor<ios17>(input = v" + i + ")");
      line(tensor("fp16", {1, 1, 1, context}) + " ks" + i + "_t = tensor_buffer_to_tensor<ios17>(input = ks" + i + ")");
      line(tensor("fp16", {1, 1, context, 1}) + " vs" + i + "_t = tensor_buffer_to_tensor<ios17>(input = vs" + i + ")");
      line(tensor("fp16", {1, 1, context, kHead}) + " k" + i + "_d = dequantize(input = k" + i + "_t, scale = fp16(0x1p-7))");
      line(tensor("fp16", {1, 1, context, kHead}) + " v" + i + "_d = dequantize(input = v" + i + "_t, scale = fp16(0x1p-7))");
      line(tensor("fp16", {1, 1, kGroup * kRows, context}) + " s" + i +
           " = matmul(transpose_x = bool(false), transpose_y = bool(true), x = q" + i + "_t, y = k" + i + "_d)");
      line(tensor("fp16", {1, 1, kGroup * kRows, context}) + " ss" + i + " = mul(x = s" + i + ", y = ks" + i + "_t)");
      line(tensor("fp16", {1, kGroup, kRows, context}) + " s4" + i + " = reshape(x = ss" + i +
           ", shape = tensor<int32, [4]>([1, " + g + ", " + r + ", " + c + "]))");
      if (history) {
        const std::string hs = std::to_string(history);
        line(tensor("fp16", {1, kGroup, kRows, history}) + " sh" + i + " = slice_by_size(x = s4" + i +
             ", begin = tensor<int32, [4]>([0, 0, 0, 0]), size = tensor<int32, [4]>([1, " + g + ", " + r + ", " + hs + "]))");
        line(tensor("fp16", {1, kGroup, kRows, kRows}) + " sc" + i + " = slice_by_size(x = s4" + i +
             ", begin = tensor<int32, [4]>([0, 0, 0, " + hs + "]), size = tensor<int32, [4]>([1, " + g + ", " + r + ", " +
             r + "]))");
        line(tensor("fp16", {1, kGroup, kRows, kRows}) + " scm" + i + " = add(x = sc" + i + ", y = m_t)");
        line(tensor("fp16", {1, kGroup, kRows, context}) + " sm" + i +
             " = concat(axis = int32(3), interleave = bool(false), values = (sh" + i + ", scm" + i + "))");
      } else {
        line(tensor("fp16", {1, kGroup, kRows, context}) + " sm" + i + " = add(x = s4" + i + ", y = m_t)");
      }
      if (std::getenv("NOSOFTMAX"))
        line(tensor("fp16", {1, kGroup, kRows, context}) + " p" + i + " = mul(x = sm" + i + ", y = fp16(0x1p-12))");
      else
        line(tensor("fp16", {1, kGroup, kRows, context}) + " p" + i + " = softmax(x = sm" + i + ", axis = int32(3))");
      line(tensor("fp16", {1, 1, kGroup * kRows, context}) + " p2" + i + " = reshape(x = p" + i +
           ", shape = tensor<int32, [4]>([1, 1, " + std::to_string(kGroup * kRows) + ", " + c + "]))");
      line(tensor("fp16", {1, 1, context, kHead}) + " vv" + i + " = mul(x = v" + i + "_d, y = vs" + i + "_t)");
      line(tensor("fp16", {1, 1, kGroup * kRows, kHead}) + " o" + i +
           " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = p2" + i + ", y = vv" + i + ")");
      outputs += (outputs.empty() ? "" : ", ") + std::string("o") + i;
    }
    if (heads > 1)
      line(tensor("fp16", {1, 1, heads * kGroup * kRows, kHead}) +
           " oall = concat(axis = int32(2), interleave = bool(false), values = (" + outputs + "))");
    else
      line(tensor("fp16", {1, 1, kGroup * kRows, kHead}) + " oall = identity(x = o0)");
    line(out.bufferType(heads * kGroup * kRows, kHead) +
         " y = tensor_to_tensor_buffer<ios17>(input = oall, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), "
         "strides = tensor<int64, [4]>(" + out.strides(heads * kGroup * kRows) + "))");
    mil += "    func a" + c + "<ios18>(" + params + ") {\n" + body + "    } -> (y);\n";
  }
  mil += "}\n";
  const auto compileStart = AwakeClock::now();
  ane::Program program(mil, {}, ane::Program::Limits{std::chrono::seconds(600), {}});
  std::printf("%u KV heads: compiled/loaded in %.1f s\n", heads,
              std::chrono::duration<double>(AwakeClock::now() - compileStart).count());

  metal::SharedEvent event = backend.newSharedEvent();
  uint64_t value = 0;
  for (uint32_t context : contexts) {
    const uint32_t procedure = program.procedure("a" + std::to_string(context));
    std::vector<ane::Surface> inputs;
    for (const std::string &name : program.inputs(procedure)) {
      if (name == "m") {
        inputs.push_back(mask);
        continue;
      }
      const size_t digits = name.find_first_of("0123456789");
      const std::string kind = name.substr(0, digits);
      const uint32_t h = static_cast<uint32_t>(std::atoi(name.c_str() + digits));
      if (kind == "q") inputs.push_back(q.at(h));
      else if (kind == "k") inputs.push_back(k.at(h));
      else if (kind == "v") inputs.push_back(v.at(h));
      else if (kind == "ks") inputs.push_back(ks.at(h));
      else if (kind == "vs") inputs.push_back(vs.at(h));
      else throw std::runtime_error("unexpected input " + name);
    }
    const ane::Program::Binding binding = program.bind(procedure, inputs, out);
    std::vector<double> samples;
    for (int rep = 0; rep < 8; ++rep) {
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
      if (!*report->result) throw std::runtime_error("evaluation failed");
      if (rep >= 2) samples.push_back(std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    // Causal work of the chunk: its history fully, its own block half.
    const double pairs = double(kRows) * (context - kRows) + double(kRows) * kRows / 2.0;
    const double flops = 4.0 * pairs * kHead * kGroup * heads;
    std::printf("%u KV heads, context %u: median %.2f ms  %.1f effective TFLOPS (causal)\n", heads, context,
                samples[samples.size() / 2], flops / (samples[samples.size() / 2] * 1e-3) / 1e12);
  }
  return 0;
}
