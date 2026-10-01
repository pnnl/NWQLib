// NWQLib's bounded file protocol over one selected public NWQ-Sim target.
// Gates use Qiskit's U(theta, phi, lambda), CX and least-significant-first wires.
#include <nwq_util.hpp>
// PURITY_CHECK enables a print-only O(2^n) norm scan after every fused gate.
// Configure it off before compiling the native kernel; the requested
// readout and sampler keep their normal computations. No extra diagnostic pass.
#undef PURITY_CHECK
#include <backendManager.hpp>
#include <private/nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NWQLIB_NWQSIM_REVISION
#define NWQLIB_NWQSIM_REVISION "unrecorded"
#endif
// build_runner defines both roundoff macros only when the NWQ-Sim revision
// descends from the revision the roundoff derivation read
// (_ROUNDOFF_BASE_REVISION in nwqsim.py); --describe reports null otherwise.
#ifndef NWQLIB_ROUNDOFF_BASE_REVISION
#define NWQLIB_ROUNDOFF_BASE_REVISION ""
#endif
#ifndef NWQLIB_BACKEND
#define NWQLIB_BACKEND "CPU"
#endif
#ifndef NWQLIB_METHOD
#define NWQLIB_METHOD "SV"
#endif

using Json = nlohmann::json;
using Index = std::uint64_t;
// Request, result, failure packet, --describe and --phase-factor format.
// nwqsim.py checks the same string; there is no reader for another format.
static const char* const protocol = "nwqlib.nwqsim/3";
namespace fs = std::filesystem;
static const std::string backend = NWQLIB_BACKEND, method = NWQLIB_METHOD;
static const bool exact_readouts = backend == "CPU" && method == "SV";
#ifdef GPU_ERROR_CHECK
static const bool gpu_error_check = true;
#else
static const bool gpu_error_check = false;
#endif

static bool compiled_target() {
    if (method != "SV" && method != "DM") return false;
    if (backend == "CPU") return true;
#ifdef MPI_ENABLED
    if (backend == "MPI") return method == "SV";
#endif
#ifdef CUDA_ENABLED
    if (backend == "NVGPU") return gpu_error_check;
#endif
#ifdef HIP_ENABLED
    if (backend == "AMDGPU") return gpu_error_check;
#endif
    return false;
}

#if defined(__aarch64__) || defined(__arm64__)
static const char* architecture = "arm64";
#elif defined(__x86_64__)
static const char* architecture = "x86_64";
#else
static const char* architecture = "other";
#endif
static const std::string roundoff_base = NWQLIB_ROUNDOFF_BASE_REVISION;
#ifdef NWQLIB_ROUNDOFF_SOURCES_IDENTICAL
static const Json roundoff_sources_identical = static_cast<bool>(NWQLIB_ROUNDOFF_SOURCES_IDENTICAL);
#else
static const Json roundoff_sources_identical = nullptr;
#endif

// --describe is inert. Execution initializes MPI before the public factory;
// all state objects are destroyed before this guard finalizes communication.
struct Ranks {
    int rank = 0, size = 1;
    Ranks(int& argc, char**& argv) {
#ifdef MPI_ENABLED
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    }
    void barrier() const {
#ifdef MPI_ENABLED
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    void abort() const {
#ifdef MPI_ENABLED
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
    ~Ranks() {
#ifdef MPI_ENABLED
        MPI_Finalize();
#endif
    }
};

static Index count(const Json& value, const std::string& name) {
    if (!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned()
                                      && value.get<std::int64_t>() < 0)) {
        throw std::invalid_argument(name + " must be a nonnegative integer");
    }
    return value.get<Index>();
}

static double real(const Json& value) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw std::invalid_argument("gate arguments must be finite real numbers");
    }
    return value.get<double>();
}

static std::complex<double> phase_factor(const Json& value) {
    if (!value.is_array() || value.size() != 2) {
        throw std::invalid_argument("an output phase factor needs two finite real components");
    }
    return {real(value.at(0)), real(value.at(1))};
}

static std::vector<Index> bits(const Json& values, Index width, bool distinct = true) {
    if (!values.is_array()) throw std::invalid_argument("bits must be an array");
    std::vector<Index> result;
    for (const auto& item : values) {
        const Index bit = count(item, "bit");
        if (bit >= width || (distinct && std::find(result.begin(), result.end(), bit) != result.end())) {
            throw std::invalid_argument("bits must be distinct and in range");
        }
        result.push_back(bit);
    }
    return result;
}

static std::string bit_string(Index value, Index width) {
    std::string result(width, '0');
    for (Index bit = 0; bit < width; ++bit) {
        result[width - 1 - bit] = ((value >> bit) & 1) ? '1' : '0';
    }
    return result;
}

// A directory claim is exclusive across processes. A crash deliberately leaves
// it behind: an uncollected attempt must be reconciled, never blindly rerun.
// Binary sidecars (OUTPUT.<name>: amplitudes, probability marginals) are
// staged in the claim and published by hard link before the result; the
// publication owns every sidecar it staged and removes each one that is still
// its own inode when the attempt ends without its terminal result.
struct Publication {
    fs::path output, claim;
    std::vector<std::string> staged;
    Index sidecar_bytes = 0;
    bool owns_claim = false, started = false, terminal = false;

    explicit Publication(const fs::path& path)
        : output(path), claim(path.string() + ".claim") {
        if (!fs::create_directory(claim)) {
            throw std::invalid_argument("native output is already claimed; reconcile that attempt before rerunning");
        }
        owns_claim = true;
    }
    fs::path sidecar(const std::string& name) const { return fs::path(output.string() + "." + name); }
    // Refuse when the result or a named sidecar already exists; an unrelated
    // OUTPUT.* file is neither refused nor touched.
    void check_outputs(const std::vector<std::string>& names) const {
        bool existing = fs::symlink_status(output).type() != fs::file_type::not_found;
        for (const auto& name : names) {
            existing = existing || fs::symlink_status(sidecar(name)).type() != fs::file_type::not_found;
        }
        if (existing) throw std::invalid_argument("native output already exists; fetch it instead of rerunning");
    }
    // Write one sidecar of `bytes` bytes through `write(stream)` and publish it.
    template <class Writer>
    Json write_sidecar(const std::string& name, const char* dtype, Index count, Index bytes, Writer write) {
        const fs::path temporary = claim / name;
        staged.push_back(name);
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.exceptions(std::ios::badbit | std::ios::failbit);
        write(stream);
        stream.close();
        if (fs::file_size(temporary) != bytes) throw std::runtime_error("native sidecar size differs from its admitted bytes");
        fs::create_hard_link(temporary, sidecar(name));
        sidecar_bytes += bytes;
        return Json{{"file", sidecar(name).filename().string()}, {"count", count}, {"bytes", bytes}, {"dtype", dtype}};
    }
    void discard_sidecars() noexcept {
        for (const auto& name : staged) {
            std::error_code error;
            // Only unlink a sidecar still owned by this attempt's staged inode.
            if (fs::equivalent(claim / name, sidecar(name), error) && !error) {
                fs::remove(sidecar(name), error);
            }
        }
    }
    ~Publication() {
        if (!owns_claim) return;
        if (!terminal) discard_sidecars();
        std::error_code error;
        if (started && !terminal) {
            // If terminal publication is unavailable, keep the exclusive
            // marker for reconciliation but release this attempt's payloads.
            for (const auto& name : staged) fs::remove(claim / name, error);
            fs::remove(claim / "result", error);
        } else {
            fs::remove_all(claim, error);
        }
    }
};

static void write_json(const Publication& publication, const Json& result, Index cap) {
    const std::string data = result.dump();
    if (data.size() > cap) throw std::length_error("native result exceeds its admitted byte limit");
    const fs::path temporary = publication.claim / "result";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.write(data.data(), static_cast<std::streamsize>(data.size()));
    stream.close();
    fs::create_hard_link(temporary, publication.output);
}

struct CompensatedSum {
    double value = 0, correction = 0;
    void add(double term) {
        const double adjusted = term - correction;
        const double next = value + adjusted;
        correction = (next - value) - adjusted;
        value = next;
    }
};

static double pauli_value(const double* re, const double* im, Index dim,
                         const std::string& label, Index width) {
    if (label.size() != width) throw std::invalid_argument("Pauli label width mismatch");
    Index flip = 0, phase_bits = 0;
    unsigned y_count = 0;
    for (Index bit = 0; bit < width; ++bit) {
        const char axis = label[width - 1 - bit];
        if (axis == 'X' || axis == 'Y') flip |= Index{1} << bit;
        if (axis == 'Z' || axis == 'Y') phase_bits |= Index{1} << bit;
        if (axis == 'Y') ++y_count;
        if (axis != 'I' && axis != 'X' && axis != 'Y' && axis != 'Z') {
            throw std::invalid_argument("unsupported Pauli axis");
        }
    }
    // Writing each Y as iXZ, Z gives the sign from the input bit and X flips it:
    // P|x> = i^(#Y) (-1)^popcount(x & phase_bits) |x XOR flip>.
    // Hence <psi|P|psi> = sum_x i^(#Y) (-1)^popcount(x & phase_bits) conj(b) a
    // with a = psi[x] and b = psi[x XOR flip]. P is Hermitian, so the sum is
    // real and only real parts are accumulated: Re(conj(b) a) = b.re a.re +
    // b.im a.im for even #Y, and Re(i conj(b) a) = b.im a.re - b.re a.im for
    // odd #Y. The factor i^2 = -1 negates both when #Y mod 4 is 2 or 3.
    // Read the native state directly: no dense P and no temporary P|psi>.
    // Compensated binary64 avoids platform-dependent software long-double
    // arithmetic while controlling cancellation in this single readout pass.
    CompensatedSum value;
    const unsigned phase = y_count % 4;
    for (Index x = 0; x < dim; ++x) {
        const Index other = x ^ flip;
        const double product = (phase % 2 == 0)
            ? re[other] * re[x] + im[other] * im[x]
            : im[other] * re[x] - re[other] * im[x];
        const bool negate = (__builtin_parityll(x & phase_bits) != 0) != (phase >= 2);
        value.add(negate ? -product : product);
    }
    return value.value;
}

// Probability marginal of the host amplitudes in the requested qubit order,
// with compensated accumulation per bin and no second state.
//
// Bit zero of a marginal index is the first observed wire. For observed wires
// (0,...,k-1), a native basis index contributes to x & (2**k-1). Unobserved
// high bits are summed into that bin. The native index itself is the bin only
// for an identity-ordered full-register observation.
//
// For native index x = sum_j x_j*2^j and ordered observed wires
// (q_0,...,q_(k-1)), the bin is b(x) = sum_i ((x >> q_i) & 1)*2^i; for the
// identity-ordered prefix q_i = i this is x mod 2^k = x & (2^k-1), and
// p_b = sum_h |psi_(b+2^k h)|^2 over h = 0..2^(w-k)-1. Premises: distinct
// validated wires below the native width, unsigned Index and an admitted
// 2^k. The fast path changes no summation order, payload or identity: each
// x adds to the same bin in the same order as the general loop.
static std::vector<CompensatedSum> marginal(const double* re, const double* im, Index dim,
                                            const std::vector<Index>& measured) {
    const Index bins = Index{1} << measured.size();
    std::vector<CompensatedSum> probabilities(bins);
    const bool identity_prefix = [&]() {
        for (std::size_t bit = 0; bit < measured.size(); ++bit)
            if (measured[bit] != bit) return false;
        return true;
    }();
    const Index mask = bins - Index{1};
    for (Index x = 0; x < dim; ++x) {
        Index bin;
        if (identity_prefix) {
            bin = x & mask;
        } else {
            bin = 0;
            for (std::size_t bit = 0; bit < measured.size(); ++bit)
                bin |= ((x >> measured[bit]) & Index{1}) << bit;
        }
        probabilities[bin].add(re[x] * re[x] + im[x] * im[x]);
    }
    return probabilities;
}

// Append one typed U or CX request gate to a native circuit.
static void append_gate(NWQSim::Circuit& circuit, const Json& gate, Index width, Index dim, const Ranks& ranks) {
    const auto wires = bits(gate.at("qubits"), width);
    // Global U/CX kernels transfer the full local state using MPI's int
    // count. Local-only gates and measurements do not make that transfer.
    if (backend == "MPI" && dim / ranks.size > static_cast<Index>(std::numeric_limits<int>::max())
        && std::any_of(wires.begin(), wires.end(), [&](Index bit) { return (Index{1} << bit) >= dim / ranks.size; })) {
        throw std::invalid_argument("MPI global gates require an int-sized local state transfer count");
    }
    const auto& params = gate.at("params");
    if (!params.is_array()) throw std::invalid_argument("gate params must be an array");
    if (gate.at("name") == "u" && wires.size() == 1 && params.size() == 3) {
        circuit.U(real(params[0]), real(params[1]), real(params[2]), wires[0]);
    } else if (gate.at("name") == "cx" && wires.size() == 2 && params.empty()) {
        circuit.CX(wires[0], wires[1]);
    } else {
        throw std::invalid_argument("native payload requires lowered U/CX gates");
    }
}

static std::shared_ptr<NWQSim::Circuit> gate_circuit(const Json& gates, Index width, Index dim, const Ranks& ranks) {
    if (!gates.is_array()) throw std::invalid_argument("gates must be an array");
    auto circuit = std::make_shared<NWQSim::Circuit>(width);
    for (const auto& gate : gates) append_gate(*circuit, gate, width, dim, ranks);
    return circuit;
}

// Trajectory readout (CPU/SV): evaluate all selected point readouts while
// advancing one coherent simulator state. A point's boundary precedes its
// reversible readout view. Pauli saves leave the state unchanged. A
// probability view applies its selected tail, saves the marginal, and applies
// the exact inverse before continuation. Count actual native operations
// through the final observation, including all executed view operations, and
// admit the sum of the point payloads before acquisition.
//
// The request's gates run up to the last point's boundary and nothing after
// it. Point k's boundary is its native gate position; the gates between two
// boundaries form one segment, evolved by one sim call (NWQ-Sim fuses within
// a segment and neither resets nor normalizes the state between calls). A
// view's inverse is the reversed, adjointed tail the adapter lowered, applied
// only when a later point follows. Nothing reinitializes, renormalizes or
// re-injects the state; a phase is applied to an output copy only.
// executed_gates reports the U/CX count actually executed: every segment, every
// forward tail and every inverse that ran.
static Json execute_trajectory(Json& request, Publication* publication, Index& native_simulations,
                               const Ranks& ranks, Index width, Index dim, Index state_bytes) {
    if (!exact_readouts) throw std::invalid_argument("selected native target supports counts only");
    const auto& observation = request.at("observation");
    if (count(request.at("shots"), "shots") != 0 || count(request.at("num_clbits"), "num_clbits") != 0) {
        throw std::invalid_argument("a trajectory has zero shots and no classical bits");
    }
    const Index seed = count(request.at("seed"), "seed");
    if (seed > static_cast<Index>(std::numeric_limits<NWQSim::IdxType>::max())) throw std::invalid_argument("invalid native shots or seed");
    const Index output_cap = count(request.at("max_output_bytes"), "max_output_bytes");
    const Index buffer_cap = count(request.at("max_buffer_bytes"), "max_buffer_bytes");
    const auto& gates = request.at("gates");
    const auto& points = observation.at("points");
    if (!gates.is_array() || !points.is_array() || points.empty()) {
        throw std::invalid_argument("a trajectory needs a gate array and at least one point");
    }
    struct Point {
        std::string id, kind;
        Json labels;
        std::vector<Index> qubits;
        std::shared_ptr<NWQSim::Circuit> segment, tail, inverse;
        bool phased = false;
        std::complex<double> phase{1.0, 0.0};
    };
    std::vector<Point> parsed;
    std::set<std::string> ids;
    std::vector<std::string> sidecar_names;
    Index previous = 0, executed = 0, marginal_buffer = 0, sidecar_bytes = 0, pauli_items = 0;
    Json entries = Json::array();
    for (std::size_t index = 0; index < points.size(); ++index) {
        const auto& item = points[index];
        Point point;
        point.id = item.at("id").get<std::string>();
        point.kind = item.at("kind").get<std::string>();
        if (!ids.insert(point.id).second) throw std::invalid_argument("duplicate trajectory point");
        const Index boundary = count(item.at("boundary"), "boundary");
        if (boundary < previous || boundary > gates.size()) {
            throw std::invalid_argument("trajectory boundaries must be nondecreasing gate positions");
        }
        point.segment = std::make_shared<NWQSim::Circuit>(width);
        for (Index position = previous; position < boundary; ++position) {
            append_gate(*point.segment, gates[position], width, dim, ranks);
        }
        executed += boundary - previous;
        previous = boundary;
        Json entry{{"id", point.id}, {"kind", point.kind}, {"boundary", boundary}};
        const std::string stem = "p" + std::to_string(index);
        if (point.kind == "pauli") {
            point.labels = item.at("labels");
            std::set<std::string> unique_labels;
            if (!point.labels.is_array() || point.labels.empty()) throw std::invalid_argument("invalid Pauli readout labels");
            for (const auto& label : point.labels) {
                const std::string text = label;
                if (text.size() != width || text.find_first_not_of("IXYZ") != std::string::npos
                    || !unique_labels.insert(text).second) {
                    throw std::invalid_argument("invalid native Pauli label");
                }
            }
            pauli_items += point.labels.size();
            entry["pauli"] = Json::object();
        } else if (point.kind == "probabilities") {
            point.qubits = bits(item.at("qubits"), width);
            if (point.qubits.empty()) throw std::invalid_argument("probabilities require a nonempty measured register");
            const Index bins = Index{1} << point.qubits.size();
            if (bins > buffer_cap / sizeof(CompensatedSum) || bins > output_cap / sizeof(double)) {
                throw std::length_error("marginal buffer exceeds byte limit");
            }
            marginal_buffer = std::max(marginal_buffer, bins * sizeof(CompensatedSum));
            sidecar_names.push_back(stem + ".probabilities");
            entry["probabilities"] = {{"file", publication->sidecar(sidecar_names.back()).filename().string()},
                                      {"count", bins}, {"bytes", bins * sizeof(double)}, {"dtype", "float64-native"}};
            if (bins * sizeof(double) > output_cap - std::min(output_cap, sidecar_bytes)) {
                throw std::length_error("native output envelope exceeds its admitted byte limit");
            }
            sidecar_bytes += bins * sizeof(double);
        } else if (point.kind == "state") {
            if (2 * sizeof(double) * dim > output_cap - std::min(output_cap, sidecar_bytes)) {
                throw std::length_error("native output envelope exceeds its admitted byte limit");
            }
            sidecar_names.push_back(stem + ".amplitudes");
            entry["amplitudes"] = {{"file", publication->sidecar(sidecar_names.back()).filename().string()},
                                   {"count", dim}, {"bytes", 2 * sizeof(double) * dim}, {"dtype", "complex128-native"}};
            sidecar_bytes += 2 * sizeof(double) * dim;
        } else {
            throw std::invalid_argument("unsupported trajectory point readout");
        }
        if (!item.at("phase_factor").is_null()) {
            if (point.kind != "state") throw std::invalid_argument("only a saved state takes an output phase");
            point.phased = true;
            point.phase = phase_factor(item.at("phase_factor"));
        }
        const auto& view = item.at("view");
        if (!view.is_null()) {
            if (point.kind == "state") throw std::invalid_argument("a readout view applies to Pauli or probability points");
            point.tail = gate_circuit(view.at("tail"), width, dim, ranks);
            point.inverse = gate_circuit(view.at("inverse"), width, dim, ranks);
            if (point.tail->num_gates() != point.inverse->num_gates()) {
                throw std::invalid_argument("a view inverse reverses its tail gate for gate");
            }
            executed += point.tail->num_gates();
            if (index + 1 < points.size()) executed += point.inverse->num_gates();
        }
        entries.push_back(std::move(entry));
        parsed.push_back(std::move(point));
    }
    if (previous != gates.size()) throw std::invalid_argument("no gate follows the last trajectory observation");
    // Every gate is in its segment circuit now; release the parsed array.
    request.erase("gates");
    publication->check_outputs(sidecar_names);
    Json result{{"format", protocol}, {"input_id", request.at("input_id")},
                {"status", "completed"}, {"backend", backend}, {"method", method}, {"ranks", ranks.size},
                {"nwqsim_revision", NWQLIB_NWQSIM_REVISION}, {"native_simulations", 0},
                {"state_buffer_bytes", state_bytes}, {"buffer_scope", "known arrays per process; GPU host plus device"},
                {"additional_workspace_bytes", nullptr}, {"kind", "trajectory"}, {"shots", 0},
                {"executed_gates", executed}, {"readout_buffer_bytes", marginal_buffer}, {"trajectory", entries}};
    // Admit the state, one marginal buffer at a time, the sidecars and the
    // Pauli values of the JSON envelope (32 bytes per value, four for its
    // punctuation, and its label) before the state factory allocates.
    if (state_bytes > buffer_cap || marginal_buffer > buffer_cap - state_bytes) {
        throw std::length_error("native state/readout arrays exceed their admitted byte limit");
    }
    const Index header_bytes = result.dump().size();
    constexpr Index number_bytes = 32;
    if (header_bytes > output_cap || sidecar_bytes > output_cap - header_bytes
        || pauli_items > (output_cap - header_bytes - sidecar_bytes) / (width + number_bytes + 4)) {
        throw std::length_error("native output envelope exceeds its admitted byte limit");
    }
    NWQSim::Config::PRINT_SIM_TRACE = false;
    NWQSim::Config::ENABLE_NOISE = false;
    NWQSim::Config::ENABLE_FUSION = true;
    auto state = BackendManager::create_state(backend, width, method);
    state->set_seed(seed);
    ranks.barrier();
    publication->started = true;
    ++native_simulations;
    for (std::size_t index = 0; index < parsed.size(); ++index) {
        auto& point = parsed[index];
        auto& entry = result["trajectory"][index];
        // Empty segments cost no gate and no call.
        if (!point.segment->is_empty()) state->sim(point.segment);
        if (point.tail && !point.tail->is_empty()) state->sim(point.tail);
        const double* re = state->get_real();
        const double* im = state->get_imag();
        const std::string stem = "p" + std::to_string(index);
        if (point.kind == "pauli") {
            for (const auto& item : point.labels) {
                const std::string label = item;
                const double value = pauli_value(re, im, dim, label, width);
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite native Pauli readout");
                entry["pauli"][label] = value;
            }
        } else if (point.kind == "probabilities") {
            const auto probabilities = marginal(re, im, dim, point.qubits);
            entry["probabilities"] = publication->write_sidecar(stem + ".probabilities", "float64-native",
                probabilities.size(), probabilities.size() * sizeof(double), [&](std::ofstream& stream) {
                    for (const auto& bin : probabilities) {
                        if (!std::isfinite(bin.value)) throw std::runtime_error("nonfinite native probability readout");
                        stream.write(reinterpret_cast<const char*>(&bin.value), sizeof(double));
                    }
                });
        } else {
            const std::complex<double> phase = point.phase;
            entry["amplitudes"] = publication->write_sidecar(stem + ".amplitudes", "complex128-native", dim,
                2 * dim * sizeof(double), [&](std::ofstream& stream) {
                    for (Index x = 0; x < dim; ++x) {
                        const auto amplitude = point.phased ? phase * std::complex<double>(re[x], im[x])
                                                            : std::complex<double>(re[x], im[x]);
                        const std::array<double, 2> pair{amplitude.real(), amplitude.imag()};
                        if (!std::isfinite(pair[0]) || !std::isfinite(pair[1])) {
                            throw std::runtime_error("nonfinite native amplitude readout");
                        }
                        stream.write(reinterpret_cast<const char*>(pair.data()), sizeof(pair));
                    }
                });
        }
        // The inverse restores the continuation state only when a later point follows.
        if (point.inverse && index + 1 < parsed.size() && !point.inverse->is_empty()) state->sim(point.inverse);
    }
    result["native_simulations"] = native_simulations;
    return result;
}

static Json execute(Json& request, Publication* publication, Index& native_simulations, const Ranks& ranks) {
    if (request.at("format") != protocol) {
        throw std::invalid_argument("unsupported native input format");
    }
    if (request.at("backend") != backend || request.at("method") != method
        || count(request.at("ranks"), "ranks") != static_cast<Index>(ranks.size)) {
        throw std::invalid_argument("input target/ranks differ from this native build/launch");
    }
    const Index width = count(request.at("num_qubits"), "num_qubits");
    // Signed NWQ-Sim index shifts and byte products must remain representable.
    if (width == 0 || width > (method == "SV" ? 58 : 29)) throw std::invalid_argument("unsupported native width");
    const Index dim = Index{1} << width;
    if (backend == "MPI" && (method != "SV" || (ranks.size & (ranks.size - 1)) != 0
                              || dim / ranks.size < 4)) {
        throw std::invalid_argument("MPI requires SV, power-of-two ranks and at least two local qubits");
    }
    const Index stored_dim = method == "SV" ? dim : dim * dim;
    const Index state_bytes = backend == "MPI" ? 32 * stored_dim / ranks.size
        : backend != "CPU" ? 48 * stored_dim + 16
        : method == "SV" ? 24 * dim + 8 : 16 * stored_dim + 8 * (dim + 1);
    const auto& observation = request.at("observation");
    const std::string kind = observation.at("kind");
    if (kind == "trajectory") {
        return execute_trajectory(request, publication, native_simulations, ranks, width, dim, state_bytes);
    }
    if (kind != "counts" && kind != "pauli" && kind != "probabilities" && kind != "amplitudes") {
        throw std::invalid_argument("unsupported native readout");
    }
    if (!exact_readouts && kind != "counts") throw std::invalid_argument("selected native target supports counts only");
    const Index shots = count(request.at("shots"), "shots");
    const Index seed = count(request.at("seed"), "seed");
    if ((kind == "counts") != (shots > 0)
        || shots > static_cast<Index>(std::numeric_limits<NWQSim::IdxType>::max())
        || seed > static_cast<Index>(std::numeric_limits<NWQSim::IdxType>::max())
        || (backend == "MPI" && shots > static_cast<Index>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument("invalid native shots or seed");
    }
    const Index output_cap = count(request.at("max_output_bytes"), "max_output_bytes");
    const Index classical_width = count(request.at("num_clbits"), "num_clbits");
    if (classical_width > 63) throw std::invalid_argument("native counts support at most 63 classical bits");
    const auto measured = bits(observation.at("qubits"), width, kind != "counts");
    const auto classical = bits(observation.at("clbits"), classical_width);
    if (kind == "counts" && (measured.empty() || measured.size() != classical.size())) {
        throw std::invalid_argument("counts require a nonempty qubit/classical map");
    }
    if (kind != "counts" && !classical.empty()) {
        throw std::invalid_argument("exact readout cannot contain classical measurement bits");
    }
    if (kind == "probabilities" && measured.empty()) {
        throw std::invalid_argument("probabilities require a nonempty measured register");
    }
    if (!request.at("gates").is_array()) throw std::invalid_argument("gates must be an array");
    auto circuit = std::make_shared<NWQSim::Circuit>(width);
    for (const auto& gate : request.at("gates")) append_gate(*circuit, gate, width, dim, ranks);
    // The circuit now holds every gate. Release the parsed gate array before
    // the state factory allocates, so it does not stay resident through the
    // evolution. The scalar fields read below remain in the request.
    request.erase("gates");
    const std::complex<double> output_phase = kind == "amplitudes"
        ? phase_factor(request.at("phase_factor")) : std::complex<double>(1.0, 0.0);
    const auto& labels = observation.at("labels");
    if (!labels.is_array()) throw std::invalid_argument("Pauli labels must be an array");
    std::set<std::string> unique_labels;
    for (const auto& item : labels) {
        const std::string label = item;
        if (label.size() != width || label.find_first_not_of("IXYZ") != std::string::npos) {
            throw std::invalid_argument("invalid native Pauli label");
        }
        if (!unique_labels.insert(label).second) throw std::invalid_argument("duplicate Pauli label");
    }
    if ((kind == "pauli") != !labels.empty()) throw std::invalid_argument("invalid Pauli readout labels");
    Json result{{"format", protocol}, {"input_id", request.at("input_id")},
                {"status", "completed"}, {"backend", backend}, {"method", method}, {"ranks", ranks.size},
                {"nwqsim_revision", NWQLIB_NWQSIM_REVISION}, {"native_simulations", 0},
                {"state_buffer_bytes", state_bytes}, {"buffer_scope", "known arrays per process; GPU host plus device"},
                {"additional_workspace_bytes", nullptr}, {"kind", kind}, {"shots", shots}};
    // Admit state, readout and serialized output together before the native
    // state factory allocates the simulator population.
    const Index buffer_cap = count(request.at("max_buffer_bytes"), "max_buffer_bytes");
    Index extra_buffers = 0, output_items = 0, key_width = 0, artifact_bytes = 0;
    if (kind == "counts") {
        const Index sample_bytes = backend == "MPI" ? 24 : backend == "CPU" ? 8 : 32;
        if (shots > buffer_cap / sample_bytes) throw std::length_error("sample buffer exceeds byte limit");
        extra_buffers = shots * sample_bytes;
        const std::set<Index> unique_bits(measured.begin(), measured.end());
        output_items = std::min(shots, Index{1} << unique_bits.size());
        key_width = classical_width;
        result["counts"] = Json::object();
    } else if (kind == "pauli") {
        output_items = labels.size();
        key_width = width;
        result["pauli"] = Json::object();
    } else if (kind == "probabilities") {
        // Dense float64 marginal sidecar of 2**k values; the compensated
        // accumulators are the resident readout buffer.
        const Index bins = Index{1} << measured.size();
        if (bins > buffer_cap / sizeof(CompensatedSum)) throw std::length_error("marginal buffer exceeds byte limit");
        extra_buffers = bins * sizeof(CompensatedSum);
        artifact_bytes = bins * sizeof(double);
        result["probabilities"] = {{"file", publication->sidecar("probabilities").filename().string()},
                                   {"count", bins}, {"bytes", artifact_bytes}, {"dtype", "float64-native"}};
    } else {
        artifact_bytes = 2 * dim * sizeof(double);
        result["amplitudes"] = {{"file", publication->sidecar("amplitudes").filename().string()}, {"count", dim},
                                {"bytes", artifact_bytes}, {"dtype", "complex128-native"}};
    }
    if (state_bytes > buffer_cap || extra_buffers > buffer_cap - state_bytes) {
        throw std::length_error("native state/readout arrays exceed their admitted byte limit");
    }
    result["readout_buffer_bytes"] = extra_buffers;
    const Index header_bytes = result.dump().size();
    // 32 bytes covers a serialized binary64 or uint64 value; four cover JSON
    // quotes, colon and comma. This is an output bound, not a peak-RSS claim.
    constexpr Index number_bytes = 32;
    if (header_bytes > output_cap || artifact_bytes > output_cap - header_bytes
        || output_items > (output_cap - header_bytes - artifact_bytes) / (key_width + number_bytes + 4)) {
        throw std::length_error("native output envelope exceeds its admitted byte limit");
    }

    // Execute exactly one admitted circuit. Readout below consumes this same
    // state, with no companion evolution or per-gate norm scan.
    NWQSim::Config::PRINT_SIM_TRACE = false;
    NWQSim::Config::ENABLE_NOISE = false;
    NWQSim::Config::ENABLE_FUSION = true;
    auto state = BackendManager::create_state(backend, width, method);
    state->set_seed(seed);
    ranks.barrier();
    if (publication) publication->started = true;
    ++native_simulations;
    state->sim(circuit);
    result["native_simulations"] = native_simulations;
    // Only the CPU/SV exact path reads host amplitudes. Other public getters
    // expose a local partition, device memory, or a density matrix.
    const double* re = exact_readouts && kind != "counts" ? state->get_real() : nullptr;
    const double* im = exact_readouts && kind != "counts" ? state->get_imag() : nullptr;
    if (kind == "counts") {
        // Project each sampled native index onto its classical bits in the
        // sampler's own outcome array, sort it and count runs, so each distinct
        // outcome is formatted once and no per-shot string is built. Projected
        // values are below 2**63 (at most 63 classical bits).
        NWQSim::IdxType* outcomes = state->measure_all(shots);
        if (ranks.rank != 0) return result;
        for (Index sample = 0; sample < shots; ++sample) {
            Index projected = 0;
            for (std::size_t bit = 0; bit < measured.size(); ++bit) {
                projected |= ((static_cast<Index>(outcomes[sample]) >> measured[bit]) & 1) << classical[bit];
            }
            outcomes[sample] = static_cast<NWQSim::IdxType>(projected);
        }
        std::sort(outcomes, outcomes + shots);
        Json histogram = Json::object();
        for (Index sample = 0; sample < shots;) {
            Index next = sample + 1;
            while (next < shots && outcomes[next] == outcomes[sample]) ++next;
            histogram[bit_string(static_cast<Index>(outcomes[sample]), classical_width)] = next - sample;
            sample = next;
        }
        result["counts"] = std::move(histogram);
    } else if (kind == "pauli") {
        result["pauli"] = Json::object();
        for (const auto& item : labels) {
            const std::string label = item;
            const double value = pauli_value(re, im, dim, label, width);
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite native Pauli readout");
            result["pauli"][label] = value;
        }
    } else if (kind == "probabilities") {
        const auto probabilities = marginal(re, im, dim, measured);
        result["probabilities"] = publication->write_sidecar("probabilities", "float64-native",
            probabilities.size(), probabilities.size() * sizeof(double), [&](std::ofstream& stream) {
                for (const auto& bin : probabilities) {
                    if (!std::isfinite(bin.value)) throw std::runtime_error("nonfinite native probability readout");
                    stream.write(reinterpret_cast<const char*>(&bin.value), sizeof(double));
                }
            });
    } else {
        // Probabilities ignore global phase, but amplitude output must restore it.
        // Stream interleaved real/imaginary pairs rather than copying the full vector.
        const std::complex<double> phase = output_phase;
        result["amplitudes"] = publication->write_sidecar("amplitudes", "complex128-native", dim,
            2 * dim * sizeof(double), [&](std::ofstream& stream) {
                for (Index x = 0; x < dim; ++x) {
                    const auto amplitude = phase * std::complex<double>(re[x], im[x]);
                    const std::array<double, 2> pair{amplitude.real(), amplitude.imag()};
                    if (!std::isfinite(pair[0]) || !std::isfinite(pair[1])) throw std::runtime_error("nonfinite native amplitude readout");
                    stream.write(reinterpret_cast<const char*>(pair.data()), sizeof(pair));
                }
            });
    }
    return result;
}

int main(int argc, char** argv) {
    if (!compiled_target()) {
        std::cerr << "selected target or its required runtime error checking is not compiled into this runner\n";
        return 2;
    }
    if (argc == 2 && std::string(argv[1]) == "--describe") {
        const Json readouts = exact_readouts ? Json{"counts", "pauli", "probabilities", "amplitudes", "trajectory"}
                                             : Json{"counts"};
        std::cout << Json{{"format", protocol}, {"nwqsim_revision", NWQLIB_NWQSIM_REVISION},
                          {"roundoff_base_revision", roundoff_base.empty() ? Json(nullptr) : Json(roundoff_base)},
                          {"roundoff_sources_identical", roundoff_sources_identical},
                          {"compiler", __VERSION__}, {"backends", {backend}}, {"methods", {method}},
                          {"architecture", architecture},
                          {"gpu_error_check", gpu_error_check},
                          {"distributed", backend == "MPI"},
                          {"per_gate_norm_check", false}, {"fusion", true}, {"noise", false},
                          {"basis_gates", {"u", "cx"}},
                          {"readouts", readouts}}.dump() << '\n';
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--phase-factor") {
        try {
            const double angle = real(Json::parse(argv[2]));
            const auto factor = std::polar(1.0, angle);
            if (!std::isfinite(factor.real()) || !std::isfinite(factor.imag())) {
                throw std::runtime_error("nonfinite output phase factor");
            }
            std::cout << Json{{"format", protocol},
                              {"phase_factor", {factor.real(), factor.imag()}}}.dump() << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 2;
        }
    }
    if (argc != 4) {
        std::cerr << "usage: nwqlib-nwqsim INPUT OUTPUT MAX_INPUT_BYTES\n";
        return 2;
    }
    Ranks ranks(argc, argv);
    const fs::path output(argv[2]);
    Index output_cap = 0;
    Index native_simulations = 0;
    Json request;
    std::unique_ptr<Publication> publication;
    try {
        if (ranks.rank == 0) {
            publication = std::make_unique<Publication>(output);
            publication->check_outputs({"amplitudes", "probabilities"});
        }
        // Nobody reads input or starts a native factory until rank zero owns
        // the exclusive claim. A rank-local error aborts the whole communicator.
        ranks.barrier();
        const std::string input_limit(argv[3]);
        if (input_limit.empty() || input_limit.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("MAX_INPUT_BYTES must be a nonnegative decimal integer");
        }
        const Index input_cap = std::stoull(input_limit);
        if (fs::file_size(argv[1]) > input_cap) throw std::length_error("native input exceeds its byte limit");
        std::ifstream stream(argv[1], std::ios::binary);
        stream >> request;
        output_cap = count(request.at("max_output_bytes"), "max_output_bytes");
        const auto result = execute(request, publication.get(), native_simulations, ranks);
        if (ranks.rank == 0) {
            write_json(*publication, result, output_cap - publication->sidecar_bytes);
            publication->terminal = true;
        }
        ranks.barrier();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (publication) publication->discard_sidecars();
        if (publication && output_cap) {
            try {
                if (!fs::exists(output)) {
                    write_json(*publication, {{"format", protocol}, {"input_id", request.value("input_id", "")},
                                    {"status", "failed"}, {"native_simulations", native_simulations},
                                    {"error", error.what()}}, output_cap);
                    publication->terminal = true;
                }
            } catch (...) {
                // Filesystem lookup and publication can both fail. Neither may
                // bypass MPI_Abort and leave peers blocked in a collective.
            }
        }
        // Run Publication cleanup before MPI_Abort, which does not unwind C++.
        // If another rank failed, rank zero keeps its claim and no unobserved
        // evolution count is invented; the caller must reconcile missing output.
        publication.reset();
        ranks.abort();
        return 1;
    }
}
