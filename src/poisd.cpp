#include "poisd.h"
#include "sha256.h"
#include "common.h"
#include <cstring>
#include <fstream>

namespace meow {
static uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }

bool Poisd::load(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { error = "cannot open " + path; return false; }
    f.seekg(0, std::ios::end); size_t size = size_t(f.tellg()); f.seekg(0);
    data_.resize(size);
    if (!f.read(reinterpret_cast<char*>(data_.data()), size)) { error = "read error"; return false; }
    static const uint8_t MAGIC[8]{'P', 'O', 'I', 'S', 'D', 'v', '1', 0};
    if (size < 12 || std::memcmp(data_.data(), MAGIC, 8) != 0) { error = "bad magic (not a POISDv1 file)"; return false; }
    size_t pos = 8;
    auto need = [&](size_t n) { if (pos + n > data_.size()) throw std::runtime_error("truncated model file"); };
    try {
        need(4); uint32_t count = rd32(&data_[pos]); pos += 4;
        for (uint32_t i = 0; i < count; ++i) {
            need(2); uint16_t nl = rd16(&data_[pos]); pos += 2;
            need(nl + 2); std::string name(reinterpret_cast<const char*>(&data_[pos]), nl); pos += nl;
            PoisdTensor e; e.dtype = data_[pos]; uint8_t nd = data_[pos + 1]; pos += 2;
            if (e.dtype > 3) throw std::runtime_error("bad dtype");
            need(4 * nd); e.count = 1;
            for (int d = 0; d < nd; ++d) { e.dims.push_back(rd32(&data_[pos])); e.count *= e.dims.back(); pos += 4; }
            need(e.bytes()); e.off = pos; pos += e.bytes();
            entries_.emplace(std::move(name), std::move(e));
        }
        if (pos != data_.size()) throw std::runtime_error("trailing data");
        for (const char* n : {"unet.conv_in.w", "unet.conv_out.w", "te.tok.w", "te.pos", "lut.gauss", "lut.exp", "lut.sigmoid", "lut.phi", "c.sig", "c.cin", "tok.vocab", "tok.merges"})
            if (!has(n)) throw std::runtime_error(std::string("missing tensor ") + n);
    } catch (const std::exception& e) { error = e.what(); return false; }
    TaggedHasher h("POI/model"); uint8_t v = 2; h.update(&v, 1); h.update(data_.data(), data_.size());
    auto d = h.digest();
    // uint256 display order is the byte-reversed digest.
    for (int i = 0; i < 32; ++i) mh_u256_[i] = d[i];
    std::array<uint8_t, 32> nat; for (int i = 0; i < 32; ++i) nat[i] = d[31 - i];
    mh_natural_ = hex(nat.data(), 32);
    return true;
}
const PoisdTensor& Poisd::get(const std::string& n) const {
    auto it = entries_.find(n); if (it == entries_.end()) throw std::runtime_error("missing tensor " + n); return it->second;
}
const int8_t* Poisd::i8(const std::string& n) const { const auto& e = get(n); if (e.dtype != 0) throw std::runtime_error(n + " is not int8"); return reinterpret_cast<const int8_t*>(data_.data() + e.off); }
std::vector<int64_t> Poisd::i32v(const std::string& n) const {
    const auto& e = get(n); if (e.dtype != 2) throw std::runtime_error(n + " is not int32");
    std::vector<int64_t> v(e.count); for (size_t i = 0; i < e.count; ++i) v[i] = int32_t(rd32(data_.data() + e.off + 4 * i)); return v;
}
std::string Poisd::blob(const std::string& n) const { const auto& e = get(n); return std::string(reinterpret_cast<const char*>(data_.data() + e.off), e.bytes()); }
}  // namespace meow
