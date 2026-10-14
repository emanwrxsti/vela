// POISDv1 model file reader (contrib/poi-sd/sdint.py "Model file" section).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <array>

namespace meow {
struct PoisdTensor {
    uint8_t dtype;               // 0 i8, 1 i16, 2 i32, 3 u8
    std::vector<uint32_t> dims;
    size_t off;                  // byte offset into file data
    size_t count;                // element count
    size_t bytes() const { static const size_t SZ[4]{1, 2, 4, 1}; return count * SZ[dtype]; }
};
class Poisd {
public:
    bool load(const std::string& path, std::string& error);
    const std::vector<uint8_t>& data() const { return data_; }
    const std::map<std::string, PoisdTensor>& entries() const { return entries_; }
    bool has(const std::string& n) const { return entries_.count(n) != 0; }
    const PoisdTensor& get(const std::string& n) const;
    const int8_t* i8(const std::string& n) const;
    std::vector<int64_t> i32v(const std::string& n) const;     // int32 tensor widened to int64
    int64_t i32s(const std::string& n) const { return i32v(n).at(0); }
    std::string blob(const std::string& n) const;
    /** model_hash = TaggedHash("POI/model")(u8 version || file bytes); natural (display) order hex. */
    std::string model_hash_natural() const { return mh_natural_; }
    std::array<uint8_t, 32> model_hash_uint256() const { return mh_u256_; }   // byte-reversed (what the seed uses)
private:
    std::vector<uint8_t> data_;
    std::map<std::string, PoisdTensor> entries_;
    std::string mh_natural_; std::array<uint8_t, 32> mh_u256_{};
};
}  // namespace meow
