#include "sha256.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
using namespace meow;
static std::string hx(const std::array<uint8_t,32>& a) {
    static const char h[]="0123456789abcdef"; std::string s;
    for (auto x:a) { s+=h[x>>4]; s+=h[x&15]; } return s;
}
int main() {
    if (hx(sha256("",0))!="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
        hx(sha256("abc",3))!="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") return 1;
    std::mt19937 rng(2917); std::vector<uint8_t> v(131072);
    for(auto& b:v) b=uint8_t(rng());
    // Padding/block boundaries, real header/x0 sizes, arbitrary chunk boundaries,
    // zero-byte updates, and copied partial states all compare with the old engine.
    std::vector<size_t> sizes={0,1,31,32,55,56,63,64,65,80,112,127,128,129,32768,32864,131072};
    for(int i=0;i<500;++i) sizes.push_back(rng()%v.size());
    for(size_t n:sizes) {
        auto ref=PortableSha256::hash(v.data(),n);
        if(sha256(v.data(),n)!=ref) return 2;
        Sha256 s; size_t pos=0;
        while(pos<n) { size_t len=std::min(n-pos,size_t(1+rng()%1000)); s.update(v.data()+pos,len); s.update(v.data(),0); pos+=len; }
        Sha256 copy=s, assigned; assigned=s;
        if(s.digest()!=ref || copy.digest()!=ref || assigned.digest()!=ref) return 3;
        for(const char* tag:{"POI/seed","POI/out","POI/model"}) {
            auto t=PortableSha256::hash(tag,std::strlen(tag)); PortableSha256 p;
            p.update(t.data(),32);p.update(t.data(),32);p.update(v.data(),n);
            TaggedHasher q(tag); q.update(v.data(),n); if(q.digest()!=p.digest())return 4;
        }
    }
    unsigned sink=0;
    auto bench=[&](bool portable) {
        auto t=std::chrono::steady_clock::now();
        for(int i=0;i<10000;++i) {v[0]=uint8_t(i);auto h=portable?PortableSha256::hash(v.data(),32768):sha256(v.data(),32768);sink+=h[0];}
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
    };
    const double old=bench(true), now=bench(false);
    std::printf("SHA256 PASS: %zu inputs, tagged hashes, split updates and copies; 32KiB old %.3fs accelerated %.3fs speedup %.2fx (sink=%u)\n",sizes.size(),old,now,old/now,sink);
    return 0;
}
