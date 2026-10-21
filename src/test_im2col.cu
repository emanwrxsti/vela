// Independent integer oracle for image-patch quantization, including padding.
#include "kernels.h"
#include "common.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <random>

int main() {
    try {
        std::mt19937 rng(20261021);
        int cases=0;
        for (int C : {4, 12, 16, 32, 320, 640, 960, 1280, 1920, 2560}) {
            for (int stride : {1, 2}) for (int mode : {0, 1, 2}) {
                const int B=2, H=5, W=7, Ho=(H-1)/stride+1, Wo=(W-1)/stride+1;
                const int rows=B*Ho*Wo, Mp=(rows+31)/32*32, Kp=(9*C+15)/16*16;
                std::vector<int32_t> x(size_t(B)*H*W*C);
                for (size_t i=0;i<x.size();++i) {
                    x[i]=mode==0 ? 0 : mode==1 ? int32_t(rng()%65537)-32768 : int32_t(rng());
                    if(mode==2 && i%19==0) x[i]=(i&1)?INT_MIN:INT_MAX;
                }
                std::vector<int8_t> expectedHi(size_t(Mp)*Kp,0), expectedLo(size_t(Mp)*Kp,-128);
                std::vector<int32_t> expectedE(rows);
                for(int r=0;r<rows;++r) {
                    const int b=r/(Ho*Wo), p=r%(Ho*Wo), oy=p/Wo, ox=p%Wo;
                    int64_t maxAbs=0;
                    for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
                        const int iy=oy*stride-1+ky, ix=ox*stride-1+kx;
                        if(iy<0||iy>=H||ix<0||ix>=W) continue;
                        for(int c=0;c<C;++c) {
                            int64_t v=x[((size_t(b)*H+iy)*W+ix)*C+c];
                            maxAbs=std::max(maxAbs,v<0?-v:v);
                        }
                    }
                    int e=0; while(e<17 && maxAbs >= (int64_t(32768)<<e)) ++e;
                    expectedE[r]=e;
                    for(int tap=0;tap<9;++tap) for(int c=0;c<C;++c) {
                        const int iy=oy*stride-1+tap/3, ix=ox*stride-1+tap%3;
                        int64_t v=(iy<0||iy>=H||ix<0||ix>=W)?0:x[((size_t(b)*H+iy)*W+ix)*C+c];
                        int64_t q=e ? (v+(int64_t(1)<<(e-1)))>>e : v;
                        q=std::clamp(q,int64_t(-32767),int64_t(32767));
                        size_t at=size_t(r)*Kp+tap*C+c;
                        expectedHi[at]=int8_t(q>>8); expectedLo[at]=int8_t((q&255)-128);
                    }
                }
                int32_t *dx=nullptr,*de=nullptr; uint32_t* pm=nullptr; int8_t *hi=nullptr,*lo=nullptr;
                CUDA_CHECK(cudaMalloc(&dx,x.size()*4)); CUDA_CHECK(cudaMalloc(&de,rows*4));
                CUDA_CHECK(cudaMalloc(&pm,B*H*W*4)); CUDA_CHECK(cudaMalloc(&hi,expectedHi.size())); CUDA_CHECK(cudaMalloc(&lo,expectedLo.size()));
                CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(hi,0x5a,expectedHi.size())); CUDA_CHECK(cudaMemset(lo,0x5a,expectedLo.size()));
                meow::k_pixmax(0,dx,B*H*W,C,pm);
                meow::k_im2col3(0,dx,pm,B,H,W,C,stride,hi,lo,de,Kp);
                CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaGetLastError());
                std::vector<int8_t> gotHi(expectedHi.size()),gotLo(expectedLo.size());std::vector<int32_t> gotE(rows);
                CUDA_CHECK(cudaMemcpy(gotHi.data(),hi,gotHi.size(),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gotLo.data(),lo,gotLo.size(),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gotE.data(),de,rows*4,cudaMemcpyDeviceToHost));
                if(gotHi!=expectedHi||gotLo!=expectedLo||gotE!=expectedE) throw std::runtime_error("im2col mismatch C="+std::to_string(C)+" stride="+std::to_string(stride)+" mode="+std::to_string(mode));
                CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(de)); CUDA_CHECK(cudaFree(pm)); CUDA_CHECK(cudaFree(hi)); CUDA_CHECK(cudaFree(lo)); ++cases;
            }
        }
        std::cout<<"PASS "<<cases<<" independent im2col cases (all output bytes, scales and padded rows)"<<std::endl;
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
