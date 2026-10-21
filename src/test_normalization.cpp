// Independent integer reference for layer normalization and quantized output.
#include "kernels.h"
#include "common.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <random>

using I = int64_t;
static I divideFloor(I a, I b) { I q=a/b; return q-(a%b<0); }
static I saturate(I x) { return std::clamp(x,I(INT_MIN),I(INT_MAX)); }
static I rounded(I x,int shift) { return saturate(shift ? divideFloor(x+(I(1)<<(shift-1)),I(1)<<shift) : x); }
static int exponent(I x) { int e=0;for(int k=15;k<32;++k)e+=x>=(I(1)<<k);return e; }
static I root(I x) { I s=I(std::sqrt(static_cast<long double>(x)));while(s*s>x)--s;while((s+1)*(s+1)<=x)++s;return s; }

// Exercise groupnorm application independently, using supplied valid statistics.
static int testGroupNorm() {
    std::mt19937 rng(20261022);int cases=0;
    for(int C:{32,64,128,320,512,640,960,1024,1280,1920,2560})
    for(int mode:{0,1,2}) for(bool activation:{false,true}) for(bool quantized:{false,true}) {
        constexpr int B=2,T=5,M=B*T,groups=32;const int Kp=(C+15)/16*16;
        std::vector<int32_t> x(M*C),g(C),beta(C),lut(2049);
        std::vector<meow::GnStat> stats(B*groups);
        for(auto& v:x)v=mode==0?0:mode==1?int32_t(rng()%2000001)-1000000:int32_t(rng());
        if(mode==2){x[0]=INT_MIN;x[1]=INT_MAX;}
        for(int c=0;c<C;++c){g[c]=mode==2?int32_t(rng()):65536;beta[c]=mode==2?int32_t(rng()):0;}
        for(int i=0;i<=2048;++i)lut[i]=i*32;
        for(auto& st:stats){st.mean=mode==0?0:mode==1?int32_t(rng()%1000001)-500000:int32_t(rng());st.e=mode==2?12:0;st.sd=65536+(rng()%100000000);}
        std::vector<int32_t> ey(M*C+16,0x5a5a5a5a),ee(M+8,0x5a5a5a5a);
        std::vector<uint32_t> epm(M+8,0x5a5a5a5a);
        std::vector<int8_t> eh(M*Kp+64,0x5a),el(M*Kp+64,0x5a);
        for(int row=0;row<M;++row){
            std::vector<I> y(C);I maxAbs=0;
            for(int c=0;c<C;++c){
                const auto& st=stats[(row/T)*groups+c/(C/groups)];
                I a=(I(x[row*C+c])-st.mean)*(I(1)<<(24-st.e))*2+st.sd;
                if(a<=-(I(1)<<52)||a>=(I(1)<<52))throw std::runtime_error("invalid oracle numerator");
                I norm=divideFloor(a,2*st.sd);
                y[c]=saturate(rounded(norm*g[c],16)+beta[c]);
                if(activation){
                    I z=y[c]+(16LL<<16),v=0;
                    if(z>=(32LL<<16))v=65536;
                    else if(z>=0){I at=z>>10,f=z&1023;v=lut[at]+((I(lut[at+1]-lut[at])*f)>>10);}
                    y[c]=rounded(y[c]*v,16);
                }
                maxAbs=std::max(maxAbs,y[c]<0?-y[c]:y[c]);
                if(!quantized)ey[row*C+c]=int32_t(y[c]);
            }
            if(quantized){
                ee[row]=exponent(maxAbs);
                for(int c=0;c<Kp;++c){I q=c<C?std::clamp(rounded(y[c],ee[row]),I(-32767),I(32767)):0;eh[row*Kp+c]=int8_t(q>>8);el[row*Kp+c]=int8_t((q&255)-128);}
            }else epm[row]=uint32_t(maxAbs);
        }
        int32_t *dx=nullptr,*dg=nullptr,*db=nullptr,*dL=nullptr,*dy=nullptr,*de=nullptr;
        uint32_t* dpm=nullptr;int8_t *dh=nullptr,*dl=nullptr;meow::GnStat* ds=nullptr;
        CUDA_CHECK(cudaMalloc(&dx,x.size()*4));CUDA_CHECK(cudaMalloc(&dg,g.size()*4));CUDA_CHECK(cudaMalloc(&db,beta.size()*4));CUDA_CHECK(cudaMalloc(&dL,lut.size()*4));CUDA_CHECK(cudaMalloc(&ds,stats.size()*sizeof(meow::GnStat)));
        CUDA_CHECK(cudaMalloc(&dy,ey.size()*4));CUDA_CHECK(cudaMalloc(&de,ee.size()*4));CUDA_CHECK(cudaMalloc(&dpm,epm.size()*4));CUDA_CHECK(cudaMalloc(&dh,eh.size()));CUDA_CHECK(cudaMalloc(&dl,el.size()));
        CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dg,g.data(),g.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,beta.data(),beta.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dL,lut.data(),lut.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(ds,stats.data(),stats.size()*sizeof(meow::GnStat),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dy,0x5a,ey.size()*4));CUDA_CHECK(cudaMemset(de,0x5a,ee.size()*4));CUDA_CHECK(cudaMemset(dpm,0x5a,epm.size()*4));CUDA_CHECK(cudaMemset(dh,0x5a,eh.size()));CUDA_CHECK(cudaMemset(dl,0x5a,el.size()));
        meow::k_gn_apply_row(0,dx,B,T,C,groups,ds,dg,db,activation?dL:nullptr,dy,dpm,quantized?dh:nullptr,dl,de,Kp);
        CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaGetLastError());
        std::vector<int32_t> y(ey.size()),e(ee.size());std::vector<uint32_t> pm(epm.size());std::vector<int8_t> h(eh.size()),l(el.size());
        CUDA_CHECK(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(e.data(),de,e.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(pm.data(),dpm,pm.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(h.data(),dh,h.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(l.data(),dl,l.size(),cudaMemcpyDeviceToHost));
        if(y!=ey||e!=ee||pm!=epm||h!=eh||l!=el)throw std::runtime_error("groupnorm mismatch C="+std::to_string(C)+" mode="+std::to_string(mode)+" activation="+std::to_string(activation)+" quantized="+std::to_string(quantized));
        CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dg));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(dL));CUDA_CHECK(cudaFree(ds));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(de));CUDA_CHECK(cudaFree(dpm));CUDA_CHECK(cudaFree(dh));CUDA_CHECK(cudaFree(dl));++cases;
    }
    return cases;
}

int main() {
    try {
        std::mt19937 rng(20261021);int cases=0;
        for(int C:{4,12,124,128,132,320,512,640,768,1024,1280}) for(int mode:{0,1,2}) {
            constexpr int M=9;constexpr I den=100000;const int Kp=(C+15)/16*16;
            std::vector<int32_t> x(M*C),g(C),beta(C);
            for(auto& v:x)v=mode==0?0:mode==1?int32_t(rng()%2000001)-1000000:int32_t(rng());
            if(mode==2){x[0]=INT_MIN;x[1]=INT_MAX;}
            for(int c=0;c<C;++c){g[c]=mode==2?int32_t(rng()):65536;beta[c]=mode==2?int32_t(rng()):0;}
            std::vector<int8_t> eh(M*Kp+64,0x5a),el(M*Kp+64,0x5a);
            std::vector<int32_t> ee(M+8,0x5a5a5a5a);
            for(int row=0;row<M;++row){
                I sum=0,mn=INT_MAX,mx=INT_MIN;
                for(int c=0;c<C;++c){I v=x[row*C+c];sum+=v;mn=std::min(mn,v);mx=std::max(mx,v);}
                I mean=divideFloor(sum+C/2,C);int e=std::max(0,exponent(std::max(mx-mean,mean-mn))-5);
                I ss=0;for(int c=0;c<C;++c){I d=rounded(I(x[row*C+c])-mean,e);ss+=d*d;}
                I one=32-2*e>=0?I(1)<<(32-2*e):0;
                I sd=root((ss/C+(one+den/2)/den)*65536);
                std::vector<I> y(C);I maxAbs=0;
                for(int c=0;c<C;++c){
                    I d=I(x[row*C+c])-mean;
                    I norm=divideFloor(d*(I(1)<<(24-e))*2+sd,2*sd);
                    y[c]=saturate(rounded(norm*g[c],16)+beta[c]);maxAbs=std::max(maxAbs,y[c]<0?-y[c]:y[c]);
                }
                ee[row]=exponent(maxAbs);
                for(int c=0;c<Kp;++c){
                    I q=c<C?std::clamp(rounded(y[c],ee[row]),I(-32767),I(32767)):0;
                    eh[row*Kp+c]=int8_t(q>>8);el[row*Kp+c]=int8_t((q&255)-128);
                }
            }
            int32_t *dx=nullptr,*dg=nullptr,*db=nullptr,*de=nullptr;int8_t *dh=nullptr,*dl=nullptr;
            CUDA_CHECK(cudaMalloc(&dx,x.size()*4));CUDA_CHECK(cudaMalloc(&dg,g.size()*4));CUDA_CHECK(cudaMalloc(&db,beta.size()*4));
            CUDA_CHECK(cudaMalloc(&de,ee.size()*4));CUDA_CHECK(cudaMalloc(&dh,eh.size()));CUDA_CHECK(cudaMalloc(&dl,el.size()));
            CUDA_CHECK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dg,g.data(),g.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,beta.data(),beta.size()*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(de,0x5a,ee.size()*4));CUDA_CHECK(cudaMemset(dh,0x5a,eh.size()));CUDA_CHECK(cudaMemset(dl,0x5a,el.size()));
            meow::k_lnorm_q(0,dx,M,C,dg,db,den,dh,dl,de,Kp);
            CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaGetLastError());
            std::vector<int8_t> h(eh.size()),l(el.size());std::vector<int32_t> e(ee.size());
            CUDA_CHECK(cudaMemcpy(h.data(),dh,h.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(l.data(),dl,l.size(),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(e.data(),de,e.size()*4,cudaMemcpyDeviceToHost));
            if(h!=eh||l!=el||e!=ee)throw std::runtime_error("layernorm mismatch C="+std::to_string(C)+" mode="+std::to_string(mode));
            CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dg));CUDA_CHECK(cudaFree(db));CUDA_CHECK(cudaFree(de));CUDA_CHECK(cudaFree(dh));CUDA_CHECK(cudaFree(dl));++cases;
        }
        std::cout<<"PASS "<<cases<<" independent layernorm cases, all quantized bytes/scales, padding and guards"<<std::endl;
        std::cout<<"PASS "<<testGroupNorm()<<" independent groupnorm application cases, activation/quantization modes, padding and guards"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
