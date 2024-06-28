#include <stdio.h>
#include <assert.h>
#define MIN_VALUE (-1e38)
#define CHANNEL_SPLIT (512 / 32)
#define EPS (1e-6)
#define TOKEN_SPLIT (512 / CHANNEL_SPLIT) // the number of split tokens
#define IDEAL_T_LEN (Tmax / TOKEN_SPLIT)


template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_id = threadIdx.x;
    const int token_id = threadIdx.y;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _tokenLength = min(T - _t, _T);
    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;
    // for saving smem, del Sc, Sd, So1
    __shared__ F Sa[CHANNEL_SPLIT][TOKEN_SPLIT], Sb[CHANNEL_SPLIT][TOKEN_SPLIT],
        So2[CHANNEL_SPLIT][TOKEN_SPLIT];
    F a = 0, b = 0, c = 0, d = 0;
    F o1 = MIN_VALUE, o2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        F no = max(o1, k[ii] - w * (i - _t));
        F e1 = exp(o1 - no);
        F e3 = exp(k[ii] - w * (i - _t) - no);
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = max(o2, k[nini] - w * exp_w);
        F e2 = exp(o2 - no);
        e3 = exp(k[nini] - w * exp_w - no);
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }

    So2[channel_id][token_id] = o2;
    Sa[channel_id][token_id] = a;
    Sb[channel_id][token_id] = b;
    __syncthreads();
    a = 0;
    b = 0;
    o2 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        F no = max(So2[channel_id][i] - w * exp_w, o2);
        a = a * exp(o2 - no) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        b = b * exp(o2 - no) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        o2 = no;
    }
    __syncthreads();
    Sa[channel_id][token_id] = c;
    Sb[channel_id][token_id] = d;
    So2[channel_id][token_id] = o1;
    __syncthreads();
    c = 0;
    d = 0;
    o1 = MIN_VALUE;
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        F no = max(So2[channel_id][i] - w * exp_w, o1);
        c = c * exp(o1 - no) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        d = d * exp(o1 - no) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        o1 = no;
    }
    c -= exp(k[_t * C] - o1) * v[_t * C];
    d -= exp(k[_t * C] - o1);
    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        F no = max(o1, u + k[ii]);
        no = max(no, o2);
        F e1 = exp(o1 - no);
        F e2 = exp(o2 - no);
        F e3 = exp(u + k[ii] - no);
        y[ii] = (c * e1 + a * e2 + e3 * v[ii])/(d * e1 + b * e2 + e3 + EPS);
        // update a, b, c, d
        const int ii2 = ((i + 1) % T) * C;
        no = max(o2 - w, k[ii]);
        e2 = exp(o2 - w - no);
        e3 = exp(k[ii] - no);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;
        no = max(o1 + w, k[ii2] + w);
        e1 = exp(o1 + w - no);
        e3 = exp(k[ii2] + w - no);
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_id = threadIdx.x;
    const int token_id = threadIdx.y;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _tokenLength = min(T - _t, _T);
    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[IDEAL_T_LEN], z[IDEAL_T_LEN], zexp[IDEAL_T_LEN];
    // for saving smem, del Sc, Sd, Sdcdw, Sdddw, So1
    __shared__ F Sa[CHANNEL_SPLIT][TOKEN_SPLIT], Sb[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ F Sdadw[CHANNEL_SPLIT][TOKEN_SPLIT], Sdbdw[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ F So2[CHANNEL_SPLIT][TOKEN_SPLIT];
    F a = 0, b = 0, c = 0, d = 0;
    F dadw = 0, dbdw = 0, dcdw = 0, dddw = 0;
    F o1 = MIN_VALUE, o2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        F no = max(o1, k[ii] - w * (i - _t));
        F e1 = exp(o1 - no);
        F e3 = exp(k[ii] - w * (i - _t) - no);
        dcdw = dcdw * e1 - (i - _t) * e3 * v[ii];
        dddw = dddw * e1 - (i - _t) * e3;
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = max(o2, k[nini] - w * exp_w);
        F e2 = exp(o2 - no);
        e3 = exp(k[nini] - w * exp_w - no);
        dadw = dadw * e2 - exp_w * e3 * v[nini];
        dbdw = dbdw * e2 - exp_w * e3;
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }
    __syncthreads();
    So2[channel_id][token_id] = o2;
    Sa[channel_id][token_id] = a;
    Sb[channel_id][token_id] = b;
    Sdadw[channel_id][token_id] = dadw;
    Sdbdw[channel_id][token_id] = dbdw;
    __syncthreads();
    a = 0;
    b = 0;
    dadw = 0;
    dbdw = 0;
    o2 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        F no = max(So2[channel_id][i] - w * exp_w, o2);
        a = a * exp(o2 - no) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        b = b * exp(o2 - no) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        dadw = dadw * exp(o2 - no) + (Sdadw[channel_id][i] - exp_w * Sa[channel_id][i])
            * exp(So2[channel_id][i] - w * exp_w - no);
        dbdw = dbdw * exp(o2 - no) + (Sdbdw[channel_id][i] - exp_w * Sb[channel_id][i])
            * exp(So2[channel_id][i] - w * exp_w - no);
        o2 = no;
    }
    __syncthreads();
    So2[channel_id][token_id] = o1;
    Sa[channel_id][token_id] = c;
    Sb[channel_id][token_id] = d;
    Sdadw[channel_id][token_id] = dcdw;
    Sdbdw[channel_id][token_id] = dddw;
    __syncthreads();
    c = 0;
    d = 0;
    dcdw = 0;
    dddw = 0;
    o1 = MIN_VALUE;
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        F no = max(So2[channel_id][i] - w * exp_w, o1);
        c = c * exp(o1 - no) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        d = d * exp(o1 - no) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - no);
        dcdw = dcdw * exp(o1 - no) + (Sdadw[channel_id][i] - exp_w * Sa[channel_id][i])
             * exp(So2[channel_id][i] - w * exp_w - no);
        dddw = dddw * exp(o1 - no) + (Sdbdw[channel_id][i] - exp_w * Sb[channel_id][i])
             * exp(So2[channel_id][i] - w * exp_w - no);
        o1 = no;
    }
    c -= exp(k[_t * C] - o1) * v[_t * C];
    d -= exp(k[_t * C] - o1);

    F gw = 0, gu = 0;
    F gc = 0, gd = 0, ga = 0, gb = 0;
    F go1 = MIN_VALUE, go2 = MIN_VALUE;
    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        F no = max(o1, u + k[ii]);
        no = max(no, o2);
        F e1 = exp(o1 - no);
        F e2 = exp(o2 - no);
        F e3 = exp(u + k[ii] - no);
        F num = (c * e1 + a * e2 + e3 * v[ii]);
        F iden = 1 / (d * e1 + b * e2 + e3 + EPS);
        y[i - _t] = num * iden;
        z[i - _t] = iden;
        zexp[i - _t] = -no;
        gw += gy[ii] * (dadw - dbdw * y[i - _t]) * iden * e2;
        gw += gy[ii] * (dcdw - dddw * y[i - _t]) * iden * e1;
        gu += gy[ii] * (v[ii] - y[i - _t]) * e3 * iden;
        gk[ii] = gy[ii] * iden * (v[ii] - y[i - _t]) * e3;
        gv[ii] = gy[ii] * iden * e3;
        // cal gc & gd for gk & gv
        F gno = max(- w + go1, -no);
        e1 = exp(- w + go1 - gno);
        e3 = gy[ii] * iden  * exp(- no - gno);
        gc = e1 * gc + e3 * y[i - _t];
        gd = e1 * gd + e3;
        go1 = gno;

        // update a, b, c, d
        const int ii2 = ((i + 1) % T) * C;
        no = max(o2 - w, k[ii]);
        e2 = exp(o2 - w - no);
        e3 = exp(k[ii] - no);
        dadw = e2 * (dadw - a);
        dbdw = e2 * (dbdw - b);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;
        no = max(o1 + w, k[ii2] + w);
        e1 = exp(o1 + w - no);
        e3 = exp(k[ii2] + w - no);
        dcdw = e1 * (c + dcdw) - e3 * v[ii2];
        dddw = e1 * (d + dddw) - e3;
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }
    __syncthreads();
    Sdadw[channel_id][token_id] = gw;
    Sdbdw[channel_id][token_id] = gu;
    __syncthreads();
    if(token_id == 0){
        const int _offsetBC = _b * C + _c;
        for(int i = 0; i < TOKEN_SPLIT; i++){
            _gw[_offsetBC] += Sdadw[channel_id][i];
            _gu[_offsetBC] += Sdbdw[channel_id][i];
        }
    }
    __syncthreads();
    for (int i = _t + _tokenLength - 1; i >=_t ; i--) {
        const int ii = i * C;
        F gno = max(-w + go2, zexp[i - _t]);
        F e2 = exp(-w + go2 - gno);
        F e3 = gy[ii] * z[i - _t] * exp(zexp[i - _t] - gno);
        ga = e2 * ga + e3 * y[i - _t];
        gb = e2 * gb + e3;
        go2 = gno;
    }
    __syncthreads();
    Sa[channel_id][token_id] = gc;
    Sb[channel_id][token_id] = gd;
    So2[channel_id][token_id] = go1;
    __syncthreads();
    gc = 0;
    gd = 0;
    go1 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        F gno = max(So2[channel_id][i] - w * exp_w, go1);
        gc = gc * exp(go1 - gno) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - gno);
        gd = gd * exp(go1 - gno) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - gno);
        go1 = gno;
    }

    __syncthreads();
    Sa[channel_id][token_id] = ga;
    Sb[channel_id][token_id] = gb;
    So2[channel_id][token_id] = go2;
    __syncthreads();
    ga = 0;
    gb = 0;
    go2 = MIN_VALUE;
    for (int i = token_id + 1; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id - 1) * _T;
        F gno = max(So2[channel_id][i] - w * exp_w, go2);
        ga = ga * exp(go2 - gno) + Sa[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - gno);
        gb = gb * exp(go2 - gno) + Sb[channel_id][i] * exp(So2[channel_id][i] - w * exp_w - gno);
        go2 = gno;
    }

    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        gk[ii] += exp(k[ii] + go1) * (gd * v[ii] - gc);
        gk[nini] += exp(k[nini] + go2) * (gb * v[nini] - ga);
        gv[ii] += exp(k[ii] + go1) * gd;
        gv[nini] += exp(k[nini] + go2) * gb;
        F gno = max(-w + go1, zexp[i - _t]);
        F e1 = exp(-w + go1 - gno);
        F e3 = gy[ii] * z[i - _t]  * exp(zexp[i - _t] - gno);
        gc = e1 * gc + e3 * y[i - _t];
        gd = e1 * gd + e3;
        go1 = gno;
        gno = max(-w + go2, zexp[ni - _t]);
        F e2 = exp(-w + go2 - gno);
        e3 = gy[nini] * z[ni - _t] * exp(zexp[ni - _t] - gno);
        ga = e2 * ga + e3 * y[ni - _t];
        gb = e2 * gb + e3;
        go2 = gno;
    }
}

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
    // 1024 threads per plock
    dim3 threadsPerBlock(min(CHANNEL_SPLIT, C), TOKEN_SPLIT); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv) {
    dim3 threadsPerBlock(min(CHANNEL_SPLIT, C), TOKEN_SPLIT); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
