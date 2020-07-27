#define LAMBDA_TYPE __device__ __host__

template<long long i, typename func_t>
__global__ void elementwise_kernel(float *a, float *b, float *c, int N, func_t f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = f(a[idx], b[idx]);
    }
}

template<long long l = 0, long long r = 128>
void _add(float *a, float *b, float *c, float alpha, int N) {
    if constexpr (l == r) {
        elementwise_kernel<l><<<(N + 255) / 256, 256>>>(a, b, c, N,
            [=] LAMBDA_TYPE (float a, float b) { return a + alpha * b; }
        );
    } else if constexpr (l + 1 == r) {
        elementwise_kernel<l><<<(N + 255) / 256, 256>>>(a, b, c, N,
            [=] LAMBDA_TYPE (float a, float b) { return a + b; }
        );
        elementwise_kernel<r><<<(N + 255) / 256, 256>>>(a, b, c, N,
            [=] LAMBDA_TYPE (float a, float b) { return a + b; }
        );
    } else {
        constexpr long long m = (l + r) / 2;
        _add<l, m>(a, b, c, alpha, N);
        _add<m, r>(a, b, c, alpha, N);
    }
}

void add(float *a, float *b, float *c, int N) {
    _add(a, b, c, 1.23, N);
}
