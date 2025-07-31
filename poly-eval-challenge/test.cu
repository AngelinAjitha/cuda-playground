#include <iostream>

#include "poly_eval.cu"

#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#define P_MOD 32843

template <unsigned P = P_MOD>
class Dummy_Scalar
{
public:
  static constexpr unsigned MODULUS = P;

  unsigned x;

  static HOST_DEVICE_INLINE Dummy_Scalar zero() { return {0}; }

  static HOST_DEVICE_INLINE Dummy_Scalar one() { return {1}; }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar)
  {
    os << scalar.x;
    return os;
  }

  friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x + p2.x) % MODULUS};
  }

  friend HOST_DEVICE_INLINE Dummy_Scalar operator*(Dummy_Scalar p1, const Dummy_Scalar& p2)
  {
    return {(p1.x * p2.x) % MODULUS};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) 
  { 
    return (p1.x == p2.x); 
  }

  friend HOST_DEVICE_INLINE bool operator!=(const Dummy_Scalar& p1, const Dummy_Scalar& p2) 
  { 
    return (p1.x != p2.x); 
  }

  friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) 
  { 
    return (p1.x == p2); 
  }

  static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar& scalar) 
  { 
    return {MODULUS - scalar.x}; 
  }

  static HOST_INLINE Dummy_Scalar rand_host()
  {
    return {static_cast<unsigned>(rand() % MODULUS)};
  }
};

typedef Dummy_Scalar<> test_scalar; 

int main(int argc, char** argv)
{
  cudaEvent_t start, stop;
  float time;

  // test parameters
  int coeffs_log_size = (argc > 1) ? atoi(argv[1]) : 22;
  int coeffs_size = 1 << coeffs_log_size;
  int domain_size = (argc > 2) ? atoi(argv[2]) : 7;
  int batch_size = (argc > 3) ? atoi(argv[3]) : 10;
  int total_coeffs_size = batch_size * coeffs_size;
  int total_results_size = batch_size * domain_size;

  printf("running poly eval of degree, 2^%d, domain_size=%d, batch_size=%d, scalar modulus=%d\n", coeffs_log_size, domain_size, batch_size, P_MOD);

  // init inputs
  test_scalar* coeffs = new test_scalar[total_coeffs_size];
  test_scalar* domain = new test_scalar[domain_size];
  for (int i = 0; i < total_coeffs_size; i++)
  {
    coeffs[i] = test_scalar::rand_host();
  }
  for (int i = 0; i < domain_size; i++)
  {
    domain[i] = test_scalar::rand_host();
  }

  std::cout << "finished generating inputs" << std::endl;

  test_scalar* results = new test_scalar[total_results_size];
  test_scalar* ref_results = new test_scalar[total_results_size];
 
  // allocate memory and copy to device
  test_scalar* d_coeffs;
  test_scalar* d_domain;
  test_scalar* d_results;
  cudaMalloc(&d_coeffs, sizeof(test_scalar) * total_coeffs_size);
  cudaMalloc(&d_domain, sizeof(test_scalar) * domain_size);
  cudaMalloc(&d_results, sizeof(test_scalar) * total_results_size);
  cudaMemcpy(d_coeffs, coeffs, sizeof(test_scalar) * total_coeffs_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_domain, domain, sizeof(test_scalar) * domain_size, cudaMemcpyHostToDevice);

  std::cout << "finished copying to device" << std::endl;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warm up
  poly_eval_horner(d_coeffs, d_domain, coeffs_size, domain_size, batch_size, d_results);
  cudaDeviceSynchronize();

  cudaEventRecord(start, 0);
  poly_eval_horner(d_coeffs, d_domain, coeffs_size, domain_size, batch_size, d_results);
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&time, start, stop);
  printf("runtime : %.3f ms.\n", time);

  // run reference and check correctness
  poly_eval_ref(coeffs, domain, coeffs_size, domain_size, batch_size, ref_results);

  cudaMemcpy(results, d_results, sizeof(test_scalar) * total_results_size, cudaMemcpyDeviceToHost);

  bool success = true;
  for (unsigned i = 0; i < total_results_size; i++) {
    if (results[i] != ref_results[i]) {
      success = false;
    }
  }
  if (success) { 
    std::cout << "success!" << std::endl;
  }
  else {
    std::cout << "fail!" << std::endl;
  }

  return 0;
}
