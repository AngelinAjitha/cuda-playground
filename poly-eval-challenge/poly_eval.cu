template <typename T>
void poly_eval_ref(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  // using Horner's method
  // example: ax^2+bx+c is computed as (1) r=a, (2) r=r*x+b, (3) r=r*x+c
  for (uint64_t idx_in_batch = 0; idx_in_batch < batch_size; ++idx_in_batch) {
    const T* curr_coeffs = coeffs + idx_in_batch * coeffs_size;
    T* curr_evals = evals + idx_in_batch * domain_size;
    for (uint64_t eval_idx = 0; eval_idx < domain_size; ++eval_idx) {
      curr_evals[eval_idx] = curr_coeffs[coeffs_size - 1];
      for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
        curr_evals[eval_idx] =
          curr_evals[eval_idx] * domain[eval_idx] + curr_coeffs[coeff_idx];
      }
    }
  }
}

template <typename T>
void poly_eval_estrin_scheme(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int total_number_polynomials = batch_size * domain_size;
  int num_blocks = total_number_polynomials; // Need to check if the number of blocks is a constraint as well

  int threads_per_block = min(prop.maxThreadsPerBlock, coeffs_size);
  int shared_memory_byte = min(prop.sharedMemPerBlock, (coeffs_size + 1)/2 * sizeof(T));

  poly_eval_estrin_scheme_hardware_constraint_kernel<<<num_blocks, threads_per_block, shared_memory_byte>>>(
    coeffs, domain, coeffs_size, domain_size, batch_size, evals);

  cudaDeviceSynchronize();
}

//template <typename T>
//__global__ void poly_eval_estrin_scheme_kernel(
//  const T* coeffs,
//  const T* domain,
//  int coeffs_size,
//  int domain_size,
//  int batch_size,
//  T* evals /*OUT*/)
//{
//  int polynomial_idx = blockIdx.x / domain_size ;
//  int domain_idx =  blockIdx.x % domain_size;
//
//  const T* curr_coeff = coeffs + polynomial_idx * coeffs_size ;
//  T x = domain[domain_idx];
//
//  extern __shared__ T intermediate_result[];
//
//  int total_repeatations_required = coeffs_size/ blockDim.x ;
//
//  // without H/W constraint
//  int total_estrin_reduction_steps = static_cast<int>(std::ceil(std::log2(coeffs_size)));
//
//  for (int i = threadIdx.x; i < )
//}


template <typename T>
__global__ void poly_eval_estrin_scheme_hardware_constraint_kernel(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  int polynomial_idx = blockIdx.x / domain_size ;
  int domain_idx =  blockIdx.x % domain_size;

  const T* curr_coeff = coeffs + polynomial_idx * coeffs_size ;
  T x = domain[domain_idx];

  extern __shared__ T intermediate_result[];

  T x_pow = x * x;
  int total_active_coefficients = (coeffs_size + 1)/2;
  // 1st iteration of estrin scheme
  if(threadIdx.x < total_active_coefficients)
  {
    T Cn = 2 * threadIdx.x < coeffs_size ? coeffs[2 * threadIdx.x]: T::zero();
    T Cn_1 = 2 * threadIdx.x + 1 < coeffs_size ? coeffs[2 * threadIdx.x + 1]: T::zero();
    intermediate_result[threadIdx.x] = Cn + Cn_1 * x ;
  }
  __syncthreads();

  //estrin's reduction
  while(total_active_coefficients > 1)
  {
    if(threadIdx.x < (total_active_coefficients + 1)/2)
    {
      T Cn = 2 * threadIdx.x < total_active_coefficients ? intermediate_result[2 * threadIdx.x]: T::zero();
      T Cn_1 = 2 * threadIdx.x + 1 < total_active_coefficients ? intermediate_result[2 * threadIdx.x + 1]: T::zero();
      intermediate_result[threadIdx.x] = Cn + Cn_1 * x_pow ;
    }
    __syncthreads();
    total_active_coefficients = (total_active_coefficients+ 1) /2;
    x_pow = x_pow * x_pow;
  }
  if (threadIdx.x == 0)
  {
  int out_idx = polynomial_idx * domain_size + domain_idx;
  evals[out_idx] = intermediate_result[0];
  }
}

template <typename T>
void poly_eval_horner(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{
  int total_results_size = domain_size * batch_size;
  int threads_per_block = 256;
  int num_blocks = (total_results_size + threads_per_block - 1) / threads_per_block;

  poly_eval_horner_kernel<<<num_blocks, threads_per_block>>>(
    coeffs, domain, coeffs_size, domain_size, batch_size, evals);
}

template <typename T>
__global__ void poly_eval_horner_kernel(
  const T* coeffs,
  const T* domain,
  int coeffs_size,
  int domain_size,
  int batch_size,
  T* evals /*OUT*/)
{

  uint64_t idx_in_batch = threadIdx.x + blockIdx.x * blockDim.x;

  uint64_t total_eval = domain_size * batch_size;
  
  uint64_t stride = blockDim.x * gridDim.x;

  for(uint64_t i = idx_in_batch; i < total_eval; i= i+ stride)
  {
    int polynomial_idx =  i/domain_size ;
    int domain_idx = i % domain_size;
    
    const T* current_poly_coeff = coeffs + polynomial_idx * coeffs_size;
    T curr_domain = domain[domain_idx];
    T* curr_evals = evals + polynomial_idx * domain_size + domain_idx;

    T result = current_poly_coeff[coeffs_size - 1];
    for (int64_t coeff_idx = coeffs_size - 2; coeff_idx >= 0; --coeff_idx) {
      result = result * curr_domain + current_poly_coeff[coeff_idx];
    }
    *curr_evals = result;
  }
}
