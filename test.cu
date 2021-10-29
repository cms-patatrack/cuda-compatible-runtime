#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

__global__
static void setSupported(bool* result) {
  *result = true;
}

bool isCudaDeviceSupported(int device) {
  bool supported = false;
  bool* supported_d;

  // select the requested device - will fail if the index is invalid
  cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess)
    return false;

  // allocate memory for the flag on the device
  status = cudaMalloc(&supported_d, sizeof(bool));
  if (status != cudaSuccess)
    return false;

  // initialise the flag on the device
  status = cudaMemset(supported_d, 0x00, sizeof(bool));
  if (status != cudaSuccess)
    return false;

  // try to set the flag on the device
  setSupported<<<1, 1>>>(supported_d);

  // check for an eventual error from launching the kernel on an unsupported device
  status = cudaGetLastError();
  if (status != cudaSuccess)
    return false;

  // wait for the kernelto run
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    return false;

  // copy the flag back to the host
  status = cudaMemcpy(&supported, supported_d, sizeof(bool), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    return false;

  // free the device memory
  status = cudaFree(supported_d);
  if (status != cudaSuccess)
    return false;

  // reset the device
  status = cudaDeviceReset();
  if (status != cudaSuccess)
    return false;

  return supported;
}

void usage(const char* program_name) {
  printf("Usage: %s [-v]\n"
         "\n"
         "Options:\n"
         "  -d        Require at least one supported CUDA device.\n"
         "  -h        Print this help message and exits.\n"
         "  -k        If there are any CUDA devices, check that at least one supports launching a CUDA kernel.\n"
         "  -v        Be more verbose.\n",
         program_name);
}

int main(int argc, const char* argv[]) {
  bool check_device = false;
  bool check_kernel = false;
  bool verbose = false;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-d") == 0) {
      check_device = true;
    } else
    if (strcmp(argv[i], "-k") == 0) {
      check_kernel = true;
    } else
    if (strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else
    if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    } else {
      printf("%s: Invalid option '%s'\n\n", argv[0], argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  cudaError_t status = cudaSuccess;

  int driverVersion = 0;
  status = cudaDriverGetVersion(&driverVersion);
  if (status != cudaSuccess) {
    if (verbose)
      printf("Unable to query the CUDA driver version: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }
  if (driverVersion == 0) {
    if (verbose)
      printf("CUDA driver not detected\n");
    return EXIT_FAILURE;
  }
  if (verbose)
    printf("CUDA driver version %d.%d\n", (driverVersion / 1000), (driverVersion % 1000 / 10));

  int runtimeVersion = 0;
  status = cudaRuntimeGetVersion(&runtimeVersion);
  if (status != cudaSuccess) {
    if (verbose) 
      printf("Unable to query the CUDA runtime version: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }
  if (verbose) 
    printf("CUDA runtime version %d.%d\n", (runtimeVersion / 1000), (runtimeVersion % 1000 / 10));
  else
    printf("%d.%d\n", (runtimeVersion / 1000), (runtimeVersion % 1000 / 10));

  int devices = 0;
  status = cudaGetDeviceCount(&devices);
  if (status != cudaSuccess) {
    if (verbose)
      printf("Unable to query the number of CUDA devices: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }

  // If requested, check that the system has at least one CUDA device
  if (check_device and devices == 0) {
    if (verbose)
      printf("No CUDA devices detected\n");
    return EXIT_FAILURE;
  }

  // If the kernel test was not requested, or if there are no CUDA devices, do not try launching a kernel
  if (not check_kernel or devices == 0)
    return EXIT_SUCCESS;

  bool supported = false;
  for (int device = 0; device < devices; ++device) {
    if (isCudaDeviceSupported(device))
      supported = true;
  }
  if (not supported) {
    if (verbose)
      printf("None of the CUDA devices supports launching and running a CUDA kernel.\n");
    return EXIT_FAILURE;
  }

  // All tests completed successfully
  return EXIT_SUCCESS;
}
