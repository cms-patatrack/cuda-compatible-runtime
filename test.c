#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

void usage(const char* program_name) {
  printf("Usage: %s [-v]\n\nOptions:\n  -h        Print a help message and exits.\n  -v        Be more verbose.\n", program_name);
}

int main(int argc, const char* argv[]) {
  int verbose = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-v") == 0) {
      verbose = 1;
    } else
    if (strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
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

  return EXIT_SUCCESS;
}
