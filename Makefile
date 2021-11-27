.PHONY: all clean

# CUDA versions
COMPILER_VERSION:=11.5.0
RUNTIME_VERSIONS:=10.0.130 10.1.243 10.2.89 11.0.228 11.1.105 11.2.2 11.3.1 11.4.3 11.5.0

# CUDA installation
CUDA_VERSIONED_BASE:=/cvmfs/patatrack.cern.ch/externals/x86_64/rhel7/nvidia
CUDA_COMPILER_BASE:=$(CUDA_VERSIONED_BASE)/cuda-$(COMPILER_VERSION)

# CUDA hardware architecture
CUDA_ARCH:=60 70 75
CUDA_ARCH_FLAGS:=$(foreach ARCH,$(CUDA_ARCH),-gencode arch=compute_$(ARCH),code=[sm_$(ARCH),compute_$(ARCH)])

# targets
TARGETS:=$(foreach VERSION,$(RUNTIME_VERSIONS),bin/test-$(VERSION))

all: $(TARGETS)

clean:
	rm -rf bin

$(TARGETS): bin

bin:
	@mkdir -p bin

define build-target
bin/test-$(1): CUDA_BASE:=$(CUDA_VERSIONED_BASE)/cuda-$(1)
bin/test-$(1): test.cu
	$(CUDA_COMPILER_BASE)/bin/nvcc -std=c++11 -O2 -g $(CUDA_ARCH_FLAGS) $$< -I $(CUDA_COMPILER_BASE)/include -L $$(CUDA_BASE)/lib64 -L $$(CUDA_BASE)/lib64/stubs --cudart static -ldl -lrt --use-local-env --compiler-bindir /usr/bin/g++ --compiler-options '-Wall -pthread -static-libgcc -static-libstdc++' -o $$@

endef

$(foreach VERSION,$(RUNTIME_VERSIONS),$(eval $(call build-target,$(VERSION))))
