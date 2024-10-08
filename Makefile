.PHONY: all clean

# aarch64, ppc64le, x86_64
ARCH:=$(shell uname -m)

# rhel7, rhel8, unknown
OS:=$(shell if [ -f /etc/redhat-release ]; then if cat /etc/redhat-release | grep -q 'release 7'; then echo 'rhel7'; elif cat /etc/redhat-release | grep -q 'release 8'; then echo 'rhel8'; else echo 'unknown'; fi; else echo 'unknown'; fi)
$(info OS detected: $(OS))

# system compiler
CXX:=/usr/bin/g++

# CUDA versions
COMPILER_VERSION:=12.4.0
RUNTIME_VERSIONS:=10.0.130 10.1.243 10.2.89 11.0.3 11.1.1 11.2.2 11.3.1 11.4.4 11.5.2 11.6.2 11.7.1 11.8.0 12.0.1 12.1.1 12.2.2 12.3.2 12.4.1 12.5.1 12.6.2

# CUDA installation
CUDA_VERSIONED_BASE:=/cvmfs/patatrack.cern.ch/externals/$(ARCH)/$(OS)/nvidia


# targets
TARGETS:=$(foreach VERSION,$(RUNTIME_VERSIONS),bin/test-$(VERSION))

all: $(TARGETS)

clean:
	rm -rf bin drivers

$(TARGETS): bin drivers

bin:
	@mkdir -p bin


define numerical_version
$(shell echo $1 | cut -d. -f1-2 | sed -e 's/\./ * 100 + /g' | bc)
endef

define version_greater_or_equal
$(shell [ $(call numerical_version, $1) -ge $(call numerical_version, $2) ] && echo true)
endef

# CUDA hardware architecture of interest to CMS supported by each CUDA version:
#   sm_60 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90
define supported_cuda_archs
$(strip $(if $(call version_greater_or_equal, $1, 11.8.0), \
  60 70 75 80 86 89 90, \
$(if $(call version_greater_or_equal, $1, 11.1.0), \
  60 70 75 80 86, \
$(if $(call version_greater_or_equal, $1, 11.0.0), \
  60 70 75 80, \
  60 70 75 \
))))
endef

# CUDA gencode flags for the architecture supported by each CUDA version:
#   -gencode arch=compute_NN,code=[sm_NN,compute_NN] ...
# These are compiled explicitly because JIT may not work with a compatibility driver.
define supported_cuda_arch_flags
$(foreach ARCH,$(call supported_cuda_archs, $1),-gencode arch=compute_$(ARCH),code=[sm_$(ARCH),compute_$(ARCH)])
endef

define build-target
bin/test-$1: CUDA_BASE:=$(CUDA_VERSIONED_BASE)/cuda-$1
bin/test-$1: test.cu
ifeq ("$(wildcard $(CUDA_VERSIONED_BASE)/cuda-$1)","")
	@echo "CUDA $1 is not available for $(OS) on $(ARCH)"
else
	@echo "CUDA $1 is available for $(OS) on $(ARCH) for GPU architectures $(call supported_cuda_archs, $1)"
	$$(CUDA_BASE)/bin/nvcc -std=c++11 -O2 -g $$(call supported_cuda_arch_flags, $1) $$< -I $$(CUDA_BASE)/include -L $$(CUDA_BASE)/lib64 -L $$(CUDA_BASE)/lib64/stubs --cudart static -ldl -lrt --use-local-env --compiler-bindir $(CXX) --compiler-options '-Wall -pthread -static-libgcc -static-libstdc++' -o $$@
endif
endef

$(foreach VERSION,$(RUNTIME_VERSIONS),$(eval $(call build-target,$(VERSION))))

drivers: $(CUDA_VERSIONED_BASE)/compat
	rsync -ar $(CUDA_VERSIONED_BASE)/compat/* drivers
