.PHONY: all clean

VERSIONS:=10.0.130 10.1.243 10.2.89 11.0.228 11.1.105 11.2.2 11.3.1 11.4.1

TARGETS:=$(foreach VERSION,$(VERSIONS),bin/test-$(VERSION))

all: $(TARGETS)

clean:
	rm -rf bin

define build-target
bin/test-$(1): CUDA_BASE:=/cvmfs/patatrack.cern.ch/externals/x86_64/rhel7/nvidia/cuda-$(1)
bin/test-$(1): test.c
	@mkdir -p bin
	/usr/bin/gcc -std=c99 -O2 -Wall $$< -I $$(CUDA_BASE)/include -L $$(CUDA_BASE)/lib64 -l cudart_static -l cuda -ldl -lrt -pthread -static-libgcc -o $$@

endef

$(foreach VERSION,$(VERSIONS),$(eval $(call build-target,$(VERSION))))
