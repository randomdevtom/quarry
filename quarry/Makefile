CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall
LDLIBS  = -lm -lpthread
# tokenizer: libllama. A recent llama.cpp is needed for Inkling 2 (inkling2_loop);
# vendor/llama.cpp is used when it has been built, the system libllama otherwise.
LLAMA_DIR ?= ../vendor/llama.cpp
ifneq ($(wildcard $(LLAMA_DIR)/build/bin/libllama.so),)
LLAMA_FLAGS = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -L$(LLAMA_DIR)/build/bin -Wl,-rpath,$(abspath $(LLAMA_DIR)/build/bin)
endif
LLAMA_LIBS = $(LLAMA_FLAGS) -lllama -lggml -lggml-base
CORE    = src/qgguf.c src/quants.c src/model.c src/experts.c src/inkling.c

all: build/quarry build/quarry-bench build/quarry-info

build/quarry: $(CORE) src/main.c src/*.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(CORE) src/main.c -o $@ $(LDLIBS) $(LLAMA_LIBS)

build/quarry-bench: $(CORE) tools/quarry_bench.c src/*.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(CORE) tools/quarry_bench.c -o $@ $(LDLIBS)

build/quarry-info: src/qgguf.c tools/quarry_info.c src/qgguf.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/qgguf.c tools/quarry_info.c -o $@

test: build/quarry
	$(CC) $(CFLAGS) $(CORE) tests/test_quants.c -o build/test_quants $(LDLIBS)
	$(CC) $(CFLAGS) $(CORE) tests/test_forward.c -o build/test_forward $(LDLIBS)
	$(CC) $(CFLAGS) $(CORE) tests/test_inkling.c -o build/test_inkling $(LDLIBS)
	$(CC) $(CFLAGS) src/qgguf.c src/quants.c tests/test_quants2.c -o build/test_quants2 $(LDLIBS)

ref: tests/ref_dump.c
	$(CC) -O2 -Wno-deprecated-declarations tests/ref_dump.c -o build/ref_dump $(LLAMA_LIBS)

clean:
	rm -rf build

.PHONY: all test clean ref
