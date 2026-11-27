CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall
LDLIBS  = -lm -lpthread
# tokenizer: libllama. A recent llama.cpp is needed for Inkling 2 (inkling2_loop);
# vendor/llama.cpp is used when it has been built, the system libllama otherwise.
LLAMA_DIR ?= ../vendor/llama.cpp
LLAMA_BUILD ?= build
LLAMA_BIN = $(LLAMA_DIR)/$(LLAMA_BUILD)/bin
ifneq ($(wildcard $(LLAMA_BIN)/libllama.so),)
LLAMA_FLAGS = -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include -L$(LLAMA_BIN) -Wl,-rpath,$(abspath $(LLAMA_BIN))
endif
LLAMA_LIBS = $(LLAMA_FLAGS) -lllama -lggml -lggml-base
# vision (Qwen3.5/3.6 images): llama.cpp's libmtmd, when it has been built
ifneq ($(wildcard $(LLAMA_BIN)/libmtmd.so),)
LLAMA_LIBS += -I$(LLAMA_DIR)/tools/mtmd -lmtmd -DQUARRY_VISION
endif
# GPU=1: the hottest experts run on the GPU through ggml (llama.cpp built with -DGGML_VULKAN=ON,
# -DGGML_CUDA=ON, ...). Without it gpu.c is a stub and everything runs on the CPU.
ifeq ($(GPU),1)
GPU_FLAGS = -DQUARRY_GPU -I$(LLAMA_DIR)/ggml/include
endif
CORE    = src/qgguf.c src/quants.c src/model.c src/experts.c src/inkling.c src/qwen35.c src/lfm2.c src/laguna.c src/gpu.c

all: build/quarry build/quarry-bench build/quarry-info

src/chat_html.h: src/chat.html
	xxd -i $< > $@

build/quarry: $(CORE) src/main.c src/*.h src/chat_html.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(GPU_FLAGS) $(CORE) src/main.c -o $@ $(LDLIBS) $(LLAMA_LIBS)

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
	$(CC) $(CFLAGS) $(CORE) tests/test_q35.c -o build/test_q35 $(LDLIBS)
	$(CC) $(CFLAGS) $(CORE) tests/test_lfm.c -o build/test_lfm $(LDLIBS)
	$(CC) $(CFLAGS) $(CORE) tests/test_lag.c -o build/test_lag $(LDLIBS)
	$(CC) $(CFLAGS) src/qgguf.c src/quants.c tests/test_kquants.c -o build/test_kquants $(LDLIBS)
	./build/test_kquants

ref: tests/ref_dump.c tests/ref_logits.c
	$(CC) -O2 -Wno-deprecated-declarations tests/ref_dump.c -o build/ref_dump $(LLAMA_LIBS)
	$(CC) -O2 -Wno-deprecated-declarations tests/ref_logits.c -o build/ref_logits $(LLAMA_LIBS)

clean:
	rm -rf build

.PHONY: all test clean ref
