CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall
LDLIBS  = -lm -lpthread
CORE    = src/qgguf.c src/quants.c src/model.c src/experts.c

all: build/quarry build/quarry-bench build/quarry-info

build/quarry: $(CORE) src/main.c src/*.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(CORE) src/main.c -o $@ $(LDLIBS) -lllama -lggml -lggml-base

build/quarry-bench: $(CORE) tools/quarry_bench.c src/*.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(CORE) tools/quarry_bench.c -o $@ $(LDLIBS)

build/quarry-info: src/qgguf.c tools/quarry_info.c src/qgguf.h
	@mkdir -p build
	$(CC) $(CFLAGS) src/qgguf.c tools/quarry_info.c -o $@

test: build/quarry
	$(CC) $(CFLAGS) $(CORE) tests/test_quants.c -o build/test_quants $(LDLIBS)
	$(CC) $(CFLAGS) $(CORE) tests/test_forward.c -o build/test_forward $(LDLIBS)

clean:
	rm -rf build

.PHONY: all test clean
