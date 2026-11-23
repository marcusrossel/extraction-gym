FEATURES ?=
SINGLE_ROOT ?=
BENCH_ARGS ?=

CARGO_FLAGS := --release $(if $(FEATURES),--features=$(FEATURES))
BENCH_FLAGS := $(if $(FEATURES),--features=$(FEATURES)) $(if $(SINGLE_ROOT),--single-root) $(BENCH_ARGS)

.PHONY: all build bench test nits clean

all: test nits bench

build:
	cargo build $(CARGO_FLAGS)

bench:
	uv run scripts/bench.py $(BENCH_FLAGS)

test:
	cargo test --release

nits:
	@if ! cargo fmt --version >/dev/null 2>&1 || ! cargo clippy --version >/dev/null 2>&1; then \
		if [ -z "$$IN_NIX_SHELL" ] && command -v rustup >/dev/null 2>&1; then \
			rustup component add rustfmt clippy; \
		fi; \
	fi
	cargo fmt -- --check
	cargo clean --doc
	cargo clippy --tests

clean:
	rm -rf output
