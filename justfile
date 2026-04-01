justfile_dir := justfile_directory()
ld_lib_path := justfile_dir + "/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:" + justfile_dir + "/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

run:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run main.py

debug:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run main.py --debug

test:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest

lint:
    uv run ruff check vox/ tests/

fmt:
    uv run ruff format vox/ tests/

typecheck:
    uv run pyright .

benchmark:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest tests/test_benchmark.py -m benchmark -v

benchmark-quick:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest tests/test_benchmark.py -m benchmark --benchmark-tier quick -v

benchmark-full:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest tests/test_benchmark.py -m benchmark --benchmark-tier full -v

benchmark-single dataset:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest tests/test_benchmark.py -m benchmark --benchmark-datasets {{dataset}} -v

install-hooks:
    uv run pre-commit install

web:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run main.py web

history:
    uv run main.py history

history-search query:
    uv run main.py history --search "{{query}}"

search query:
    uv run main.py search "{{query}}"

stats:
    uv run main.py stats
