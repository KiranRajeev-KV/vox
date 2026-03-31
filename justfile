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
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run pytest tests/test_benchmark.py -v

install-hooks:
    uv run pre-commit install

web:
    LD_LIBRARY_PATH="{{ld_lib_path}}" uv run main.py web
