# Running Tests

## Quick Reference

```bash
# Activate venv first
source ~/Documents/appmana/.venv/bin/activate

# Run all unit tests, stop on first failure, grep for failures
python -m pytest tests/unit/ -x 2>&1 | grep FAILED

# Run a specific test file
python -m pytest tests/unit/test_cli_args.py -x 2>&1 | grep FAILED

# Run a specific test class or function
python -m pytest tests/unit/test_cli_args.py::TestFastArg -x 2>&1 | grep FAILED
python -m pytest tests/unit/test_cli_args.py::test_default_values -x 2>&1 | grep FAILED

# Run with verbose output (shows each test name + PASSED/FAILED)
python -m pytest tests/unit/ -x --override-ini="addopts=" -v 2>&1 | grep -E "FAILED|PASSED"

# Run with short traceback on failure
python -m pytest tests/unit/ -x --tb=short 2>&1 | grep FAILED
```

## Why Output Is Noisy

`pytest.ini` has `addopts = -s` which disables output capture, so all logging and print statements appear in the output. This makes `tail` and `head` unreliable for finding the summary line.

**Best practice**: Always pipe through `grep FAILED`. If grep returns nothing (and exits with code 1), all tests passed. If grep returns lines (exit code 0), those are the failures.

**Note on exit codes**: `grep` exits with code 1 when there are no matches. So `grep FAILED` returning exit code 1 with no output means **success** (zero failures). Exit code 0 with output means there are failures.

## Test Categories

| Directory | What | How to Run |
|-----------|------|------------|
| `tests/unit/` | Fast unit tests | `python -m pytest tests/unit/ -x` |
| `tests/inference/` | GPU inference tests (slow) | `python -m pytest tests/inference/ -k "workflow-name and normalvram"` |
| `tests/execution/` | Execution engine tests | `python -m pytest tests/execution/` |

## Filtering

```bash
# By marker
python -m pytest tests/ -m "not inference" -x

# By keyword
python -m pytest tests/unit/ -k "test_fast" -x

# By file pattern
python -m pytest tests/unit/test_cli*.py -x
```

## Parallel Execution

```bash
python -m pytest tests/unit/ -x -n auto  # requires pytest-xdist
```
