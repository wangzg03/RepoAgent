"""Public checks fixed before inference. Run only inside the task Docker sandbox."""

import importlib
import math
from pathlib import Path
import subprocess
import sys
import unittest

suite = unittest.defaultTestLoader.discover("tests")
assert suite.countTestCases() > 0, "No model-written tests were discovered"
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful(), "Model-written tests failed"
print("MODEL_TESTS=" + str(result.testsRun), flush=True)

module = importlib.import_module("calculator")
checks = 0
cases = [
    ("add", 2, 3, 5), ("subtract", 9, 4, 5), ("multiply", 3, 4, 12),
    ("divide", 9, 3, 3), ("add", -4, 2, -2), ("subtract", -4, -2, -2),
    ("multiply", -2, -3, 6), ("divide", -3, 2, -1.5),
    ("add", .1, .2, .3), ("multiply", 1.5, 2, 3),
]
for operation, a, b, expected in cases:
    actual = module.calculate(operation, a, b)
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), operation + " returned the wrong value"
    checks += 1
for operation, a, b in [("divide", 1, 0), ("unknown", 1, 2)]:
    try:
        module.calculate(operation, a, b)
    except ValueError:
        checks += 1
    else:
        raise AssertionError(operation + " must raise ValueError")
for operation, a, b, expected in cases[:4] + [cases[7], cases[8]]:
    call = subprocess.run([sys.executable, "calculator.py", operation, str(a), str(b)],
                          capture_output=True, text=True, timeout=10, check=False)
    assert call.returncode == 0, "CLI operation failed: " + operation
    assert math.isclose(float(call.stdout.strip()), expected, rel_tol=1e-9, abs_tol=1e-9), "Wrong CLI value"
    checks += 1
help_result = subprocess.run([sys.executable, "calculator.py", "--help"], capture_output=True,
                             text=True, timeout=10, check=False)
assert help_result.returncode == 0 and help_result.stdout.strip(), "--help must succeed and display usage"
checks += 1
zero = subprocess.run([sys.executable, "calculator.py", "divide", "1", "0"], capture_output=True,
                      text=True, timeout=10, check=False)
assert zero.returncode != 0 and zero.stderr.strip(), "Division by zero must fail with an error"
assert "zero" in zero.stderr.lower() or "零" in zero.stderr, "Division-by-zero error must be clear"
checks += 1
unknown = subprocess.run([sys.executable, "calculator.py", "unknown", "1", "2"], capture_output=True,
                         text=True, timeout=10, check=False)
assert unknown.returncode != 0 and unknown.stderr.strip(), "Unknown CLI operation must fail"
checks += 1
readme = Path("README.md")
assert readme.is_file() and readme.read_text(encoding="utf-8").strip(), "README is required"
checks += 1
print(f"PUBLIC_ACCEPTANCE={checks}/{checks}", flush=True)
