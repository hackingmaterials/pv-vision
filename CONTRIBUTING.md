# Contributing to pv-vision

## Running the tests

```bash
pip install -e . pytest
python -m pytest -q
```

The tests run on CPU and need no model weights. They use the example images
in `examples/`.

CI runs the same tests on Python 3.10 and 3.12 for every pull request.

## Pull requests

- Open a pull request against `main`; direct pushes to `main` are blocked.
- Include a test for every bug fix or behavior change.
- Changes to cropping, model architectures, training defaults, defect
  classification, crack analysis or `data/` are always reviewed by a
  maintainer.
