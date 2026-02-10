# gori-deep-train-core

A Python package containing many basic primitives used to build pipelines for the [`gori-deep-train`](https://github.com/RedReservoir/gori-deep-train) project. You can read further detailed documentation [here](https://gori-deep-train-core.readthedocs.io/en/latest/index.html).

This is a fully functional package, but it is neither stable, distributable or in a production-ready state. No tests are being written, new features will constantly be implemented, heavy refactors may occur, and some documentation may not be up to date.

As such, no dependencies with accurate versions are listed in the `pyproject.toml` file, and none will be installed when installing this package. However, the list below roughly outlines other used packages, which you will need to install. 

  - [`gori-py-utils`](https://github.com/RedReservoir/gori-py-utils)
  - `numpy`
  - `torch`
  - `torchvision`
  - `timm`

You can install this Python package directly from GitHub:

```
pip install git+https://github.com/RedReservoir/gori-deep-train-core
```
