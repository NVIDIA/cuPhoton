# Changelog

## Unreleased

### Breaking changes

- Python 3.11 is no longer supported. Use Python 3.12, 3.13, or 3.14.
- Photutils is now optional and is no longer installed by `pip install cuphoton`.
  Install `cuphoton[photometry]` for CPU photometry or `cuphoton[gpu]` for the
  combined GPU and photometry dependencies.
