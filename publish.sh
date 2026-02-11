#!/usr/bin/env bash
# Publish summscriber to PyPI.
# Prerequisites:
#   pip install build twine
#   PyPI account and token (https://pypi.org/manage/account/token/)
#   For first upload: twine upload will prompt for username __token__ and password (pypi-...)
#   Or set: export TWINE_USERNAME=__token__  export TWINE_PASSWORD=pypi-your-token

set -e
cd "$(dirname "$0")"

echo "Cleaning old build artifacts..."
rm -rf build/ dist/ *.egg-info

echo "Building package..."
python -m build

echo "Uploading to PyPI..."
twine upload dist/*

echo "Done. Check https://pypi.org/project/summscriber/"
