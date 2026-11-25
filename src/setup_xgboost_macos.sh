#!/bin/bash
# macOS only. Run once after creating the virtual environment (and again if xgboost is reinstalled):
#     bash src/setup_xgboost_macos.sh
#
# Why: the xgboost wheel for macOS does not ship the OpenMP runtime (libomp.dylib) and only looks
# for it under Homebrew (/opt/homebrew/opt/libomp/lib). scikit-learn's wheel does ship one. This
# script tells xgboost to also look in scikit-learn's folder, so no system-wide install is needed.
# It changes one file inside .venv and nothing outside it.
#
# Alternative that needs no script: install Homebrew, then `brew install libomp`.
# To undo: .venv/bin/python -m pip install --force-reinstall --no-deps xgboost
set -euo pipefail

cd "$(dirname "$0")/.."
SITE=$(.venv/bin/python -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
LIB="$SITE/xgboost/lib/libxgboost.dylib"
OMP_DIR="$SITE/sklearn/.dylibs"
RPATH="@loader_path/../../sklearn/.dylibs"

[ -f "$LIB" ] || { echo "xgboost is not installed in .venv"; exit 1; }
[ -f "$OMP_DIR/libomp.dylib" ] || { echo "scikit-learn's libomp.dylib not found"; exit 1; }

if otool -l "$LIB" | grep -q "$RPATH"; then
    echo "already set up"
else
    install_name_tool -add_rpath "$RPATH" "$LIB"
    codesign --force --sign - "$LIB"      # re-sign locally; required on Apple silicon after editing
    echo "added search path: $RPATH"
fi

.venv/bin/python -I -c "import xgboost; print('xgboost', xgboost.__version__, 'loads')"
