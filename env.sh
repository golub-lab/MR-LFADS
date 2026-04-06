#!/usr/bin/env bash
set -euo pipefail

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"

mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/compiler_vars.sh" <<'EOF'
#!/usr/bin/env bash
export CC=gcc
export CXX=g++
EOF

cat > "$DEACTIVATE_DIR/compiler_vars.sh" <<'EOF'
#!/usr/bin/env bash
unset CC
unset CXX
EOF

chmod +x "$ACTIVATE_DIR/compiler_vars.sh" "$DEACTIVATE_DIR/compiler_vars.sh"

python -m pip install -r requirements.txt
pip install -e .