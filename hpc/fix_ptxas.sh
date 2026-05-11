#!/bin/bash
# Run this ONCE on HPC headnode to fix ptxas/nvlink for compute nodes

VENV="/data/beegfs/home/saifi/rlt_ur5e/openpi_ur5e/openpi-ur5e/.venv"
SYSROOT="${VENV}/x86_64-conda-linux-gnu/sysroot"

echo "Fixing ptxas and nvlink wrappers..."

# Find the real ptxas (either conda-installed or pip-installed)
if [[ -f "${VENV}/bin/ptxas.real" ]]; then
    REAL_PTXAS="${VENV}/bin/ptxas.real"
elif [[ -f "${VENV}/bin/ptxas" && ! -L "${VENV}/bin/ptxas" ]]; then
    mv "${VENV}/bin/ptxas" "${VENV}/bin/ptxas.real"
    REAL_PTXAS="${VENV}/bin/ptxas.real"
else
    REAL_PTXAS=$(find "${VENV}" -path "*/nvidia/cuda_nvcc/bin/ptxas" -type f 2>/dev/null | head -1)
fi

if [[ -z "${REAL_PTXAS}" || ! -f "${REAL_PTXAS}" ]]; then
    echo "ERROR: Cannot find real ptxas binary!"
    echo "Install it: ~/.local/bin/micromamba install -p ${VENV} \"cuda-nvcc>=12.6\" -c nvidia -y"
    exit 1
fi

echo "Real ptxas: ${REAL_PTXAS}"
echo "Version: $(${REAL_PTXAS} --version 2>&1 | head -1)"

# Create wrapper that UNSETS LD_LIBRARY_PATH before running ptxas
# This prevents sysroot glibc from interfering with ptxas execution
cat > "${VENV}/bin/ptxas" << EOF
#!/bin/bash
# Wrapper: clears LD_LIBRARY_PATH so ptxas uses system libs (not sysroot)
unset LD_LIBRARY_PATH
exec "${REAL_PTXAS}" "\$@"
EOF
chmod +x "${VENV}/bin/ptxas"

# Same for nvlink
REAL_NVLINK=""
if [[ -f "${VENV}/bin/nvlink.real" ]]; then
    REAL_NVLINK="${VENV}/bin/nvlink.real"
elif [[ -f "${VENV}/bin/nvlink" && ! -L "${VENV}/bin/nvlink" ]]; then
    mv "${VENV}/bin/nvlink" "${VENV}/bin/nvlink.real"
    REAL_NVLINK="${VENV}/bin/nvlink.real"
else
    REAL_NVLINK=$(find "${VENV}" -path "*/nvidia/cuda_nvcc/bin/nvlink" -type f 2>/dev/null | head -1)
fi

if [[ -n "${REAL_NVLINK}" && -f "${REAL_NVLINK}" ]]; then
    cat > "${VENV}/bin/nvlink" << EOF
#!/bin/bash
unset LD_LIBRARY_PATH
exec "${REAL_NVLINK}" "\$@"
EOF
    chmod +x "${VENV}/bin/nvlink"
    echo "nvlink wrapper created"
fi

# Verify
echo ""
echo "Testing ptxas wrapper:"
"${VENV}/bin/ptxas" --version
echo ""
echo "Done! ptxas/nvlink wrappers will work even with LD_LIBRARY_PATH set."
