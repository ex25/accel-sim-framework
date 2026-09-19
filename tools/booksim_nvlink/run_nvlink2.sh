#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
booksim_root=${BOOKSIM_ROOT:-}
booksim_bin=${BOOKSIM_BIN:-}
out_dir=${OUT_DIR:-$(mktemp -d /tmp/accelsim-booksim-nvlink2.XXXXXX)}

if [[ -z "${booksim_root}" && -z "${booksim_bin}" ]]; then
  echo "Set BOOKSIM_ROOT to an external BookSim2 checkout or BOOKSIM_BIN." >&2
  exit 2
fi
if [[ -z "${booksim_bin}" ]]; then
  booksim_bin="${booksim_root}/src/booksim"
fi
if [[ ! -x "${booksim_bin}" ]]; then
  echo "BookSim binary is not executable: ${booksim_bin}" >&2
  exit 2
fi

mkdir -p "${out_dir}"
log="${out_dir}/nvlink2.log"
config="${repo_root}/tools/booksim_nvlink/nvlink2_config"

set +e
(cd "${repo_root}" && "${booksim_bin}" "${config}") >"${log}" 2>&1
rc=$?
set -e
cat "${log}"
# BookSim's historical main() returns -1 (shell status 255) when
# TrafficManager::Run() reports success. Treat that status as successful but
# preserve it in the metadata/report.
if [[ ${rc} -ne 0 && ${rc} -ne 255 ]]; then
  echo "BookSim failed with exit code ${rc}; log=${log}" >&2
  exit ${rc}
fi

grep -q "Node 0" "${log}"
grep -q "Node 1" "${log}"
grep -q "Packet latency average =" "${log}"
grep -q "Hops average = 2" "${log}"
echo "NVLINK2_SMOKE_PASS exit=${rc} log=${log}"
