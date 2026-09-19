#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
booksim_root=${BOOKSIM_ROOT:-}
booksim_bin=${BOOKSIM_BIN:-}
out_dir=${OUT_DIR:-$(mktemp -d /tmp/accelsim-booksim-nvlink2-sweep.XXXXXX)}

if [[ -z "${booksim_bin}" ]]; then
  [[ -n "${booksim_root}" ]] || {
    echo "Set BOOKSIM_ROOT or BOOKSIM_BIN." >&2
    exit 2
  }
  booksim_bin="${booksim_root}/src/booksim"
fi
[[ -x "${booksim_bin}" ]] || { echo "Not executable: ${booksim_bin}" >&2; exit 2; }
mkdir -p "${out_dir}"

config="${repo_root}/tools/booksim_nvlink/nvlink2_config"
network="${repo_root}/tools/booksim_nvlink/nvlink2_network"
csv="${out_dir}/nvlink2_sweep.csv"
printf 'link_latency,packet_size,latency_cycles,accepted_rate,sim_cycles\n' >"${csv}"

for link_latency in 1 4 8; do
  network_variant="${out_dir}/network-${link_latency}"
  sed "s/router 1 1/router 1 ${link_latency}/" "${network}" >"${network_variant}"
  for packet_size in 1 4 16; do
    log="${out_dir}/lat${link_latency}-pkt${packet_size}.log"
    set +e
    (cd "${repo_root}" && "${booksim_bin}" "${config}" \
      "network_file=${network_variant}" "packet_size=${packet_size}") >"${log}" 2>&1
    rc=$?
    set -e
    [[ ${rc} -eq 0 || ${rc} -eq 255 ]] || {
      echo "BookSim failed for latency=${link_latency} packet=${packet_size}: rc=${rc}" >&2
      exit ${rc}
    }
    grep -q 'Node 0' "${log}"
    grep -q 'Node 1' "${log}"
    grep -q 'Hops average = 2' "${log}"
    latency=$(awk '/Packet latency average = [0-9]/ { value=$5 } END { print value }' "${log}")
    accepted=$(awk '/Accepted packet rate average = [0-9]/ { value=$6 } END { print value }' "${log}")
    cycles=$(awk '/Time taken is [0-9]+ cycles/ { value=$4 } END { print value }' "${log}")
    [[ -n "${latency}" && -n "${accepted}" && -n "${cycles}" ]] || {
      echo "Missing final statistics for ${log}" >&2
      exit 1
    }
    printf '%s,%s,%s,%s,%s\n' "${link_latency}" "${packet_size}" \
      "${latency}" "${accepted}" "${cycles}" >>"${csv}"
  done
done

cat "${csv}"
echo "NVLINK2_SWEEP_PASS csv=${csv}"
