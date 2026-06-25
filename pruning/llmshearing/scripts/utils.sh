get_free_gpus() {
  # Return the top-{num_gpus} GPU indices with the most available VRAM.
  # Prefers idle GPUs (no compute processes) first; fills remaining slots
  # from non-idle GPUs sorted by free VRAM descending.
  local num_gpus=${1:-1}

  mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ')

  local -a idle_gpus=()
  local -a busy_gpus=()   # "free,idx" for sorting

  for line in "${gpus[@]}"; do
    local idx="${line%%,*}"
    local free="${line##*,}"

    if ! nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
      idle_gpus+=("${free},${idx}")
    else
      busy_gpus+=("${free},${idx}")
    fi
  done

  # Sort each group by free VRAM descending
  mapfile -t idle_sorted  < <(printf '%s\n' "${idle_gpus[@]}"  | sort -t, -k1 -rn)
  mapfile -t busy_sorted  < <(printf '%s\n' "${busy_gpus[@]}"  | sort -t, -k1 -rn)

  local -a selected=()
  for entry in "${idle_sorted[@]}" "${busy_sorted[@]}"; do
    [[ -z "$entry" ]] && continue
    (( ${#selected[@]} < num_gpus )) || break
    selected+=("${entry##*,}")
  done

  if (( ${#selected[@]} == 0 )); then
    echo "No GPU found" >&2
    return 1
  fi

  # Print as comma-separated list
  local IFS=','
  echo "${selected[*]}"
}

get_random_port() {
  # Return a random available TCP port in the range 20000-60000.
  local port
  while true; do
    port=$(( RANDOM % 40001 + 20000 ))
    if ! ss -tlnH "sport = :${port}" 2>/dev/null | grep -q .; then
      echo "${port}"
      return 0
    fi
  done
}