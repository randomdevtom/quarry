#!/usr/bin/env bash
# Quarry installer: checks the machine, builds everything, gets a model, picks settings for
# your RAM, and (optionally) runs Quarry as a background service with Tailscale and opencode.
#
#   ./install.sh                 interactive
#   ./install.sh --yes           accept the defaults (download the default model, install the service)
#   ./install.sh --model PATH    use a GGUF you already have
#   ./install.sh --uninstall     remove the service, config and links (keeps models and the build)
#
# Options: --port N (8090)  --ctx N (16384)  --cache GB (auto)  --models-dir DIR (~/models/quarry)
#          --no-service  --tailscale / --no-tailscale  --opencode / --no-opencode  --yes
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QDIR="$ROOT/quarry"
LLAMA_DIR="$ROOT/vendor/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT="e6d624b9a4b95c6b52f5ca384d70b75581f9821a"   # tested with this version (tokenizer only)
DEFAULT_MODEL_URL="https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF/resolve/main/Qwen3.8-35B-A3B-Q3_K_M.gguf"
DEFAULT_MODEL_NAME="Qwen3.8-35B-A3B-Q3_K_M.gguf"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/quarry"
CONF="$CONF_DIR/quarry.env"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
BIN_LINK="$HOME/.local/bin/quarry"

MODEL="" PORT=8090 CTX=16384 CACHE="" MODELS_DIR="$HOME/models/quarry"
YES=0 SERVICE=ask TAILSCALE=ask OPENCODE=ask UNINSTALL=0

# ---------- output helpers ----------
if [ -t 1 ]; then B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; O=$'\e[38;5;179m'; N=$'\e[0m'; else B= D= G= Y= R= O= N=; fi
step() { printf '\n%s==>%s %s%s%s\n' "$O" "$N" "$B" "$*" "$N"; }
ok()   { printf '   %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '   %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%s✗ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }
ask()  {   # ask "question" default(y/n) -> 0 for yes
  local q="$1" def="$2" a
  if [ "$YES" = 1 ] || [ ! -t 0 ]; then [ "$def" = y ]; return; fi
  read -r -p "   $q [$( [ "$def" = y ] && echo Y/n || echo y/N )] " a || true
  a="${a:-$def}"; [[ "$a" =~ ^[Yy] ]]
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --no-service) SERVICE=no; shift ;;
    --tailscale) TAILSCALE=yes; shift ;;
    --opencode) OPENCODE=yes; shift ;;
    --no-tailscale) TAILSCALE=no; shift ;;
    --no-opencode) OPENCODE=no; shift ;;
    --yes|-y) YES=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
done

printf '%s\n' "${O}  ▬▬▬▬▬▬▬${N}" "${O}   ▬▬▬▬▬ ${N}  ${B}Quarry${N} installer" "${O}    ▬▬▬  ${N}  ${D}MoE models bigger than your RAM, on the CPU${N}"

# ---------- uninstall ----------
if [ "$UNINSTALL" = 1 ]; then
  step "Removing the Quarry service and config"
  if command -v systemctl >/dev/null && systemctl --user list-unit-files quarry.service >/dev/null 2>&1; then
    systemctl --user disable --now quarry.service 2>/dev/null || true
  fi
  rm -f "$UNIT_DIR/quarry.service" && systemctl --user daemon-reload 2>/dev/null || true
  [ -L "$BIN_LINK" ] && rm -f "$BIN_LINK"
  rm -rf "$CONF_DIR"
  ok "done (models in $MODELS_DIR and the build in $ROOT are kept)"
  exit 0
fi

# ---------- 1. machine check ----------
step "Checking this machine"
[ "$(uname -s)" = Linux ] || die "Quarry runs on Linux only"
[ "$(uname -m)" = x86_64 ] || die "Quarry needs an x86-64 CPU"
grep -qw avx512_vnni /proc/cpuinfo || die "this CPU has no AVX-512 VNNI (needs Intel Ice Lake or newer, or AMD Zen 4 or newer)"
ok "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//') (AVX-512 VNNI)"
MEM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
ok "RAM: ${MEM_GB} GB"
missing=()
for t in gcc g++ make cmake git curl python3; do command -v "$t" >/dev/null || missing+=("$t"); done
if [ ${#missing[@]} -gt 0 ]; then
  hint="install them with your package manager"
  command -v pacman >/dev/null && hint="sudo pacman -S --needed base-devel cmake git curl python"
  command -v apt-get >/dev/null && hint="sudo apt install build-essential cmake git curl python3"
  command -v dnf >/dev/null && hint="sudo dnf install gcc gcc-c++ make cmake git curl python3"
  die "missing tools: ${missing[*]}  ->  $hint"
fi
echo 'int main(void){return 0;}' | gcc -fopenmp -x c - -o /dev/null 2>/dev/null || die "gcc has no OpenMP support (install libgomp)"
ok "build tools: gcc $(gcc -dumpversion), cmake, git, curl"

# ---------- 2. llama.cpp (tokenizer library) ----------
step "Building the tokenizer library (llama.cpp)"
if [ -f "$LLAMA_DIR/build/bin/libllama.so" ]; then
  ok "already built ($LLAMA_DIR)"
else
  if [ ! -d "$LLAMA_DIR/.git" ]; then
    mkdir -p "$LLAMA_DIR"
    git -C "$LLAMA_DIR" init -q
    git -C "$LLAMA_DIR" remote add origin "$LLAMA_REPO"
  fi
  git -C "$LLAMA_DIR" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF >/dev/null
  printf '   %sthis takes a few minutes…%s\n' "$D" "$N"
  cmake --build "$LLAMA_DIR/build" --target llama -j "$(nproc)" >/dev/null
  ok "built libllama"
fi

# ---------- 3. Quarry ----------
step "Building Quarry"
make -C "$QDIR" -j "$(nproc)" >/dev/null
ok "built $QDIR/build/quarry"
mkdir -p "$(dirname "$BIN_LINK")"
ln -sf "$QDIR/build/quarry" "$BIN_LINK"
ok "linked $BIN_LINK"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH to run 'quarry' from anywhere";; esac

# ---------- 4. model ----------
step "Model"
if [ -z "$MODEL" ] && [ -f "$CONF" ]; then MODEL="$(. "$CONF"; echo "${QUARRY_MODEL:-}")"; fi
if [ -n "$MODEL" ]; then
  [ -f "$MODEL" ] || die "model not found: $MODEL"
  ok "using $MODEL"
else
  MODEL="$MODELS_DIR/$DEFAULT_MODEL_NAME"
  if [ -f "$MODEL" ] && [ ! -f "$MODEL.part" ]; then
    ok "found $MODEL"
  else
    echo "   Default model: Qwen3.8-35B-A3B Distill, Q3_K_M (17.7 GB, a Qwen3.6-35B-A3B fine-tune)."
    ask "Download it to $MODELS_DIR?" y || die "no model. Rerun with --model /path/to/model.gguf"
    mkdir -p "$MODELS_DIR"
    free_gb=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc 0-9)
    [ "${free_gb:-0}" -ge 18 ] || die "only ${free_gb} GB free in $MODELS_DIR, 18 GB needed"
    curl -L --fail -C - --progress-bar -o "$MODEL.part" "$DEFAULT_MODEL_URL"
    mv "$MODEL.part" "$MODEL"
    ok "downloaded $MODEL"
  fi
fi
ARCH="$("$QDIR/build/quarry-info" "$MODEL" 2>/dev/null | sed -n 's/^arch: \([^ ]*\).*/\1/p' | head -1)"
case "$ARCH" in
  qwen35moe|qwen3moe|mellum|inkling2_loop) ok "architecture: $ARCH" ;;
  *) die "unsupported model architecture '${ARCH:-unknown}' (Quarry runs qwen35moe, qwen3moe, mellum and inkling2_loop)" ;;
esac

# ---------- 5. settings for this machine ----------
step "Settings"
if [ -z "$CACHE" ]; then
  # leave ~6.5 GB for the OS, other apps, dense weights and buffers; the rest caches experts
  CACHE=$(awk -v m="$MEM_GB" 'BEGIN { c = m - 6.5; if (c < 1) c = 1; if (c > 48) c = 48; printf "%.1f", c }')
fi
ok "expert cache: ${CACHE} GB (of ${MEM_GB} GB RAM; change QUARRY_CACHE_GB in $CONF)"
ok "context: $CTX tokens, port: $PORT"
mkdir -p "$CONF_DIR"
cat > "$CONF" <<EOF
# Quarry settings (read by the quarry.service user unit). Restart after editing:
#   systemctl --user restart quarry
QUARRY_MODEL=$MODEL
QUARRY_PORT=$PORT
QUARRY_CTX=$CTX
QUARRY_CACHE_GB=$CACHE
EOF
ok "wrote $CONF"

# ---------- 6. background service ----------
step "Background service"
URL="http://127.0.0.1:$PORT/"
if [ "$SERVICE" != no ] && command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1 \
   && { [ "$SERVICE" = yes ] || ask "Run Quarry in the background now and at login (systemd user service)?" y; }; then
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/quarry.service" <<EOF
[Unit]
Description=Quarry local LLM server
After=network.target

[Service]
EnvironmentFile=$CONF
ExecStart=$QDIR/build/quarry serve -m \${QUARRY_MODEL} --host 127.0.0.1 --port \${QUARRY_PORT} -c \${QUARRY_CTX}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  # a server started by hand on the same port would block the service
  if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && ! systemctl --user is-active -q quarry.service; then
    warn "something is already serving on port $PORT; stop it (e.g. pkill -x quarry) and run: systemctl --user restart quarry"
    systemctl --user enable -q quarry.service
  else
    systemctl --user enable -q --now quarry.service
    printf '   %sloading the model…%s' "$D" "$N"
    for _ in $(seq 1 180); do curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; printf '.'; done; echo
    curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && ok "running at $URL" || warn "not answering yet; check: journalctl --user -u quarry -f"
  fi
  SERVICE=yes
else
  SERVICE=no
  ok "skipped. Start it by hand: quarry serve -m \"$MODEL\" --port $PORT -c $CTX"
fi

# ---------- 7. Tailscale (other devices) ----------
step "Tailscale"
TS_URL=""
if ! command -v tailscale >/dev/null; then
  ok "not installed; skipped (Quarry stays on this machine)"
elif ! tailscale status >/dev/null 2>&1; then
  warn "installed but not logged in. After 'sudo tailscale up', run: tailscale serve --bg $PORT"
elif [ "$TAILSCALE" = yes ] || { [ "$TAILSCALE" = ask ] && ask "Share the chat with your tailnet (HTTPS, tailnet only)?" n; }; then
  if tailscale serve --bg "$PORT" >/dev/null 2>&1 || sudo tailscale serve --bg "$PORT" >/dev/null 2>&1; then
    TS_URL="https://$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')/"
    ok "available on your tailnet at $TS_URL"
  else
    warn "tailscale serve failed; run it yourself: sudo tailscale serve --bg $PORT"
  fi
else
  ok "skipped"
fi

# ---------- 8. opencode ----------
step "opencode"
OC="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json"
if [ "$OPENCODE" = yes ] || { [ "$OPENCODE" = ask ] && command -v opencode >/dev/null && ask "Add Quarry as a provider in opencode?" y; }; then
  mkdir -p "$(dirname "$OC")"
  [ -f "$OC" ] && cp "$OC" "$OC.bak.$(date +%s)"
  python3 - "$OC" "$PORT" "$CTX" "$(basename "$MODEL" .gguf)" <<'PY'
import json, os, sys
path, port, ctx, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = {}
if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f)
cfg.setdefault("$schema", "https://opencode.ai/config.json")
cfg.setdefault("provider", {})["quarry"] = {
    "npm": "@ai-sdk/openai-compatible",
    "name": "Quarry (local)",
    "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
    "models": {"quarry": {"name": f"{name} (Quarry)", "tool_call": True,
                          "limit": {"context": ctx, "output": 4096}}},
}
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
  ok "added provider 'quarry' to $OC (pick model quarry/quarry); backup saved next to it"
else
  ok "skipped"
fi

# ---------- done ----------
printf '\n%s%sQuarry is ready.%s\n' "$G" "$B" "$N"
[ "$SERVICE" = yes ] && echo "   Chat:      $URL"
[ -n "$TS_URL" ] && echo "   Tailnet:   $TS_URL"
echo "   API:       http://127.0.0.1:$PORT/v1  (OpenAI-compatible)"
echo "   Terminal:  quarry run -m \"$MODEL\""
echo "   Settings:  $CONF"
[ "$SERVICE" = yes ] && echo "   Logs:      journalctl --user -u quarry -f"
echo "   Remove:    ./install.sh --uninstall"
