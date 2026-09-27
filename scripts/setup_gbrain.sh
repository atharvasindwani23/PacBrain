#!/usr/bin/env bash
# Install the official project in this checkout, without changing shell profiles.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
task_runtime="$task_root/.runtime"
task_bun_version=1.3.11
task_gbrain_commit=e78f1c38b947b053f3a46881340f74f316be855a
task_brain_home=${GBRAIN_HOME:-"$task_root/data/gbrain-home"}
case "$(uname -sm)" in
  'Darwin arm64') task_asset=bun-darwin-aarch64 ;;
  'Darwin x86_64') task_asset=bun-darwin-x64 ;;
  'Linux x86_64') task_asset=bun-linux-x64-baseline ;;
  'Linux aarch64') task_asset=bun-linux-aarch64 ;;
  *) echo 'Unsupported platform: use Bun >=1.3.11 and the official GBrain source.' >&2; exit 1 ;;
esac
mkdir -p "$task_runtime/bun" "$task_runtime/gbrain" "$task_runtime/bin" "$task_brain_home"
task_brain_home=$(cd "$task_brain_home" && pwd -P)
task_bun="$task_runtime/bun/$task_asset/bun"
if [[ ! -x "$task_bun" ]]; then
  task_release="https://github.com/oven-sh/bun/releases/download/bun-v$task_bun_version"
  curl --fail --silent --show-error --location "$task_release/$task_asset.zip" --output "$task_runtime/bun/bun.zip"
  curl --fail --silent --show-error --location "$task_release/SHASUMS256.txt" --output "$task_runtime/bun/SHASUMS256.txt"
  python3 - "$task_runtime/bun" "$task_asset.zip" <<'PY'
import hashlib, sys
from pathlib import Path
root, name = Path(sys.argv[1]), sys.argv[2]
expected = next(row.split()[0] for row in (root/'SHASUMS256.txt').read_text().splitlines() if row.split()[1] == name)
actual = hashlib.sha256((root/'bun.zip').read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit('Bun checksum mismatch; refusing to install.')
PY
  unzip -qo "$task_runtime/bun/bun.zip" -d "$task_runtime/bun"
fi
if [[ ! -d "$task_runtime/gbrain/repo/.git" ]]; then
  git clone --depth 1 --branch latest-stable https://github.com/garrytan/gbrain.git "$task_runtime/gbrain/repo"
fi
if [[ "$(git -C "$task_runtime/gbrain/repo" rev-parse HEAD)" != "$task_gbrain_commit" ]]; then
  git -C "$task_runtime/gbrain/repo" fetch --depth 1 origin "$task_gbrain_commit"
  git -C "$task_runtime/gbrain/repo" checkout --detach "$task_gbrain_commit"
fi
(
  cd "$task_runtime/gbrain/repo"
  "$task_bun" install --frozen-lockfile --ignore-scripts --cache-dir "$task_runtime/gbrain/cache"
)
# Keep the launcher independent of whichever directory calls it.
cat > "$task_runtime/bin/gbrain" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
case "$(uname -sm)" in
  'Darwin arm64') task_asset=bun-darwin-aarch64 ;;
  'Darwin x86_64') task_asset=bun-darwin-x64 ;;
  'Linux x86_64') task_asset=bun-linux-x64-baseline ;;
  'Linux aarch64') task_asset=bun-linux-aarch64 ;;
esac
export GBRAIN_HOME=${GBRAIN_HOME:-"$task_root/data/gbrain-home"}
export GBRAIN_SKIP_STARTUP_HOOKS=1
unset DATABASE_URL GBRAIN_DATABASE_URL GBRAIN_SOURCE GBRAIN_BRAIN_ID
mkdir -p "$GBRAIN_HOME"
cd "$GBRAIN_HOME"
export GBRAIN_HOME="$PWD"
exec "$task_root/.runtime/bun/$task_asset/bun" "$task_root/.runtime/gbrain/repo/src/cli.ts" "$@"
SH
chmod +x "$task_runtime/bin/gbrain"
if [[ ! -f "$task_brain_home/.gbrain/config.json" ]]; then
  GBRAIN_HOME="$task_brain_home" "$task_runtime/bin/gbrain" init --pglite --no-embedding --non-interactive --db-only
fi
GBRAIN_HOME="$task_brain_home" "$task_runtime/bin/gbrain" --version
echo "GBRAIN_BIN=$task_runtime/bin/gbrain"
echo "GBRAIN_HOME=$task_brain_home"
echo 'Keyless mode: explicit memory and keyword retrieval work without API keys.'
