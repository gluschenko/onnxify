#!/usr/bin/env bash

set -euo pipefail

CONFIGURATION="Release"
PACK_OUTPUT="artifacts/nupkgs/onnxify-cli"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: ./install-onnxify-cli.sh [options]

Options:
  -c, --configuration <value>  Build configuration to pack. Default: Release
  -o, --pack-output <path>     Output directory for the packed NuGet package.
                               Default: artifacts/nupkgs/onnxify-cli
  -n, --dry-run                Print the commands without executing them.
  -h, --help                   Show this help message.
EOF
}

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

run_command() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
        return
    fi

    "$@"
}

get_first_xml_value() {
    local tag="$1"
    local file_path="$2"

    sed -n "s:.*<${tag}>\\([^<]*\\)</${tag}>.*:\\1:p" "$file_path" | head -n 1
}

test_global_tool_installed() {
    local package_id="$1"
    local tool_list

    tool_list="$(dotnet tool list --global)" || fail "Failed to query globally installed dotnet tools."

    awk 'NR > 2 { print $1 }' <<<"$tool_list" | grep -Fxq "$package_id"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--configuration)
            [[ $# -ge 2 ]] || fail "Missing value for $1"
            CONFIGURATION="$2"
            shift 2
            ;;
        -o|--pack-output)
            [[ $# -ge 2 ]] || fail "Missing value for $1"
            PACK_OUTPUT="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$SCRIPT_DIR/src/Onnxify.CLI/Onnxify.CLI.csproj"

[[ -f "$PROJECT_PATH" ]] || fail "Project file was not found: $PROJECT_PATH"
command -v dotnet >/dev/null 2>&1 || fail "The 'dotnet' command was not found in PATH."

if [[ "$PACK_OUTPUT" != /* ]]; then
    PACK_OUTPUT="$SCRIPT_DIR/$PACK_OUTPUT"
fi

PACKAGE_ID="$(get_first_xml_value "PackageId" "$PROJECT_PATH")"
if [[ -z "$PACKAGE_ID" ]]; then
    PACKAGE_ID="$(basename "${PROJECT_PATH%.csproj}")"
fi

PACKAGE_VERSION="$(get_first_xml_value "PackageVersion" "$PROJECT_PATH")"
if [[ -z "$PACKAGE_VERSION" ]]; then
    PACKAGE_VERSION="$(get_first_xml_value "Version" "$PROJECT_PATH")"
fi
if [[ -z "$PACKAGE_VERSION" ]]; then
    VERSION_PREFIX="$(get_first_xml_value "VersionPrefix" "$PROJECT_PATH")"
    VERSION_SUFFIX="$(get_first_xml_value "VersionSuffix" "$PROJECT_PATH")"
    if [[ -n "${VERSION_PREFIX:-}" ]]; then
        if [[ -n "${VERSION_SUFFIX:-}" ]]; then
            PACKAGE_VERSION="$VERSION_PREFIX-$VERSION_SUFFIX"
        else
            PACKAGE_VERSION="$VERSION_PREFIX"
        fi
    fi
fi

[[ -n "$PACKAGE_VERSION" ]] || fail "Could not determine package version from $PROJECT_PATH"

NUPKG_PATH="$PACK_OUTPUT/$PACKAGE_ID.$PACKAGE_VERSION.nupkg"

if [[ ! -d "$PACK_OUTPUT" ]]; then
    run_command mkdir -p "$PACK_OUTPUT"
fi

run_command \
    dotnet pack "$PROJECT_PATH" \
    --configuration "$CONFIGURATION" \
    --output "$PACK_OUTPUT" \
    --nologo

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '\nDry run complete.\n'
    exit 0
fi

[[ -f "$NUPKG_PATH" ]] || fail "Packed tool package was not found: $NUPKG_PATH"

if test_global_tool_installed "$PACKAGE_ID"; then
    run_command \
        dotnet tool update "$PACKAGE_ID" \
        --global \
        --add-source "$PACK_OUTPUT" \
        --version "$PACKAGE_VERSION" \
        --ignore-failed-sources
else
    run_command \
        dotnet tool install "$PACKAGE_ID" \
        --global \
        --add-source "$PACK_OUTPUT" \
        --version "$PACKAGE_VERSION" \
        --ignore-failed-sources
fi

printf '\nPacked package: %s\n' "$NUPKG_PATH"
printf 'Tool command: onnxify\n'
