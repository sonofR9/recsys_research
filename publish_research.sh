#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--dry-run|--apply|--push] [--destination PATH] [--message TEXT]"
}

source_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
destination="$source_directory/../recsys_research"
mode=dry-run
message="Publish research snapshot $(date --iso-8601=seconds)"

while (($#)); do
    case "$1" in
        --dry-run)
            mode=dry-run
            shift
            ;;
        --apply)
            mode=apply
            shift
            ;;
        --push)
            mode=push
            shift
            ;;
        --destination)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            destination=$2
            shift 2
            ;;
        --message)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            message=$2
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }
git -C "$source_directory" rev-parse --show-toplevel >/dev/null 2>&1 || {
    echo "Source is not inside a Git repository: $source_directory" >&2
    exit 1
}
[[ -f "$source_directory/.publishignore" ]] || { echo "Missing $source_directory/.publishignore" >&2; exit 1; }
[[ -d "$destination/.git" ]] || { echo "Destination is not a Git repository: $destination" >&2; exit 1; }

destination=$(cd -- "$destination" && pwd -P)
[[ "$destination" != "$source_directory" ]] || { echo "Source and destination must differ" >&2; exit 1; }
case "$destination/" in
    "$source_directory/"*)
        echo "Destination cannot be inside the source: $destination" >&2
        exit 1
        ;;
esac
case "$source_directory/" in
    "$destination/"*)
        echo "Destination cannot contain the source: $destination" >&2
        exit 1
        ;;
esac

if [[ "$mode" != dry-run ]]; then
    if ! destination_status=$(git -C "$destination" status --porcelain --untracked-files=all); then
        echo "Cannot read destination Git status" >&2
        exit 1
    fi
    if [[ -n "$destination_status" ]]; then
        echo "Destination has changes; commit, remove, or ignore them first" >&2
        exit 1
    fi
fi

publish_excluded() {
    local relative_path=$1
    local pattern
    while IFS= read -r pattern; do
        pattern=${pattern%$'\r'}
        [[ -z "$pattern" || "$pattern" == \#* ]] && continue
        anchored=false
        if [[ "$pattern" == /* ]]; then
            anchored=true
            pattern=${pattern#/}
        fi
        directory_pattern=false
        if [[ "$pattern" == */ ]]; then
            directory_pattern=true
            pattern=${pattern%/}
        fi
        if [[ "$directory_pattern" == true ]]; then
            [[ "$relative_path" == $pattern/* ]] && return 0
            [[ "$anchored" == false && "$relative_path" == */$pattern/* ]] && return 0
        else
            [[ "$relative_path" == $pattern ]] && return 0
            [[ "$relative_path" == $pattern/* ]] && return 0
            if [[ "$anchored" == false ]]; then
                [[ "$relative_path" == */$pattern ]] && return 0
                [[ "$relative_path" == */$pattern/* ]] && return 0
            fi
        fi
    done < "$source_directory/.publishignore"
    return 1
}

temporary_directory=$(mktemp -d "$destination/.git/publish.XXXXXX")
trap 'rm -rf -- "$temporary_directory"' EXIT
if ! git -C "$source_directory" ls-files --cached --others --exclude-standard -z -- . \
    > "$temporary_directory/candidates"; then
    echo "Cannot enumerate source files" >&2
    exit 1
fi
mapfile -d '' candidates < "$temporary_directory/candidates"
source_files=()
declare -A source_file_set=()
for relative_path in "${candidates[@]}"; do
    [[ -e "$source_directory/$relative_path" || -L "$source_directory/$relative_path" ]] || continue
    publish_excluded "$relative_path" && continue
    source_files+=("$relative_path")
    source_file_set["$relative_path"]=1
done

: > "$temporary_directory/manifest"
for relative_path in "${source_files[@]}"; do
    printf '%s\0' "$relative_path" >> "$temporary_directory/manifest"
done

if ! git -C "$destination" ls-files -z > "$temporary_directory/destination"; then
    echo "Cannot enumerate destination files" >&2
    exit 1
fi
mapfile -d '' destination_files < "$temporary_directory/destination"
stale_files=()
for relative_path in "${destination_files[@]}"; do
    [[ -n ${source_file_set["$relative_path"]+present} ]] || stale_files+=("$relative_path")
done

rsync_arguments=(
    --archive
    --itemize-changes
    --omit-dir-times
    --from0
    --files-from="$temporary_directory/manifest"
)

for relative_path in "${stale_files[@]}"; do
    echo "*deleting $relative_path"
done

if [[ "$mode" == dry-run ]]; then
    for relative_path in "${source_files[@]}"; do
        source_path="$source_directory/$relative_path"
        destination_path="$destination/$relative_path"
        if [[ -L "$source_path" ]]; then
            [[ -L "$destination_path" ]] && \
                [[ $(readlink -- "$source_path") == $(readlink -- "$destination_path") ]] && continue
        elif [[ -f "$source_path" ]]; then
            if [[ -f "$destination_path" && ! -L "$destination_path" ]] && \
                cmp --silent -- "$source_path" "$destination_path"; then
                source_mode=$(stat --format=%a -- "$source_path")
                destination_mode=$(stat --format=%a -- "$destination_path")
                [[ "$source_mode" == "$destination_mode" ]] && continue
            fi
        fi
        printf '> %q\n' "$relative_path"
    done
    echo "Dry run only. Use --apply to copy or --push to copy, commit, and push."
    exit 0
fi

if ((${#stale_files[@]})); then
    git -C "$destination" rm --quiet -- "${stale_files[@]}"
    for relative_path in "${stale_files[@]}"; do
        parent_directory=$(dirname -- "$relative_path")
        while [[ "$parent_directory" != . ]]; do
            rmdir -- "$destination/$parent_directory" 2>/dev/null || break
            parent_directory=$(dirname -- "$parent_directory")
        done
    done
fi
rsync "${rsync_arguments[@]}" "$source_directory/" "$destination/"

if [[ "$mode" == apply ]]; then
    git -C "$destination" status --short
    exit 0
fi

git -C "$destination" remote get-url origin >/dev/null
git -C "$destination" add --all
git -C "$destination" add --force \
    --pathspec-from-file="$temporary_directory/manifest" --pathspec-file-nul
if ! git -C "$destination" diff --cached --quiet; then
    git -C "$destination" commit -m "$message"
fi
branch=$(git -C "$destination" symbolic-ref --short HEAD)
git -C "$destination" push --set-upstream origin "$branch"
