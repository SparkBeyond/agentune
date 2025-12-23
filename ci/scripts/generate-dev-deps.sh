#!/usr/bin/env bash
set -euo pipefail

PYPROJECT="pyproject.toml"
OUTPUT="requirements-dev.txt"

# Clear the output file
> "$OUTPUT"

in_dev_group=0
while IFS= read -r line; do
    # Trim leading/trailing spaces
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    # Detect start and end of dev group
    if [[ "$line" == "[tool.poetry.group.dev.dependencies]" ]]; then
        in_dev_group=1
        continue
    elif [[ "$line" =~ ^\[.*\]$ ]]; then
        in_dev_group=0
    fi

    if [[ $in_dev_group -eq 1 ]] && [[ "$line" =~ ^[^#]*= ]]; then
        # Split into package and version
        pkg="${line%%=*}"
        ver="${line#*=}"

        # Remove spaces
        pkg="${pkg%"${pkg##*[![:space:]]}"}"
        pkg="${pkg#"${pkg%%[![:space:]]*}"}"
        ver="${ver%"${ver##*[![:space:]]}"}"
        ver="${ver#"${ver%%[![:space:]]*}"}"

        # Remove quotes
        ver="${ver%\"}"
        ver="${ver#\"}"

        # Convert caret or >= to exact
        ver="${ver#^}"
        ver="${ver#>=}"

        echo "$pkg==$ver" >> "$OUTPUT"
    fi
done < "$PYPROJECT"

echo "Generated $OUTPUT:"
cat "$OUTPUT"

