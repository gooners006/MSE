#!/bin/sh

MAX_SIZE=$((100 * 1024 * 1024)) # 100 MB

git diff --cached --name-only --diff-filter=AM | while read -r file; do
    size=$(git cat-file -s :$file 2>/dev/null || echo 0)

    if [ "$size" -gt "$MAX_SIZE" ]; then
        echo "Commit blocked: '$file' exceeds 100 MB"
        exit 1
    fi
done

exit 0
