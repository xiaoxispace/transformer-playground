#!/bin/sh
# download_shc.sh - Download the latest SHC XML for each play.
#
# Usage: ./download_shc.sh links.txt plays/
#
# For each play URL, fetch the page, find every SHC XML link, pick the
# latest one (highest leading SHC id number), and save it as:
#
#     <Title_With_Underscores>_<original_shc_filename>.xml
#
# Outputs three statistics at the end:
#   - plays processed
#   - files downloaded
#   - files that failed

if [ $# -ne 2 ]; then
    echo "Usage: $0 <links.txt> <output_dir>" >&2
    exit 1
fi

LINKS="$1"
OUTDIR="$2"

if [ ! -f "$LINKS" ]; then
    echo "Error: file '$LINKS' not found" >&2
    exit 1
fi

mkdir -p "$OUTDIR" || exit 1

TMP_PAGE=$(mktemp) || exit 1
TMP_SHCS=$(mktemp) || exit 1
TMP_TITLE=$(mktemp) || exit 1
trap 'rm -f "$TMP_PAGE" "$TMP_SHCS" "$TMP_TITLE"' EXIT INT TERM

processed=0
downloaded=0
failed=0

while IFS= read -r url; do
    [ -z "$url" ] && continue

    processed=$((processed + 1))
    slug=$(printf '%s' "$url" | sed -e 's#.*/##')
    printf 'Processing %s\n' "$slug"

    if ! curl -sL --fail -o "$TMP_PAGE" "$url"; then
        printf '  WARNING: failed to fetch %s\n' "$url" >&2
        failed=$((failed + 1))
        continue
    fi

    # Unescape entities (works whether the page is raw or browser-saved).
    sed -e 's/&lt;/</g' -e 's/&gt;/>/g' -e 's/&amp;/\&/g' "$TMP_PAGE" > "$TMP_TITLE.body"

    # Extract the play title from the page's <h1>...</h1>.
    # The play pages put the title in the header block inside an <h1>.
    title=$(sed -n 's/.*<h1>\([^<]*\)<\/h1>.*/\1/p' "$TMP_TITLE.body" | head -n 1)

    # Normalize title for use in a filename:
    #   - strip HTML entities that may remain (e.g. &#039;)
    #   - decode a few common ones
    #   - collapse anything non-alphanumeric into underscores
    #   - squeeze repeated underscores, trim leading/trailing underscores
    safe_title=$(printf '%s' "$title" \
        | sed -e "s/&#039;/'/g" -e "s/&amp;/\&/g" -e "s/&quot;/\"/g" \
        | sed -e 's/[^A-Za-z0-9]/_/g' \
        | sed -e 's/__*/_/g' -e 's/^_//' -e 's/_$//')

    # Fallback if extraction failed: use the slug.
    [ -z "$safe_title" ] && safe_title="$slug"

    # Collect all SHC xml links.
    grep -o 'https://emed\.folger\.edu/sites/default/files/shc/[0-9A-Za-z_-]*\.xml' \
        "$TMP_TITLE.body" | sort -u > "$TMP_SHCS"

    if [ ! -s "$TMP_SHCS" ]; then
        printf '  WARNING: no SHC encoding found for %s\n' "$slug" >&2
        failed=$((failed + 1))
        rm -f "$TMP_TITLE.body"
        continue
    fi

    # Latest = highest leading number (filenames look like NNN-AAAAAAA.xml).
    latest=$(sort -n "$TMP_SHCS" | tail -n 1)
    shc_name=$(printf '%s' "$latest" | sed -e 's#.*/##')

    outfile="$OUTDIR/${safe_title}_${shc_name}"

    printf '  Title: %s\n' "$title"
    printf '  SHC:   %s\n' "$shc_name"

    if curl -sL --fail -o "$outfile" "$latest"; then
        printf '  Saved -> %s\n' "$outfile"
        downloaded=$((downloaded + 1))
    else
        printf '  WARNING: failed to download %s\n' "$latest" >&2
        failed=$((failed + 1))
    fi

    rm -f "$TMP_TITLE.body"
done < "$LINKS"

printf '\n========================================\n'
printf 'Download statistics\n'
printf '========================================\n'
printf '  Plays processed : %d\n' "$processed"
printf '  Downloaded      : %d\n' "$downloaded"
printf '  Failed          : %d\n' "$failed"
printf '========================================\n'

[ "$failed" -eq 0 ]