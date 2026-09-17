#!/usr/bin/env bash
# Package the Atari-100k figure bundle for Overleaf.
#   bash docs/figures/atari100k/make_overleaf_zip.sh
# -> docs/figures/atari100k_overleaf.zip.  Upload it as a NEW Overleaf project
# (main.tex compiles every figure), or copy its figures/atari100k/ folder into the
# thesis project and \input the files as figures/atari100k/README.md describes.
# Re-run after experiments/src/export_atari100k_pgfplots.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="$(cd "$HERE/.." && pwd)/atari100k_overleaf.zip"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/atari100k_overleaf.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

DEST="$STAGE/figures/atari100k"
mkdir -p "$DEST"
cp "$HERE"/atari100k_pgfplots_preamble.tex "$HERE"/fig_*.tex "$HERE"/README.md "$DEST/"
cp -r "$HERE/data" "$DEST/data"
# the notebook's own vector figures (poster bars, IQM, HNS per game, Hankel spectra),
# for \includegraphics where a matplotlib version is wanted
NBF="$REPO/experiments/atari/cached/atari100k_effrainbow_5seed_figures"
if compgen -G "$NBF/*.pdf" > /dev/null; then
    mkdir -p "$DEST/notebook_pdf"
    cp "$NBF"/*.pdf "$DEST/notebook_pdf/"
fi

cat > "$STAGE/main.tex" <<'TEX'
% Test document for the Atari-100k figures (pdfLaTeX).  To use them in the thesis,
% copy figures/atari100k/ into the thesis project, \input the preamble file in the
% preamble and the fig_*.tex files where they belong (see figures/atari100k/README.md).
% figures/atari100k/notebook_pdf/ holds the notebook's matplotlib figures as vector
% PDFs for \includegraphics.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\input{figures/atari100k/atari100k_pgfplots_preamble}   % sets \figdata = figures/atari100k/data
\begin{document}
\input{figures/atari100k/fig_model_free_bars}
\input{figures/atari100k/fig_delta_hns}
\input{figures/atari100k/fig_learning_curves_showcase}
\clearpage
\input{figures/atari100k/fig_learning_curves}
\end{document}
TEX

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)
echo "wrote $OUT: $(unzip -l "$OUT" | tail -1 | awk '{printf "%.1f MB, %s files", $1/1e6, $2}')"
