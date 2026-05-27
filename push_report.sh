#!/bin/bash
# push_report.sh — push report + figures to GitHub (main) and directly to Overleaf
#
# Usage:  bash push_report.sh ["commit message"]
#
# What it does:
#   1. Commits report/report.tex and report/references.bib to GitHub (main).
#   2. Fetches the current Overleaf state, applies report.tex, references.bib
#      AND the entire figure/ tree from main onto it, then pushes back to Overleaf.
#
# NOTE: any locally modified figures (shown by `git status`) must be committed
#   to main *before* running this script, otherwise the pre-commit versions are
#   what gets sent to Overleaf.  Run:
#       git add figure/ && git commit -m "Regenerate figures"
#   first if needed.
set -e

MSG="${1:-Update report}"

# ── 1. Push to GitHub main ─────────────────────────────────────────────────
echo "→ Pushing to GitHub (main)..."
git add report/report.tex report/references.bib
if git diff --cached --quiet; then
  echo "  (no changes to commit)"
else
  git commit -m "$MSG" -m "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  git push origin main
fi

# ── 2. Push to Overleaf (fetch current state, apply changes, push back) ────
# Figures are included so that Overleaf always compiles against the latest PNGs.
echo "→ Pushing to Overleaf (report + figures)..."
git fetch overleaf master:overleaf-master --quiet
git checkout overleaf-master --quiet
git checkout main -- report/report.tex report/references.bib figure/
if git diff --cached --quiet; then
  echo "  (Overleaf already up to date)"
else
  # Count changed files for a useful log line
  N=$(git diff --cached --name-only | wc -l | tr -d ' ')
  echo "  Committing $N changed file(s) to Overleaf..."
  git commit -m "$MSG" --quiet
  git push overleaf overleaf-master:master --quiet
fi
git checkout main --quiet
git branch -D overleaf-master --quiet

echo "✓ Done — Overleaf updated automatically."
