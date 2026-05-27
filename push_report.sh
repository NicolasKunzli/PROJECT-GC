#!/bin/bash
# push_report.sh — push report changes to GitHub (main) and directly to Overleaf
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

# ── 2. Push to Overleaf (fetch current state, apply changes, push back) ───
echo "→ Pushing to Overleaf..."
git fetch overleaf master:overleaf-master --quiet
git checkout overleaf-master --quiet
git checkout main -- report/report.tex report/references.bib
if git diff --cached --quiet; then
  echo "  (Overleaf already up to date)"
else
  git commit -m "$MSG" --quiet
  git push overleaf overleaf-master:master --quiet
fi
git checkout main --quiet
git branch -D overleaf-master --quiet

echo "✓ Done — Overleaf updated automatically."
