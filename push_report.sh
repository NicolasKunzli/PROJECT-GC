#!/bin/bash
# push_report.sh — push report changes to both main and the Overleaf-linked report branch

set -e

echo "→ Pushing to main..."
git add report/report.tex report/references.bib
git diff --cached --quiet && echo "  (no changes to commit)" || \
  git commit -m "${1:-Update report}" \
    -m "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin main

echo "→ Syncing report branch..."
git show HEAD:report/report.tex   > /tmp/_report.tex
git show HEAD:report/references.bib > /tmp/_references.bib

git fetch origin report
git checkout report
cp /tmp/_report.tex   report.tex
cp /tmp/_references.bib references.bib
git add report.tex references.bib
git diff --cached --quiet && echo "  (already up to date)" || \
  git commit -m "${1:-Update report}"
git push origin report

git checkout main
rm -f /tmp/_report.tex /tmp/_references.bib
echo "✓ Done — pull from the report branch on Overleaf."
