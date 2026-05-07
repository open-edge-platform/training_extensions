---
name: otx-docs-update
description: Update documentation (README, CHANGELOG, Sphinx docs, or inline docstrings) to reflect changes, fixes, or new features. Use when a PR modifies behavior that should be documented for users or developers.
---

# OTX Documentation Update

## Quick Start

- Identify affected documentation: `README.md`, `CHANGELOG.md`, `library/docs/source/*.rst`, or docstrings.
- For `CHANGELOG.md`, follow the existing format (e.g., [Keep a Changelog](https://keepachangelog.com/)).
- For Sphinx docs, work in `library/docs/source/`.
- Use clear, concrete wording. Avoid vague or marketing-heavy language.

## Workflow

1. **Analyze Changes**: Review the code changes or fixes to determine what needs to be documented.
2. **Locate Docs**: Find the relevant documentation files:
   - User-facing features: `README.md` or `library/docs/source/guide/`.
   - Technical details: `library/docs/source/explanation/` or inline docstrings.
   - Project history: `CHANGELOG.md`.
3. **Draft Updates**: Apply the documentation changes, matching the existing style and tone.
4. **Verify**:
   - For Markdown: Ensure formatting is consistent.
   - For Sphinx: (If dependencies are available) Run `sphinx-build -b html source build` in `library/docs/` to check for build errors.

## Style Guidelines

- Use active voice and concrete examples.
- Match the existing frequency of comments and documentation.
- Ensure all technical terms match the actual implementation.

## Coordination Notes

- Update `CHANGELOG.md` in the same PR as the code changes.
- If a change affects multiple areas (e.g., Backend and UI), ensure documentation reflects the end-to-end impact.
