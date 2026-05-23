# Upstream Sync Notes

This fork uses upstream lesson content from Rohit Ghumare as the original base,
then adds the AI Engineering from Zero UI, Indonesian lesson files, and local
learning flow changes.

## Safe Update Flow

1. Fetch upstream without merging:

   ```bash
   git fetch upstream
   ```

2. Audit lesson-content drift:

   ```bash
   node scripts/upstream-lesson-audit.mjs
   ```

3. If upstream changes `phases/**/docs/en.md`, treat the matching
   `phases/**/docs/id.md` as stale until reviewed.

4. Merge on a branch, not directly into the published branch:

   ```bash
   git switch -c sync/upstream-lessons
   git merge upstream/main
   ```

5. Regenerate only missing or stale Indonesian lesson files. Keep technical
   terms such as `gradient`, `loss`, `tokenizer`, `attention`, `embedding`,
   `fine-tuning`, `pipeline`, and library/API names in English.

6. Validate before publishing:

   ```bash
   node scripts/i18n-status.mjs
   node --check site/i18n.js
   node --check site/app.js
   git diff --check
   ```

## Rule of Thumb

- `docs/en.md` follows upstream content when useful.
- `docs/id.md` belongs to this fork and should be reviewed after upstream
  English changes.
- `site/` contains AI Engineering from Zero product/UI changes. Do not blindly
  overwrite it during upstream sync.
