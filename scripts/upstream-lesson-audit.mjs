#!/usr/bin/env node
/**
 * Audit lesson-content drift before merging upstream changes.
 *
 * This script is intentionally read-only. Run it after `git fetch upstream` to
 * see which upstream English lesson files changed and which Indonesian lesson
 * files may need regeneration or review.
 *
 * Usage:
 *   git fetch upstream
 *   node scripts/upstream-lesson-audit.mjs
 *   node scripts/upstream-lesson-audit.mjs --base=main --target=upstream/main
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const args = Object.fromEntries(
  process.argv.slice(2).map(arg => {
    const [key, value = ''] = arg.replace(/^--/, '').split('=');
    return [key, value];
  }),
);

const base = args.base || 'HEAD';
const target = args.target || 'upstream/main';

function git(args, options = {}) {
  return execFileSync('git', args, {
    cwd: root,
    encoding: 'utf8',
    stdio: options.stdio || ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function refExists(ref) {
  try {
    git(['rev-parse', '--verify', '--quiet', ref]);
    return true;
  } catch {
    return false;
  }
}

function fileExistsAtRef(ref, file) {
  try {
    git(['cat-file', '-e', `${ref}:${file}`]);
    return true;
  } catch {
    return false;
  }
}

function objectHash(ref, file) {
  try {
    return git(['rev-parse', `${ref}:${file}`]);
  } catch {
    return '';
  }
}

function localExists(file) {
  return fs.existsSync(path.join(root, file));
}

function lessonDirFromEn(enFile) {
  return enFile.replace(/\/docs\/en\.md$/, '');
}

if (!refExists(target)) {
  console.error(`Missing git ref: ${target}`);
  console.error('Run `git fetch upstream` first, or pass --target=<ref>.');
  process.exit(2);
}

const diffOutput = git([
  'diff',
  '--name-status',
  `${base}..${target}`,
  '--',
  'phases',
]);

const rows = diffOutput
  ? diffOutput.split('\n').map(line => line.split(/\t+/))
  : [];

const englishChanges = rows
  .filter(parts => parts[1]?.endsWith('/docs/en.md'))
  .map(parts => {
    const status = parts[0];
    const en = parts[1];
    const lesson = lessonDirFromEn(en);
    const id = `${lesson}/docs/id.md`;
    const currentHasId = localExists(id);
    const targetHasEn = fileExistsAtRef(target, en);
    const currentHasEn = fileExistsAtRef(base, en);
    const hashChanged = currentHasEn && targetHasEn
      ? objectHash(base, en) !== objectHash(target, en)
      : true;
    return { status, lesson, en, id, currentHasId, hashChanged };
  });

const added = englishChanges.filter(row => row.status.startsWith('A'));
const modified = englishChanges.filter(row => row.status.startsWith('M'));
const deleted = englishChanges.filter(row => row.status.startsWith('D'));
const missingId = englishChanges.filter(row => !row.currentHasId && !row.status.startsWith('D'));

console.log(`Base   : ${base}`);
console.log(`Target : ${target}`);
console.log('');
console.log(`English lesson docs changed : ${englishChanges.length}`);
console.log(`Added upstream lessons       : ${added.length}`);
console.log(`Modified upstream lessons    : ${modified.length}`);
console.log(`Deleted upstream lessons     : ${deleted.length}`);
console.log(`Missing local docs/id.md     : ${missingId.length}`);
console.log('');

if (!englishChanges.length) {
  console.log('No upstream English lesson content changes detected.');
  process.exit(0);
}

console.log('Review list:');
for (const row of englishChanges) {
  const flags = [];
  if (!row.currentHasId && !row.status.startsWith('D')) flags.push('needs-id');
  if (row.status.startsWith('M') && row.currentHasId) flags.push('id-review');
  if (row.status.startsWith('D')) flags.push('upstream-deleted');
  console.log(`- [${row.status}] ${row.lesson}${flags.length ? ` (${flags.join(', ')})` : ''}`);
}

console.log('');
console.log('Suggested safe flow:');
console.log('1. Merge or cherry-pick upstream changes on a branch.');
console.log('2. For added lessons, generate Indonesian docs with scripts/translate-lessons-id.mjs.');
console.log('3. For modified lessons, review the matching docs/id.md before publishing.');
console.log('4. Run node scripts/i18n-status.mjs and git diff --check before commit.');
