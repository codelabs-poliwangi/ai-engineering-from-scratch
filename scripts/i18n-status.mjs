#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const phasesDir = path.join(root, 'phases');

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

const englishDocs = walk(phasesDir)
  .filter(file => file.endsWith(path.join('docs', 'en.md')))
  .sort();

const rows = englishDocs.map(enPath => {
  const idPath = enPath.replace(path.join('docs', 'en.md'), path.join('docs', 'id.md'));
  const relLesson = path.relative(root, path.dirname(path.dirname(enPath)));
  return {
    lesson: relLesson,
    en: path.relative(root, enPath),
    id: path.relative(root, idPath),
    hasId: fs.existsSync(idPath),
  };
});

const missing = rows.filter(row => !row.hasId);
const existing = rows.filter(row => row.hasId);

console.log(`English docs : ${rows.length}`);
console.log(`Indonesian   : ${existing.length}`);
console.log(`Missing ID   : ${missing.length}`);
console.log('');

if (missing.length) {
  console.log('Missing docs/id.md:');
  for (const row of missing) console.log(`- ${row.lesson}`);
}
