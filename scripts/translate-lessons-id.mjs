#!/usr/bin/env node
/**
 * Batch-create Indonesian lesson files.
 *
 * This script is intentionally provider-based. It preserves fenced code blocks
 * and inline code, and tells the translator to keep technical AI/programming
 * terms in English.
 *
 * Usage:
 *   OPENAI_API_KEY=... node scripts/translate-lessons-id.mjs --limit 5
 *   OPENAI_API_KEY=... node scripts/translate-lessons-id.mjs --all
 */
import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const phasesDir = path.join(root, 'phases');
const args = new Set(process.argv.slice(2));
const limitArg = process.argv.find(arg => arg.startsWith('--limit='));
const limit = args.has('--all') ? Infinity : Number(limitArg?.split('=')[1] || 5);

const technicalTerms = [
  'AI', 'API', 'Python', 'TypeScript', 'Rust', 'Julia', 'Node.js', 'CUDA',
  'MPS', 'GPU', 'CPU', 'uv', 'pnpm', 'cargo', 'juliaup', 'pip', 'NumPy',
  'PyTorch', 'JAX', 'transformers', 'virtual environment', 'package manager',
  'toolchain', 'runtime', 'framework', 'code', 'script', 'terminal', 'shell',
  'vector', 'matrix', 'matrix multiplication', 'dot product', 'shape',
  'forward pass', 'neural network', 'gradient', 'broadcasting',
];

function walk(dir, out = []) {
  return fs.readdir(dir, { withFileTypes: true }).then(async entries => {
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full, out);
      else out.push(full);
    }
    return out;
  });
}

function protectMarkdown(markdown) {
  const protectedBlocks = [];
  const protectedText = markdown
    .replace(/```[\s\S]*?```/g, match => {
      const token = `@@PROTECTED_BLOCK_${protectedBlocks.length}@@`;
      protectedBlocks.push(match);
      return token;
    })
    .replace(/`[^`\n]+`/g, match => {
      const token = `@@PROTECTED_BLOCK_${protectedBlocks.length}@@`;
      protectedBlocks.push(match);
      return token;
    });
  return { protectedText, protectedBlocks };
}

function restoreMarkdown(text, protectedBlocks) {
  return text.replace(/@@PROTECTED_BLOCK_(\d+)@@/g, (_, index) => protectedBlocks[Number(index)] || '');
}

async function translateWithOpenAI(markdown, relPath) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY is required.');

  const { protectedText, protectedBlocks } = protectMarkdown(markdown);
  const prompt = [
    'Translate this lesson Markdown from English to Indonesian.',
    'Keep the Markdown structure exactly.',
    'Do not translate fenced code blocks, inline code placeholders, URLs, package names, commands, API names, or math symbols.',
    'Keep these technical terms in English unless Indonesian wording is clearly more natural:',
    technicalTerms.join(', '),
    'Use natural Indonesian for explanations. It is okay to keep mixed English-Indonesian technical phrasing for clarity.',
    `Lesson path: ${relPath}`,
    '',
    protectedText,
  ].join('\n');

  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: process.env.OPENAI_TRANSLATION_MODEL || 'gpt-4.1-mini',
      input: prompt,
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API failed ${response.status}: ${await response.text()}`);
  }
  const data = await response.json();
  const text = data.output_text || data.output?.flatMap(item => item.content || []).map(c => c.text || '').join('') || '';
  if (!text.trim()) throw new Error('Empty translation response.');
  return restoreMarkdown(text.trim() + '\n', protectedBlocks);
}

const docs = (await walk(phasesDir))
  .filter(file => file.endsWith(path.join('docs', 'en.md')))
  .sort();

let created = 0;
for (const enPath of docs) {
  if (created >= limit) break;
  const idPath = enPath.replace(path.join('docs', 'en.md'), path.join('docs', 'id.md'));
  try {
    await fs.access(idPath);
    continue;
  } catch {}

  const relLesson = path.relative(root, path.dirname(path.dirname(enPath)));
  const markdown = await fs.readFile(enPath, 'utf8');
  console.log(`Translating ${relLesson}`);
  const translated = await translateWithOpenAI(markdown, relLesson);
  await fs.writeFile(idPath, translated, 'utf8');
  created++;
}

console.log(`Created ${created} Indonesian lesson file(s).`);
