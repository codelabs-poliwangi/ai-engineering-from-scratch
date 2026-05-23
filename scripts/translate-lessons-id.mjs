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
 *   node scripts/translate-lessons-id.mjs --provider=google --limit=5
 *   node scripts/translate-lessons-id.mjs --polish-existing
 */
import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(new URL('..', import.meta.url).pathname);
const phasesDir = path.join(root, 'phases');
const args = new Set(process.argv.slice(2));
const limitArg = process.argv.find(arg => arg.startsWith('--limit='));
const providerArg = process.argv.find(arg => arg.startsWith('--provider='));
const limit = args.has('--all') ? Infinity : Number(limitArg?.split('=')[1] || 5);
const provider = providerArg?.split('=')[1] || (process.env.OPENAI_API_KEY ? 'openai' : 'google');

const technicalTerms = [
  'AI', 'API', 'Python', 'TypeScript', 'Rust', 'Julia', 'Node.js', 'CUDA',
  'MPS', 'GPU', 'CPU', 'uv', 'pnpm', 'cargo', 'juliaup', 'pip', 'NumPy',
  'PyTorch', 'JAX', 'transformers', 'virtual environment', 'package manager',
  'toolchain', 'runtime', 'framework', 'code', 'script', 'terminal', 'shell',
  'vector', 'matrix', 'matrix multiplication', 'dot product', 'shape',
  'forward pass', 'neural network', 'gradient', 'broadcasting',
  'loss', 'optimizer', 'learning rate', 'weight', 'bias', 'feature',
  'dataset', 'sample', 'batch', 'mini-batch', 'training', 'inference',
  'embedding', 'attention', 'Transformer', 'tokenizer', 'tokenization',
  'fine-tuning', 'prompt', 'system prompt', 'pipeline', 'input', 'output',
  'dimensionality reduction', 'eigenvalue', 'eigenvector',
  'eigendecomposition', 'covariance matrix', 'principal component',
  'explained variance ratio', 'reconstruction error', 'nearest neighbor',
  'perplexity', 'manifold', 'kernel', 'kernel function',
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

function splitForTranslate(text, maxLength = 3500) {
  const chunks = [];
  let current = '';
  for (const part of text.split(/(\n{2,})/)) {
    if ((current + part).length > maxLength && current.trim()) {
      chunks.push(current);
      current = '';
    }
    if (part.length > maxLength) {
      for (let i = 0; i < part.length; i += maxLength) {
        chunks.push(part.slice(i, i + maxLength));
      }
    } else {
      current += part;
    }
  }
  if (current.trim()) chunks.push(current);
  return chunks;
}

function tidyTechnicalTerms(text) {
  const replacements = [
    [/\bAnda\b/g, 'kamu'],
    [/\bPelajaran\b/g, 'Lesson'],
    [/\bpelajaran\b/g, 'lesson'],
    [/\bTahap\b/g, 'Phase'],
    [/\btahap\b/g, 'phase'],
    [/\bBahasa:\b/g, 'Language:'],
    [/\bJenis:\b/g, 'Type:'],
    [/\bPrasyarat:\b/g, 'Prerequisites:'],
    [/\*\*Bahasa:\*\*/g, '**Language:**'],
    [/\*\*Jenis:\*\*/g, '**Type:**'],
    [/\*\*Prasyarat:\*\*/g, '**Prerequisites:**'],
    [/\bTujuan Pembelajaran\b/g, 'Tujuan Pembelajaran'],
    [/\bMasalahnya\b/g, 'Masalah'],
    [/\bBangun\b/g, 'Build'],
    [/\bPelajari\b/g, 'Learn'],
    [/\bGunakan [Ii]tu\b/g, 'Pakai'],
    [/\bMembangunnya\b/g, 'Bangun'],
    [/\bBangun Itu\b/g, 'Bangun'],
    [/\bKirimkan\b/g, 'Kirim'],
    [/## KirimPelajaran/g, '## Kirim\n\nLesson'],
    [/## KirimkanPelajaran/g, '## Kirim\n\nLesson'],
    [/## Gunakan ituLihat/g, '## Pakai\n\nLihat'],
    [/## Gunakan itu/g, '## Pakai\n\n'],
    [/## PakaiLihat/g, '## Pakai\n\nLihat'],
    [/\bKetentuan Utama\b/g, 'Istilah Kunci'],
    [/\bBacaan Lebih Lanjut\b/g, 'Bacaan Lanjutan'],
    [/\bfungsi kerugian\b/g, 'loss function'],
    [/\bfungsi kerugiannya\b/g, 'loss function-nya'],
    [/\bFungsi kerugian\b/g, 'Loss function'],
    [/\blanskap kerugian\b/g, 'loss landscape'],
    [/\bLanskap kerugian\b/g, 'Loss landscape'],
    [/\bnilai kerugian\b/g, 'nilai loss'],
    [/\bkerugian\b/g, 'loss'],
    [/\bKerugian\b/g, 'Loss'],
    [/\bkecepatan pemelajaran\b/g, 'learning rate'],
    [/\bKecepatan pemelajaran\b/g, 'Learning rate'],
    [/\bkecepatan pembelajaran\b/g, 'learning rate'],
    [/\bKecepatan pembelajaran\b/g, 'Learning rate'],
    [/\bkecepatan pembelajarannya\b/g, 'learning rate-nya'],
    [/\bKecepatan pembelajarannya\b/g, 'Learning rate-nya'],
    [/\bjadwal learning rate\b/g, 'learning rate schedule'],
    [/\bJadwal learning rate\b/g, 'Learning rate schedule'],
    [/\bpenurunan gradient\b/g, 'gradient descent'],
    [/\bPenurunan gradient\b/g, 'Gradient descent'],
    [/\bpenurunan gradien\b/g, 'gradient descent'],
    [/\bPenurunan gradien\b/g, 'Gradient descent'],
    [/\bgradien\b/g, 'gradient'],
    [/\bGradien\b/g, 'Gradient'],
    [/\bpengoptimal\b/g, 'optimizer'],
    [/\bPengoptimal\b/g, 'Optimizer'],
    [/\bpengoptimalan\b/g, 'optimization'],
    [/\bPengoptimalan\b/g, 'Optimization'],
    [/\bpengoptimalannya\b/g, 'optimization-nya'],
    [/\bPengoptimalannya\b/g, 'Optimization-nya'],
    [/\bpengoptimalnya\b/g, 'optimizer-nya'],
    [/\bPengoptimalnya\b/g, 'Optimizer-nya'],
    [/\boptimasi\b/g, 'optimization'],
    [/\bOptimasi\b/g, 'Optimization'],
    [/\bbobot\b/g, 'weight'],
    [/\bBobot\b/g, 'Weight'],
    [/\bbeban\b/g, 'weight'],
    [/\bBeban\b/g, 'Weight'],
    [/\bkumpulan data\b/g, 'dataset'],
    [/\bKumpulan data\b/g, 'Dataset'],
    [/\bsubkumpulan\b/g, 'subset'],
    [/\bSubkumpulan\b/g, 'Subset'],
    [/\bKelompok mini\b/g, 'Mini-batch'],
    [/\bkelompok mini\b/g, 'mini-batch'],
    [/\bAngkatan GD\b/g, 'Batch GD'],
    [/\bSet data keseluruhan\b/g, 'Seluruh dataset'],
    [/\bSeluruh kumpulan data\b/g, 'Seluruh dataset'],
    [/\bTepat\b/g, 'Akurat'],
    [/\bmelampaui batas\b/g, 'overshoot'],
    [/\bterlalu besar menyebabkan perbedaan\b/g, 'terlalu besar membuat training diverge'],
    [/\bKomputasi limbah yang terlalu kecil\b/g, 'terlalu kecil membuang compute'],
    [/\bKecerdasan Buatan\b/g, 'AI'],
    [/\bkecerdasan buatan\b/g, 'AI'],
    [/\bpembelajaran mesin\b/g, 'machine learning'],
    [/\bPembelajaran Mesin\b/g, 'Machine Learning'],
    [/\bjaringan saraf\b/g, 'neural network'],
    [/\bJaringan Saraf\b/g, 'Neural Network'],
    [/\bgradien\b/g, 'gradient'],
    [/\bGradien\b/g, 'Gradient'],
    [/\bvektor\b/g, 'vector'],
    [/\bVektor\b/g, 'Vector'],
    [/\bmatriks\b/g, 'matrix'],
    [/\bMatriks\b/g, 'Matrix'],
    [/\blapisan\b/g, 'layer'],
    [/\bLapisan\b/g, 'Layer'],
    [/\bkerangka kerja\b/g, 'framework'],
    [/\bKerangka Kerja\b/g, 'Framework'],
    [/\bkode\b/g, 'code'],
    [/\bKode\b/g, 'Code'],
    [/\bmodel bahasa besar\b/g, 'large language model'],
    [/\bModel Bahasa Besar\b/g, 'Large Language Model'],
    [/\bPengurangan Dimensi\b/g, 'Dimensionality Reduction'],
    [/\bpengurangan dimensi\b/g, 'dimensionality reduction'],
    [/\bPengurangan dimension\b/g, 'Dimensionality Reduction'],
    [/\bpengurangan dimension\b/g, 'dimensionality reduction'],
    [/\breduksi dimensi\b/g, 'dimensionality reduction'],
    [/\bReduksi Dimensi\b/g, 'Dimensionality Reduction'],
    [/\breduksi dimension\b/g, 'dimensionality reduction'],
    [/\bReduksi Dimension\b/g, 'Dimensionality Reduction'],
    [/\bKutukan dimension\b/g, 'Curse of dimensionality'],
    [/\bkutukan dimension\b/g, 'curse of dimensionality'],
    [/\bberdimension tinggi\b/g, 'high-dimensional'],
    [/\bBerdimension tinggi\b/g, 'High-dimensional'],
    [/\bdimension tinggi\b/g, 'high-dimensional'],
    [/\bDimension tinggi\b/g, 'High-dimensional'],
    [/\bdimensi\b/g, 'dimension'],
    [/\bDimensi\b/g, 'Dimension'],
    [/\bfitur\b/g, 'feature'],
    [/\bFitur\b/g, 'Feature'],
    [/\bsampel\b/g, 'sample'],
    [/\bSampel\b/g, 'Sample'],
    [/\bmasukan\b/g, 'input'],
    [/\bMasukan\b/g, 'Input'],
    [/\bkeluaran\b/g, 'output'],
    [/\bKeluaran\b/g, 'Output'],
    [/\bpelatihan\b/g, 'training'],
    [/\bPelatihan\b/g, 'Training'],
    [/\bpelatihannya\b/g, 'training-nya'],
    [/\bPelatihannya\b/g, 'Training-nya'],
    [/\bdata training\b/g, 'training data'],
    [/\bData training\b/g, 'Training data'],
    [/\binferensi\b/g, 'inference'],
    [/\bInferensi\b/g, 'Inference'],
    [/\baktivasi\b/g, 'activation'],
    [/\bAktivasi\b/g, 'Activation'],
    [/\bbias\b/g, 'bias'],
    [/\bBias\b/g, 'Bias'],
    [/\bparameter\b/g, 'parameter'],
    [/\bParameter\b/g, 'Parameter'],
    [/\bskalar\b/g, 'scalar'],
    [/\bSkalar\b/g, 'Scalar'],
    [/\btranspose\b/g, 'transpose'],
    [/\bTranspose\b/g, 'Transpose'],
    [/\bdeterminan\b/g, 'determinant'],
    [/\bDeterminan\b/g, 'Determinant'],
    [/\binvers\b/g, 'inverse'],
    [/\bInvers\b/g, 'Inverse'],
    [/\bnorma\b/g, 'norm'],
    [/\bNorma\b/g, 'Norm'],
    [/\bjarak\b/g, 'distance'],
    [/\bJarak\b/g, 'Distance'],
    [/\btetangga terdekat\b/g, 'nearest neighbor'],
    [/\bTetangga terdekat\b/g, 'Nearest neighbor'],
    [/\bpenyematan\b/g, 'embedding'],
    [/\bPenyematan\b/g, 'Embedding'],
    [/\bpenyematannya\b/g, 'embedding-nya'],
    [/\bPenyematannya\b/g, 'Embedding-nya'],
    [/\bmenyematkan\b/g, 'embed'],
    [/\bMenyematkan\b/g, 'Embed'],
    [/\bdisematkan\b/g, 'di-embed'],
    [/\bDisematkan\b/g, 'Di-embed'],
    [/\bperhatian\b/g, 'attention'],
    [/\bPerhatian\b/g, 'Attention'],
    [/\bperhatiannya\b/g, 'attention-nya'],
    [/\bPerhatiannya\b/g, 'Attention-nya'],
    [/\bmasker attention\b/g, 'attention mask'],
    [/\bMasker attention\b/g, 'Attention mask'],
    [/\btransformator\b/g, 'Transformer'],
    [/\bTransformator\b/g, 'Transformer'],
    [/\btransformatornya\b/g, 'Transformer-nya'],
    [/\bTransformatornya\b/g, 'Transformer-nya'],
    [/\btokenisasi\b/g, 'tokenization'],
    [/\bTokenisasi\b/g, 'Tokenization'],
    [/\bpenyetelan halus\b/g, 'fine-tuning'],
    [/\bPenyetelan halus\b/g, 'Fine-tuning'],
    [/\bperintah sistem\b/g, 'system prompt'],
    [/\bPerintah sistem\b/g, 'System prompt'],
    [/\bperintah\b/g, 'prompt'],
    [/\bPerintah\b/g, 'Prompt'],
    [/\bsaluran\b/g, 'pipeline'],
    [/\bSaluran\b/g, 'Pipeline'],
    [/\bNilai Eigen\b/g, 'Eigenvalue'],
    [/\bnilai eigen\b/g, 'eigenvalue'],
    [/\bNilai eigen\b/g, 'Eigenvalue'],
    [/\beigenvalue terbesar\b/g, 'largest eigenvalue'],
    [/\bVector Eigen\b/g, 'Eigenvector'],
    [/\bvector eigen\b/g, 'eigenvector'],
    [/\bVektor Eigen\b/g, 'Eigenvector'],
    [/\bvektor eigen\b/g, 'eigenvector'],
    [/\bVector eigen\b/g, 'Eigenvector'],
    [/\bvector Eigen\b/g, 'eigenvector'],
    [/\bVector eigennya\b/g, 'Eigenvector-nya'],
    [/\bvector eigennya\b/g, 'eigenvector-nya'],
    [/\bdekomposisi eigend\b/g, 'eigendecomposition'],
    [/\bDekomposisi eigend\b/g, 'Eigendecomposition'],
    [/\bdekomposisi eigen\b/g, 'eigendecomposition'],
    [/\bDekomposisi eigen\b/g, 'Eigendecomposition'],
    [/\bmatrix kovarians\b/g, 'covariance matrix'],
    [/\bMatrix kovarians\b/g, 'Covariance matrix'],
    [/\bMatrix kovariansnya\b/g, 'Covariance matrix-nya'],
    [/\bmatrix kovariansnya\b/g, 'covariance matrix-nya'],
    [/\bmatriks kovarians\b/g, 'covariance matrix'],
    [/\bMatriks kovarians\b/g, 'Covariance matrix'],
    [/\bMatriks kovariansnya\b/g, 'Covariance matrix-nya'],
    [/\bmatriks kovariansnya\b/g, 'covariance matrix-nya'],
    [/\bkomponen utama\b/g, 'principal component'],
    [/\bKomponen utama\b/g, 'Principal component'],
    [/\bratio varians yang dijelaskan\b/g, 'explained variance ratio'],
    [/\brasio varians yang dijelaskan\b/g, 'explained variance ratio'],
    [/\bRasio varians yang dijelaskan\b/g, 'Explained variance ratio'],
    [/\bvarians yang dijelaskan\b/g, 'explained variance'],
    [/\bVarians yang dijelaskan\b/g, 'Explained variance'],
    [/### Menjelaskan rasio variansSetiap/g, '### Explained variance ratio\n\nSetiap'],
    [/\bvariansi kumulatif yang dijelaskan\b/g, 'cumulative explained variance'],
    [/\bVariansi kumulatif yang dijelaskan\b/g, 'Cumulative explained variance'],
    [/\bMetode siku\b/g, 'Elbow method'],
    [/\bmetode siku\b/g, 'elbow method'],
    [/\bkebingungan\b/g, 'perplexity'],
    [/\bKebingungan\b/g, 'Perplexity'],
    [/\bfungsi kernel\b/g, 'kernel function'],
    [/\bFungsi kernel\b/g, 'Kernel function'],
    [/\bmatrix kernel\b/g, 'kernel matrix'],
    [/\bMatrix kernel\b/g, 'Kernel matrix'],
    [/\bmatriks kernel\b/g, 'kernel matrix'],
    [/\bMatriks kernel\b/g, 'Kernel matrix'],
    [/\bkesalahan rekonstruksi\b/g, 'reconstruction error'],
    [/\bKesalahan Rekonstruksi\b/g, 'Reconstruction Error'],
    [/\bpemrosesan awal\b/g, 'preprocessing'],
    [/\bPemrosesan awal\b/g, 'Preprocessing'],
    [/\bprapemrosesan\b/g, 'preprocessing'],
    [/\bPrapemrosesan\b/g, 'Preprocessing'],
    [/\bperdagangan(?:nya)?\b/g, 'tradeoff'],
    [/\bpengorbanan(?:nya)?\b/g, 'tradeoff'],
    [/\bHAI\(/g, 'O('],
    [/\balfa\b/g, 'alpha'],
    [/\bAlfa\b/g, 'Alpha'],
    [/\bAnalisis Komponen Utama \(PCA\)/g, 'Principal Component Analysis (PCA)'],
    [/\bAljabar Linier\b/g, 'Linear Algebra'],
    [/\bProbabilitas & Distribusi\b/g, 'Probability & Distributions'],
    [/\bkamu tidak dapat merencanakannya\b/g, 'kamu tidak bisa mem-plot-nya'],
    [/\bmerencanakannya\b/g, 'mem-plot-nya'],
    [/\bPoin jauh\b/g, 'Point jauh'],
    [/\bTitik dekat\b/g, 'Point dekat'],
    [/\bDekomposisi sendiri\b/g, 'Eigendecomposition'],
    [/\bdekomposisi sendiri\b/g, 'eigendecomposition'],
    [/\bdekomposisi\b/g, 'decomposition'],
    [/\bDekomposisi\b/g, 'Decomposition'],
    [/\bTransformasi inverse\b/g, 'Inverse transform'],
    [/\btransformasi inverse\b/g, 'inverse transform'],
    [/\bFungsi kehilangan\b/g, 'Loss function'],
    [/\bfungsi kehilangan\b/g, 'loss function'],
    [/\bLanskap yang hilang\b/g, 'loss landscape'],
    [/\blanskap yang hilang\b/g, 'loss landscape'],
    [/\bHilangnya gradient\b/g, 'Vanishing gradient'],
    [/\bhilangnya gradient\b/g, 'vanishing gradient'],
    [/\bpropagasi mundur\b/g, 'backpropagation'],
    [/\bPropagasi mundur\b/g, 'Backpropagation'],
    [/\bumpan mundur\b/g, 'backward pass'],
    [/\bUmpan mundur\b/g, 'Backward pass'],
    [/\blapis demi lapis\b/g, 'layer by layer'],
    [/\bsublapisan\b/g, 'sublayer'],
    [/\bSublapisan\b/g, 'Sublayer'],
    [/\bkepala attention\b/g, 'attention head'],
    [/\bKepala attention\b/g, 'Attention head'],
    [/\bpembobotan ulang\b/g, 'reweighting'],
    [/\bPembobotan ulang\b/g, 'Reweighting'],
    [/\bpembobotan\b/g, 'weighting'],
    [/\bPembobotan\b/g, 'Weighting'],
  ];
  return replacements.reduce((value, [pattern, replacement]) => value.replace(pattern, replacement), text);
}

function polishMarkdown(markdown) {
  return transformOutsideCode(markdown, tidyTechnicalTerms);
}

function transformOutsideCode(markdown, transform) {
  const chunks = markdown.split(/(```[\s\S]*?```)/g);
  return chunks.map(chunk => {
    if (chunk.startsWith('```')) return chunk;
    return chunk.split(/(`[^`\n]+`)/g).map(part => (
      part.startsWith('`') ? part : transform(part)
    )).join('');
  }).join('');
}

async function translateWithGoogle(markdown) {
  const { protectedText, protectedBlocks } = protectMarkdown(markdown);
  const chunks = splitForTranslate(protectedText);
  const translated = [];

  for (const chunk of chunks) {
    const params = new URLSearchParams({
      client: 'gtx',
      sl: 'en',
      tl: 'id',
      dt: 't',
      q: chunk,
    });
    const response = await fetch(`https://translate.googleapis.com/translate_a/single?${params.toString()}`, {
      headers: { 'User-Agent': 'Mozilla/5.0' },
    });
    if (!response.ok) {
      throw new Error(`Google Translate failed ${response.status}: ${await response.text()}`);
    }
    const data = await response.json();
    translated.push(data?.[0]?.map(item => item?.[0] || '').join('') || '');
    await new Promise(resolve => setTimeout(resolve, 200));
  }

  return tidyTechnicalTerms(restoreMarkdown(translated.join(''), protectedBlocks).trim() + '\n');
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

async function translateLesson(markdown, relPath) {
  if (provider === 'openai') return translateWithOpenAI(markdown, relPath);
  if (provider === 'google') return translateWithGoogle(markdown, relPath);
  throw new Error(`Unsupported provider: ${provider}`);
}

const docs = (await walk(phasesDir))
  .filter(file => file.endsWith(path.join('docs', 'en.md')))
  .sort();

if (args.has('--polish-existing')) {
  let polished = 0;
  for (const idPath of (await walk(phasesDir)).filter(file => file.endsWith(path.join('docs', 'id.md'))).sort()) {
    const before = await fs.readFile(idPath, 'utf8');
    const after = polishMarkdown(before);
    if (after !== before) {
      await fs.writeFile(idPath, after, 'utf8');
      polished++;
    }
  }
  console.log(`Polished ${polished} Indonesian lesson file(s).`);
  process.exit(0);
}

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
  console.log(`Translating ${relLesson} with ${provider}`);
  const translated = await translateLesson(markdown, relLesson);
  await fs.writeFile(idPath, translated, 'utf8');
  created++;
}

console.log(`Created ${created} Indonesian lesson file(s).`);
