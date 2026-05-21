/**
 * Site internationalization helpers.
 *
 * Indonesian lessons should translate explanation and classroom guidance, while
 * keeping technical vocabulary, code, API names, and math symbols in English.
 */
(function () {
  var STORAGE_KEY = 'aifs:lang';
  var LEGACY_STORAGE_KEY = 'aifs:lesson-lang';
  var SUPPORTED_LANGS = ['en', 'id'];
  var DEFAULT_LANG = 'en';
  var GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/rohitg00/ai-engineering-from-scratch/main/';

  var LANG_LABELS = {
    en: 'English',
    id: 'Indonesia',
  };

  var UI_TEXT = {
    en: {
      'nav.contents': 'Contents',
      'nav.catalog': 'Catalog',
      'nav.roadmap': 'Roadmap',
      'nav.glossary': 'Glossary',
      'nav.home': 'Home',
      'nav.report': 'Report',
      'nav.reportSuggest': 'Report / Suggest',
      'search.aria': 'Search (⌘K)',
      'theme.aria': 'Toggle theme',
      'home.meta.left': 'FIG_000 · curriculum v1.0 · 2026',
      'home.meta.right': 'AI Engineering',
      'home.tagline': '416 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported.',
      'home.attribution': 'Run on your own machine.',
      'home.how.title': 'How this works',
      'home.how.p1': "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.",
      'home.how.p2': "This curriculum is the spine. 20 phases, 416 lessons, four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it's doing under the hood.",
      'home.how.p3': 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free and built to run on your own laptop.',
      'home.progress.title': 'Current Progress',
      'home.progress.finished': 'Finished Lessons',
      'home.progress.phases': 'Phases',
      'home.progress.languages': 'Languages',
      'home.progress.glossary': 'Glossary Terms',
      'home.toc.title': 'Curriculum · 20 phases · 416 lessons',
      'home.toc.subtitle': 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
      'home.legend.complete': 'Complete',
      'home.legend.progress': 'In progress',
      'home.legend.planned': 'Planned',
      'home.modal.saved': 'Progress saved in browser only',
      'home.modal.reset': 'Reset progress',
      'home.modal.confirmReset': 'Clear all your local progress (quiz answers and completed lessons)? This cannot be undone.',
      'home.modal.review': 'Review',
      'home.modal.read': 'Read',
      'home.modal.completed': 'completed',
      'home.colophon.title': 'Run Locally',
      'home.colophon.body': 'No paywall, no signup, no account. Every lesson has runnable code in Python, TypeScript, Rust, or Julia. Clone the repo and serve locally in one command.',
      'home.footer': '© 2026 · free forever',
      'site.footer': 'AI Engineering from Scratch · free forever.',
      'catalog.title': 'Lesson Catalog',
      'catalog.subtitle': 'Every lesson across all 20 phases. Search, filter, sort.',
      'catalog.search': 'Search lessons...',
      'catalog.allPhases': 'All Phases',
      'catalog.allStatus': 'All Status',
      'catalog.complete': 'Complete',
      'catalog.planned': 'Planned',
      'catalog.phase': 'Phase',
      'catalog.lesson': 'Lesson',
      'catalog.type': 'Type',
      'catalog.language': 'Language',
      'catalog.status': 'Status',
      'catalog.empty': 'No lessons match your filters.',
      'catalog.count': '{shown} of {total} lessons',
      'glossary.title': 'AI Glossary',
      'glossary.subtitle': 'What people say vs what things actually mean',
      'glossary.search': 'Search terms...',
      'glossary.count': '{shown} of {total} terms',
      'glossary.empty': 'No terms match your search.',
      'glossary.says': 'What people say',
      'glossary.means': 'What it actually means',
      'cmd.aria': 'Search lessons and glossary',
      'cmd.placeholder': 'Search lessons and glossary...',
      'cmd.search': 'Search',
      'cmd.results': 'Search results',
      'cmd.navigate': 'navigate',
      'cmd.open': 'open',
      'cmd.close': 'close',
      'cmd.emptyPrompt': 'Type to search 400+ lessons and glossary terms',
      'cmd.noResults': 'No results for',
      'roadmap.title': 'Roadmap',
      'roadmap.subtitle': 'Click any phase to see its prerequisites and what it unlocks downstream.',
      'roadmap.clear': 'Clear selection',
      'roadmap.scroll': 'Scroll to explore the full graph',
      'roadmap.none': 'None. This is a starting point.',
      'roadmap.final': 'Final destination. End of the curriculum.',
      'roadmap.lessonsComplete': '{done} of {total} lessons complete',
      'roadmap.prereqCount': '{count} prerequisite phases',
      'roadmap.unlockCount': '{count} phases unlocked',
      'roadmap.prerequisites': 'Prerequisites',
      'roadmap.unlocks': 'Unlocks',
      'roadmap.read': 'Read',
      'roadmap.github': 'View source',
      'lesson.loading': 'Loading lesson...',
      'phase.0.name': 'Setup & Tooling',
      'phase.0.desc': 'Get your environment ready for everything that follows.',
      'phase.1.name': 'Math Foundations',
      'phase.1.desc': 'The intuition behind every AI algorithm, through code.',
      'phase.2.name': 'ML Fundamentals',
      'phase.2.desc': 'Classical ML — still the backbone of most production AI.',
      'phase.3.name': 'Deep Learning Core',
      'phase.3.desc': 'Neural networks from first principles. No frameworks until you build one.',
      'phase.4.name': 'Computer Vision',
      'phase.4.desc': 'From pixels to understanding — image, video, 3D, VLMs, and world models.',
      'phase.5.name': 'NLP: Foundations to Advanced',
      'phase.5.desc': 'Language is the interface to intelligence.',
      'phase.6.name': 'Speech & Audio',
      'phase.6.desc': 'Hear, understand, speak.',
      'phase.7.name': 'Transformers Deep Dive',
      'phase.7.desc': 'The architecture that changed everything.',
      'phase.8.name': 'Generative AI',
      'phase.8.desc': 'Create images, video, audio, 3D, and more.',
      'phase.9.name': 'Reinforcement Learning',
      'phase.9.desc': 'The foundation of RLHF and game-playing AI.',
      'phase.10.name': 'LLMs from Scratch',
      'phase.10.desc': 'Build, train, and understand large language models.',
      'phase.11.name': 'LLM Engineering',
      'phase.11.desc': 'Put LLMs to work in production.',
      'phase.12.name': 'Multimodal AI',
      'phase.12.desc': 'See, hear, read, and reason across modalities.',
      'phase.13.name': 'Tools & Protocols',
      'phase.13.desc': 'The interfaces between AI and the real world.',
      'phase.14.name': 'Agent Engineering',
      'phase.14.desc': 'Build agents from first principles — loop, memory, planning, frameworks, production.',
      'phase.15.name': 'Autonomous Systems',
      'phase.15.desc': 'Long-horizon agents, self-improvement, and the 2026 safety stack.',
      'phase.16.name': 'Multi-Agent & Swarms',
      'phase.16.desc': 'Coordination, emergence, and collective intelligence.',
      'phase.17.name': 'Infrastructure & Production',
      'phase.17.desc': 'Ship AI to the real world.',
      'phase.18.name': 'Ethics, Safety & Alignment',
      'phase.18.desc': 'Build AI that helps humanity. Not optional.',
      'phase.19.name': 'Capstone Projects',
      'phase.19.desc': 'End-to-end shippable products, 20-40 hours each.',
    },
    id: {
      'nav.contents': 'Konten',
      'nav.catalog': 'Katalog',
      'nav.roadmap': 'Roadmap',
      'nav.glossary': 'Glosarium',
      'nav.home': 'Beranda',
      'nav.report': 'Laporkan',
      'nav.reportSuggest': 'Laporkan / Usulkan',
      'search.aria': 'Cari (⌘K)',
      'theme.aria': 'Ganti tema',
      'home.meta.left': 'FIG_000 · kurikulum v1.0 · 2026',
      'home.meta.right': 'AI Engineering',
      'home.tagline': '416 lesson. 20 phase. Setiap algorithm dibangun dari raw math sebelum framework digunakan.',
      'home.attribution': 'Jalankan di laptop sendiri.',
      'home.how.title': 'Cara belajar',
      'home.how.p1': 'Banyak materi AI mengajarkan potongan yang terpisah: paper di satu tempat, fine-tuning post di tempat lain, lalu agent demo yang terlihat keren tetapi tidak menyambung. Akhirnya kamu bisa membuat chatbot, tetapi belum tentu bisa menjelaskan loss curve. Kamu bisa memasang function ke agent, tetapi belum tentu paham attention di dalam model yang memanggilnya.',
      'home.how.p2': 'Kurikulum ini menjadi spine. 20 phase, 416 lesson, empat bahasa: Python, TypeScript, Rust, Julia. Linear algebra di awal, autonomous swarms di akhir. Setiap algorithm dibangun dari raw math lebih dulu. Backprop. Tokenizer. Attention. Agent loop. Saat PyTorch muncul, kamu sudah tahu apa yang terjadi di bawahnya.',
      'home.how.p3': 'Setiap lesson memakai loop yang sama: baca problem, turunkan math, tulis code, jalankan test, simpan artifact. Tidak ada video lima menit, tidak ada copy-paste deploy, tidak ada jalan pintas yang membuat konsep hilang. Gratis dan dibuat agar bisa berjalan di laptop sendiri.',
      'home.progress.title': 'Progress Saat Ini',
      'home.progress.finished': 'Lesson Selesai',
      'home.progress.phases': 'Phase',
      'home.progress.languages': 'Bahasa Code',
      'home.progress.glossary': 'Istilah Glosarium',
      'home.toc.title': 'Kurikulum · 20 phase · 416 lesson',
      'home.toc.subtitle': 'Klik satu phase untuk melihat lesson di dalamnya. Satu lesson dianggap matang saat math, code, dan test sudah ditulis.',
      'home.legend.complete': 'Selesai',
      'home.legend.progress': 'Sedang dikerjakan',
      'home.legend.planned': 'Direncanakan',
      'home.modal.saved': 'Progress disimpan hanya di browser ini',
      'home.modal.reset': 'Reset progress',
      'home.modal.confirmReset': 'Hapus semua progress lokalmu, termasuk quiz answers dan completed lessons? Ini tidak bisa dibatalkan.',
      'home.modal.review': 'Review',
      'home.modal.read': 'Baca',
      'home.modal.completed': 'selesai',
      'home.colophon.title': 'Run Locally',
      'home.colophon.body': 'Tidak ada paywall, tidak ada signup, tidak ada akun. Setiap lesson punya runnable code dalam Python, TypeScript, Rust, atau Julia, tergantung konsepnya.',
      'home.footer': '© 2026 · gratis selamanya',
      'site.footer': 'AI Engineering from Scratch · gratis selamanya.',
      'catalog.title': 'Katalog Lesson',
      'catalog.subtitle': 'Semua lesson dari 20 phase. Bisa dicari, difilter, dan diurutkan.',
      'catalog.search': 'Cari lesson...',
      'catalog.allPhases': 'Semua Phase',
      'catalog.allStatus': 'Semua Status',
      'catalog.complete': 'Selesai',
      'catalog.planned': 'Direncanakan',
      'catalog.phase': 'Phase',
      'catalog.lesson': 'Lesson',
      'catalog.type': 'Type',
      'catalog.language': 'Bahasa Code',
      'catalog.status': 'Status',
      'catalog.empty': 'Tidak ada lesson yang cocok dengan filter.',
      'catalog.count': '{shown} dari {total} lesson',
      'glossary.title': 'Glosarium AI',
      'glossary.subtitle': 'Yang sering orang katakan vs makna teknis sebenarnya',
      'glossary.search': 'Cari istilah...',
      'glossary.count': '{shown} dari {total} istilah',
      'glossary.empty': 'Tidak ada istilah yang cocok.',
      'glossary.says': 'Yang orang katakan',
      'glossary.means': 'Makna teknisnya',
      'cmd.aria': 'Cari lesson dan glosarium',
      'cmd.placeholder': 'Cari lesson dan glosarium...',
      'cmd.search': 'Cari',
      'cmd.results': 'Hasil pencarian',
      'cmd.navigate': 'navigasi',
      'cmd.open': 'buka',
      'cmd.close': 'tutup',
      'cmd.emptyPrompt': 'Ketik untuk mencari 400+ lesson dan istilah glosarium',
      'cmd.noResults': 'Tidak ada hasil untuk',
      'roadmap.title': 'Roadmap',
      'roadmap.subtitle': 'Klik phase untuk melihat prerequisite dan apa yang dibuka setelahnya.',
      'roadmap.clear': 'Hapus pilihan',
      'roadmap.scroll': 'Scroll untuk menjelajahi graph penuh',
      'roadmap.none': 'Tidak ada. Ini titik awal.',
      'roadmap.final': 'Tujuan akhir. Ujung kurikulum.',
      'roadmap.lessonsComplete': '{done} dari {total} lesson selesai',
      'roadmap.prereqCount': '{count} prerequisite phase',
      'roadmap.unlockCount': '{count} phase terbuka setelah ini',
      'roadmap.prerequisites': 'Prerequisites',
      'roadmap.unlocks': 'Unlocks',
      'roadmap.read': 'Baca',
      'roadmap.github': 'Lihat sumber',
      'lesson.loading': 'Memuat lesson...',
      'phase.0.name': 'Setup & Tooling',
      'phase.0.desc': 'Siapkan lingkungan kerja sebelum memulai semua materi.',
      'phase.1.name': 'Fondasi Matematika',
      'phase.1.desc': 'Intuisi di balik setiap algoritma AI, lewat kode.',
      'phase.2.name': 'Fundamental ML',
      'phase.2.desc': 'ML klasik — tetap jadi tulang punggung sebagian besar AI produksi.',
      'phase.3.name': 'Core Deep Learning',
      'phase.3.desc': 'Neural network dari prinsip pertama. Tanpa framework sampai kamu membangunnya sendiri.',
      'phase.4.name': 'Computer Vision',
      'phase.4.desc': 'Dari piksel ke pemahaman — gambar, video, 3D, VLM, dan world model.',
      'phase.5.name': 'NLP: Dasar hingga Lanjutan',
      'phase.5.desc': 'Bahasa adalah antarmuka kecerdasan.',
      'phase.6.name': 'Speech & Audio',
      'phase.6.desc': 'Dengar, pahami, bicara.',
      'phase.7.name': 'Transformers: Bedah Mendalam',
      'phase.7.desc': 'Arsitektur yang mengubah segalanya.',
      'phase.8.name': 'Generative AI',
      'phase.8.desc': 'Buat gambar, video, audio, 3D, dan lebih banyak lagi.',
      'phase.9.name': 'Reinforcement Learning',
      'phase.9.desc': 'Fondasi RLHF dan AI bermain game.',
      'phase.10.name': 'LLM dari Nol',
      'phase.10.desc': 'Bangun, latih, dan pahami large language model.',
      'phase.11.name': 'LLM Engineering',
      'phase.11.desc': 'Terapkan LLM di produksi.',
      'phase.12.name': 'Multimodal AI',
      'phase.12.desc': 'Lihat, dengar, baca, dan nalar lintas modalitas.',
      'phase.13.name': 'Tools & Protokol',
      'phase.13.desc': 'Antarmuka antara AI dan dunia nyata.',
      'phase.14.name': 'Agent Engineering',
      'phase.14.desc': 'Bangun agent dari prinsip pertama — loop, memori, perencanaan, framework, produksi.',
      'phase.15.name': 'Sistem Otonom',
      'phase.15.desc': 'Agent jangka panjang, self-improvement, dan safety stack 2026.',
      'phase.16.name': 'Multi-Agent & Swarm',
      'phase.16.desc': 'Koordinasi, kemunculan perilaku, dan kecerdasan kolektif.',
      'phase.17.name': 'Infrastruktur & Produksi',
      'phase.17.desc': 'Kirimkan AI ke dunia nyata.',
      'phase.18.name': 'Etika, Keamanan & Alignment',
      'phase.18.desc': 'Bangun AI yang membantu manusia. Ini bukan opsional.',
      'phase.19.name': 'Proyek Capstone',
      'phase.19.desc': 'Produk end-to-end siap kirim, masing-masing 20-40 jam.',
    },
  };

  var FALLBACK_MESSAGES = {
    id: 'Versi Indonesia final untuk lesson ini belum tersedia. Tampilan ini memakai auto Indonesian draft: penjelasan umum dilokalkan, sementara code dan technical terms tetap dipertahankan dalam English. Untuk hasil final, tambahkan file <code>docs/id.md</code> pada lesson ini.',
  };

  var TECHNICAL_TERMS = [
    'AI',
    'API',
    'activation',
    'agent',
    'backpropagation',
    'bias',
    'broadcasting',
    'class',
    'code',
    'dense layer',
    'determinant',
    'dot product',
    'embedding',
    'element-wise',
    'feature',
    'forward pass',
    'framework',
    'gradient',
    'identity matrix',
    'input',
    'inverse',
    'layer',
    'learning rate',
    'matrix',
    'matrix multiplication',
    'model',
    'neural network',
    'norm',
    'output',
    'parameter',
    'scalar',
    'shape',
    'tensor',
    'transpose',
    'vector',
    'weight',
  ];

  var MARKDOWN_REPLACEMENTS_ID = [
    [/^## Learning Objectives$/gm, '## Learning Objectives'],
    [/^## The Problem$/gm, '## The Problem'],
    [/^## The Concept$/gm, '## The Concept'],
    [/^## Build It$/gm, '## Build It'],
    [/^## Use It$/gm, '## Use It'],
    [/^## Ship It$/gm, '## Ship It'],
    [/^## Exercises$/gm, '## Exercises'],
    [/^## Key Terms$/gm, '## Key Terms'],
    [/^## Further Reading$/gm, '## Further Reading'],
    [/\*\*Type:\*\*/g, '**Type:**'],
    [/\*\*Languages:\*\*/g, '**Languages:**'],
    [/\*\*Prerequisites:\*\*/g, '**Prerequisites:**'],
    [/\*\*Time:\*\*/g, '**Time:**'],
    [/\bYou are about to learn\b/g, 'Kamu akan belajar'],
    [/\bYou want to build\b/g, 'Kamu ingin membangun'],
    [/\bYou read the code and see this\b/g, 'Kamu membaca code dan melihat ini'],
    [/\bIf your environment is broken\b/g, 'Kalau environment-mu bermasalah'],
    [/\bevery single lesson becomes\b/g, 'setiap lesson menjadi'],
    [/\binstead of learning\b/g, 'bukan proses belajar'],
    [/\bMost people skip\b/g, 'Banyak orang melewati'],
    [/\bThen they spend hours debugging\b/g, 'Lalu mereka menghabiskan waktu berjam-jam untuk debugging'],
    [/\bWe're going to do this once, properly\b/g, 'Kita akan melakukan ini sekali saja, dengan benar'],
    [/\bThis lesson\b/g, 'Lesson ini'],
    [/\bThis course\b/g, 'Course ini'],
    [/\bYour environment is now ready\b/g, 'Environment-mu sekarang siap'],
    [/\bRun the verification script\b/g, 'Jalankan verification script'],
    [/\bNo GPU\? No problem\b/g, 'Tidak punya GPU? Tidak masalah'],
    [/\bMost lessons work on CPU\b/g, 'Sebagian besar lesson bisa berjalan di CPU'],
    [/\bFor training-heavy lessons\b/g, 'Untuk training-heavy lessons'],
    [/\bCheck your system\b/g, 'Cek system-mu'],
    [/\binstall the basics\b/g, 'install tools dasar'],
    [/\bWe use\b/g, 'Kita memakai'],
    [/\bit's\b/g, 'tool ini'],
    [/\bfaster than\b/g, 'lebih cepat daripada'],
    [/\band handles\b/g, 'dan menangani'],
    [/\bautomatically\b/g, 'secara otomatis'],
    [/\bFor TypeScript lessons\b/g, 'Untuk TypeScript lessons'],
    [/\bFor performance-critical lessons\b/g, 'Untuk performance-critical lessons'],
    [/\bFor math-heavy lessons\b/g, 'Untuk math-heavy lessons'],
    [/\bwhere .* shines\b/g, 'ketika tool ini lebih cocok dipakai'],
    [/\bSee `/g, 'Lihat `'],
    [/\bfor a prompt that helps\b/g, 'untuk prompt yang membantu'],
    [/\bdiagnose environment issues\b/g, 'mendiagnosis environment issues'],
    [/\bRun\b/g, 'Jalankan'],
    [/\bCreate\b/g, 'Buat'],
    [/\bWrite\b/g, 'Tulis'],
    [/\band run each one\b/g, 'dan jalankan semuanya'],
    [/\bfrom scratch\b/g, 'from scratch'],
    [/\bproperly\b/g, 'dengan benar'],
    [/\bbottom-up\b/g, 'dari bawah ke atas'],
    [/\bEach layer depends on the one below it\b/g, 'Setiap layer bergantung pada layer di bawahnya'],
    [/\bThe Problem\b/g, 'The Problem'],
    [/\bThe Concept\b/g, 'The Concept'],
  ];

  function normalizeLang(lang) {
    lang = String(lang || '').toLowerCase();
    return SUPPORTED_LANGS.indexOf(lang) >= 0 ? lang : null;
  }

  function readStoredLang() {
    try {
      return normalizeLang(localStorage.getItem(STORAGE_KEY)) || normalizeLang(localStorage.getItem(LEGACY_STORAGE_KEY));
    } catch (e) {
      return null;
    }
  }

  function writeStoredLang(lang) {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
      localStorage.setItem(LEGACY_STORAGE_KEY, lang);
    } catch (e) {
      // localStorage may be disabled.
    }
  }

  function getInitialLang(params) {
    return normalizeLang(params.get('lang')) || readStoredLang() || DEFAULT_LANG;
  }

  function currentLang() {
    return normalizeLang(document.documentElement.getAttribute('data-lang')) || readStoredLang() || DEFAULT_LANG;
  }

  function text(key, replacements) {
    var lang = currentLang();
    var value = (UI_TEXT[lang] && UI_TEXT[lang][key]) || UI_TEXT.en[key] || key;
    replacements = replacements || {};
    Object.keys(replacements).forEach(function (name) {
      value = value.replace(new RegExp('\\{' + name + '\\}', 'g'), replacements[name]);
    });
    return value;
  }

  function applyStaticText(root) {
    root = root || document;
    document.documentElement.setAttribute('lang', currentLang() === 'id' ? 'id' : 'en');
    root.querySelectorAll('[data-i18n]').forEach(function (el) {
      el.textContent = text(el.getAttribute('data-i18n'));
    });
    root.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      el.innerHTML = text(el.getAttribute('data-i18n-html'));
    });
    root.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      el.setAttribute('placeholder', text(el.getAttribute('data-i18n-placeholder')));
    });
    root.querySelectorAll('[data-i18n-aria]').forEach(function (el) {
      el.setAttribute('aria-label', text(el.getAttribute('data-i18n-aria')));
      el.setAttribute('title', text(el.getAttribute('data-i18n-aria')));
    });
  }

  function updateLanguageToggle(root) {
    root = root || document;
    var lang = currentLang();
    root.querySelectorAll('.language-toggle button[data-lang]').forEach(function (btn) {
      var active = btn.getAttribute('data-lang') === lang;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  function setLanguage(lang, params) {
    lang = normalizeLang(lang) || DEFAULT_LANG;
    document.documentElement.setAttribute('data-lang', lang);
    writeStoredLang(lang);
    if (params) updateUrlLang(params, lang);
    updateLanguageToggle();
    applyStaticText();
    document.dispatchEvent(new CustomEvent('aifs:langchange', { detail: { lang: lang } }));
  }

  function initGlobalLanguage(params) {
    var lang = getInitialLang(params || new URLSearchParams(window.location.search));
    document.documentElement.setAttribute('data-lang', lang);
    updateLanguageToggle();
    applyStaticText();
    document.addEventListener('click', function (e) {
      var btn = e.target.closest('.language-toggle button[data-lang]');
      if (!btn) return;
      setLanguage(btn.getAttribute('data-lang'), params || new URLSearchParams(window.location.search));
    });
  }

  function lessonMarkdownUrls(path, lang) {
    var file = path + '/docs/' + lang + '.md';
    var urls = [];

    if (isLocalHost()) {
      urls.push(new URL('../' + file, window.location.href).href);
    }

    urls.push(GITHUB_RAW_BASE + file);
    return urls;
  }

  function isLocalHost() {
    return /^(localhost|127\.0\.0\.1|\[::1\])$/.test(window.location.hostname);
  }

  function updateUrlLang(params, lang) {
    params.set('lang', lang);
    window.history.replaceState({}, '', window.location.pathname + '?' + params.toString());
  }

  function fallbackMessage(lang) {
    return FALLBACK_MESSAGES[lang] || '';
  }

  function localizeLessonMarkdown(md, lang) {
    if (lang !== 'id') return md;
    return transformMarkdownOutsideCode(md, function (text) {
      var out = text;
      MARKDOWN_REPLACEMENTS_ID.forEach(function (entry) {
        out = out.replace(entry[0], entry[1]);
      });
      return out;
    });
  }

  function transformMarkdownOutsideCode(md, transform) {
    var chunks = md.split(/(```[\s\S]*?```)/g);
    return chunks.map(function (chunk) {
      if (chunk.indexOf('```') === 0) return chunk;
      var inline = chunk.split(/(`[^`\n]+`)/g);
      return inline.map(function (part) {
        if (part.indexOf('`') === 0) return part;
        return transform(part);
      }).join('');
    }).join('');
  }

  window.AIFSI18n = {
    DEFAULT_LANG: DEFAULT_LANG,
    STORAGE_KEY: STORAGE_KEY,
    SUPPORTED_LANGS: SUPPORTED_LANGS.slice(),
    TECHNICAL_TERMS: TECHNICAL_TERMS.slice(),
    fallbackMessage: fallbackMessage,
    applyStaticText: applyStaticText,
    currentLang: currentLang,
    getInitialLang: getInitialLang,
    initGlobalLanguage: initGlobalLanguage,
    lessonMarkdownUrls: lessonMarkdownUrls,
    localizeLessonMarkdown: localizeLessonMarkdown,
    normalizeLang: normalizeLang,
    setLanguage: setLanguage,
    t: text,
    updateUrlLang: updateUrlLang,
    updateLanguageToggle: updateLanguageToggle,
    writeStoredLang: writeStoredLang,
    labels: LANG_LABELS,
  };
})();
