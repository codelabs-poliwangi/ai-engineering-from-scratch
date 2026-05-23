/**
 * Optional Supabase auth + progress sync.
 *
 * The public site still works without Supabase. When a Supabase anon key is
 * configured, logged-in learners can sync local progress into their profile.
 */
(function () {
  var PROJECT_URL = 'https://crkjukavntvvqlgxtiaf.supabase.co';
  var ANON_KEY_STORAGE = 'aifz:supabase-anon-key';
  var PROGRESS_TABLE = 'learning_progress';
  var PROFILE_TABLE = 'profiles';
  var syncTimer = null;
  var client = null;
  var authListenerAttached = false;

  function getLang() {
    return document.documentElement.getAttribute('data-lang') || localStorage.getItem('aifs:lang') || 'en';
  }

  function t(en, id) {
    return getLang() === 'id' ? id : en;
  }

  function getConfig() {
    var custom = window.AIFSSupabaseConfig || {};
    return {
      url: custom.url || PROJECT_URL,
      anonKey: custom.anonKey || localStorage.getItem(ANON_KEY_STORAGE) || ''
    };
  }

  function isConfigured() {
    var cfg = getConfig();
    var sdk = window.supabase || (typeof supabase !== 'undefined' ? supabase : null);
    return !!(sdk && cfg.url && cfg.anonKey && cfg.anonKey.length > 20);
  }

  function getClient() {
    if (client) return client;
    if (!isConfigured()) return null;
    var cfg = getConfig();
    var sdk = window.supabase || (typeof supabase !== 'undefined' ? supabase : null);
    client = sdk.createClient(cfg.url, cfg.anonKey);
    return client;
  }

  function setAnonKey(key) {
    if (key) localStorage.setItem(ANON_KEY_STORAGE, key.trim());
    else localStorage.removeItem(ANON_KEY_STORAGE);
    client = null;
  }

  function getSession() {
    var sb = getClient();
    if (!sb) return Promise.resolve({ data: { session: null }, error: null });
    return sb.auth.getSession();
  }

  function getUser() {
    return getSession().then(function (res) {
      return res && res.data ? res.data.session && res.data.session.user : null;
    });
  }

  function signIn(email, password) {
    var sb = getClient();
    if (!sb) return Promise.reject(new Error('Supabase is not configured.'));
    return sb.auth.signInWithPassword({ email: email, password: password });
  }

  function signUp(email, password, fullName) {
    var sb = getClient();
    if (!sb) return Promise.reject(new Error('Supabase is not configured.'));
    var lang = getLang();
    var redirectTo = window.location.origin + window.location.pathname.replace(/login\.html$/, 'profile.html') + '?verified=1&lang=' + encodeURIComponent(lang);
    return sb.auth.signUp({
      email: email,
      password: password,
      options: {
        data: { full_name: fullName || '' },
        emailRedirectTo: redirectTo
      }
    });
  }

  function signOut() {
    var sb = getClient();
    if (!sb) return Promise.resolve();
    return sb.auth.signOut();
  }

  function upsertProfile(fullName) {
    var sb = getClient();
    if (!sb) return Promise.resolve({ data: null, error: null });
    return getUser().then(function (user) {
      if (!user) return { data: null, error: null };
      var name = fullName || user.user_metadata && user.user_metadata.full_name || '';
      var payload = {
        id: user.id,
        email: user.email,
        updated_at: new Date().toISOString()
      };
      if (name) payload.full_name = name;
      return sb.from(PROFILE_TABLE).upsert(payload).select().single();
    });
  }

  function getProfile() {
    var sb = getClient();
    if (!sb) return Promise.resolve({ data: null, error: null });
    return getUser().then(function (user) {
      if (!user) return { data: null, error: null };
      return sb.from(PROFILE_TABLE).select('*').eq('id', user.id).maybeSingle();
    });
  }

  function isAdmin() {
    return getProfile().then(function (res) {
      var role = res && res.data ? res.data.role : '';
      return role === 'owner' || role === 'admin';
    });
  }

  function listLearners() {
    var sb = getClient();
    if (!sb) return Promise.resolve({ data: [], error: null });
    return sb.from(PROFILE_TABLE)
      .select('id,email,full_name,role,created_at,updated_at,learning_progress(progress,updated_at)')
      .order('updated_at', { ascending: false });
  }

  function syncProgress() {
    var sb = getClient();
    if (!sb || !window.AIFSProgress) return Promise.resolve({ skipped: true });
    return getUser().then(function (user) {
      if (!user) return { skipped: true };
      return sb.from(PROGRESS_TABLE).upsert({
        user_id: user.id,
        progress: window.AIFSProgress.getState(),
        updated_at: new Date().toISOString()
      }).select().single();
    });
  }

  function loadRemoteProgress() {
    var sb = getClient();
    if (!sb || !window.AIFSProgress) return Promise.resolve({ data: null, error: null });
    return getUser().then(function (user) {
      if (!user) return { data: null, error: null };
      return sb.from(PROGRESS_TABLE).select('progress, updated_at').eq('user_id', user.id).maybeSingle()
        .then(function (res) {
          if (res && res.data && res.data.progress) {
            window.AIFSProgress.mergeState(res.data.progress);
          }
          return res;
        });
    });
  }

  function scheduleSync() {
    if (!isConfigured()) return;
    clearTimeout(syncTimer);
    syncTimer = setTimeout(syncProgress, 1600);
  }

  function summarizeProgress() {
    var state = window.AIFSProgress ? window.AIFSProgress.getState() : { lessons: {} };
    var total = 0;
    var completed = 0;
    var visited = 0;
    var answers = 0;
    var correct = 0;
    var phases = typeof PHASES !== 'undefined' ? PHASES : window.PHASES;
    if (phases) {
      phases.forEach(function (phase) {
        (phase.lessons || []).forEach(function (lesson) {
          if (lesson.status !== 'planned') total++;
        });
      });
    }
    Object.keys(state.lessons || {}).forEach(function (path) {
      var lesson = state.lessons[path] || {};
      if (lesson.visitedAt) visited++;
      if (lesson.completedAt) completed++;
      Object.keys(lesson.answers || {}).forEach(function (qid) {
        answers++;
        if (lesson.answers[qid] && lesson.answers[qid].correct) correct++;
      });
    });
    return {
      total: total,
      completed: completed,
      visited: visited,
      answers: answers,
      correct: correct,
      percent: total ? Math.round((completed / total) * 100) : 0,
      updatedAt: state.updatedAt || 0
    };
  }

  function updateAuthLinks(user) {
    var links = document.querySelectorAll('[data-auth-link]');
    links.forEach(function (link) {
      if (user) {
        link.href = 'profile.html';
        link.textContent = t('Profile', 'Profil');
      } else {
        link.href = 'login.html';
        link.textContent = t('Login', 'Masuk');
      }
    });
  }

  function boot() {
    getUser().then(function (user) {
      updateAuthLinks(user);
      if (user) {
        upsertProfile().then(loadRemoteProgress).then(syncProgress);
      }
    });
    if (window.AIFSProgress) window.AIFSProgress.onChange(scheduleSync);
    var sb = getClient();
    if (sb && !authListenerAttached) {
      authListenerAttached = true;
      sb.auth.onAuthStateChange(function (_event, session) {
        updateAuthLinks(session && session.user);
        if (session && session.user) upsertProfile().then(loadRemoteProgress).then(syncProgress);
      });
    }
  }

  document.addEventListener('DOMContentLoaded', boot);
  window.addEventListener('aifz:supabase-loaded', boot);

  window.AIFSAuth = {
    getConfig: getConfig,
    isConfigured: isConfigured,
    setAnonKey: setAnonKey,
    getClient: getClient,
    getSession: getSession,
    getUser: getUser,
    signIn: signIn,
    signUp: signUp,
    signOut: signOut,
    upsertProfile: upsertProfile,
    getProfile: getProfile,
    isAdmin: isAdmin,
    listLearners: listLearners,
    syncProgress: syncProgress,
    loadRemoteProgress: loadRemoteProgress,
    summarizeProgress: summarizeProgress
  };
})();
