/**
 * Lesson access policy for the static UI.
 *
 * This controls navigation and copy in the public frontend. It is not a secure
 * paywall while lesson Markdown remains publicly hosted.
 */
(function () {
  var FREE_PHASE_IDS = [0];
  var FREE_LESSONS_PER_PHASE = 2;
  var state = { user: null, signedIn: false, ready: false };

  function currentLang() {
    return document.documentElement.getAttribute('data-lang') || localStorage.getItem('aifs:lang') || 'en';
  }

  function text(en, id) {
    return currentLang() === 'id' ? id : en;
  }

  function getPhaseIndex(phaseOrIndex) {
    if (typeof phaseOrIndex === 'number') return phaseOrIndex;
    var phases = typeof PHASES !== 'undefined' ? PHASES : window.PHASES;
    if (!phases || !phaseOrIndex) return -1;
    for (var i = 0; i < phases.length; i++) {
      if (phases[i] === phaseOrIndex || phases[i].id === phaseOrIndex.id) return i;
    }
    return -1;
  }

  function isReadable(lesson) {
    return !!(lesson && (lesson.status === 'complete' || lesson.url));
  }

  function hasStoredSupabaseSession() {
    try {
      if (window.AIFSAuth && window.AIFSAuth.hasLocalSession) return window.AIFSAuth.hasLocalSession();
      for (var i = 0; i < localStorage.length; i++) {
        var key = localStorage.key(i);
        if (!key || !/^sb-.*-auth-token$/.test(key)) continue;
        var raw = localStorage.getItem(key);
        if (!raw) continue;
        var parsed = JSON.parse(raw);
        var expiresAt = parsed.expires_at || parsed.expiresAt || 0;
        if (expiresAt && expiresAt * 1000 <= Date.now()) continue;
        if (parsed.access_token || parsed.currentSession || parsed.user) return true;
      }
    } catch (_) {}
    return false;
  }

  function isSignedIn() {
    return !!(state.user || state.signedIn || hasStoredSupabaseSession());
  }

  function isFreeLesson(phaseOrIndex, lessonIndex) {
    var phaseIndex = getPhaseIndex(phaseOrIndex);
    var phases = typeof PHASES !== 'undefined' ? PHASES : window.PHASES;
    var phase = phases && phases[phaseIndex];
    if (!phase) return false;
    if (FREE_PHASE_IDS.indexOf(phase.id) !== -1) return true;
    return lessonIndex < FREE_LESSONS_PER_PHASE;
  }

  function canOpenLesson(phaseOrIndex, lessonIndex, lesson) {
    if (!isReadable(lesson)) return false;
    return isFreeLesson(phaseOrIndex, lessonIndex) || isSignedIn();
  }

  function accessLabel(phaseOrIndex, lessonIndex, lesson) {
    if (!isReadable(lesson)) return text('Planned', 'Direncanakan');
    if (isFreeLesson(phaseOrIndex, lessonIndex)) return text('Free', 'Gratis');
    return isSignedIn() ? text('Unlocked', 'Terbuka') : text('Login', 'Login');
  }

  function isLockedPath(path) {
    var phases = typeof PHASES !== 'undefined' ? PHASES : window.PHASES;
    if (!phases || !path) return false;
    for (var i = 0; i < phases.length; i++) {
      var lessons = phases[i].lessons || [];
      for (var j = 0; j < lessons.length; j++) {
        var lessonPath = window.AIFSProgress && lessons[j].url ? window.AIFSProgress.extractPath(lessons[j].url) : '';
        if (lessonPath === path) return !canOpenLesson(i, j, lessons[j]);
      }
    }
    return false;
  }

  function loginHref() {
    return loginHrefFor(location.pathname.split('/').pop() + location.search + location.hash);
  }

  function loginHrefFor(nextUrl) {
    return 'login.html?next=' + encodeURIComponent(nextUrl || 'profile.html');
  }

  function refresh() {
    if (!window.AIFSAuth) {
      state.user = null;
      state.signedIn = hasStoredSupabaseSession();
      state.ready = true;
      document.dispatchEvent(new CustomEvent('aifz:accesschange', { detail: state }));
      return Promise.resolve(state);
    }
    return window.AIFSAuth.getUser().then(function (user) {
      state.user = user || null;
      state.signedIn = !!user || hasStoredSupabaseSession();
      state.ready = true;
      document.dispatchEvent(new CustomEvent('aifz:accesschange', { detail: state }));
      return state;
    }).catch(function () {
      state.user = null;
      state.signedIn = hasStoredSupabaseSession();
      state.ready = true;
      document.dispatchEvent(new CustomEvent('aifz:accesschange', { detail: state }));
      return state;
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    refresh();
    window.addEventListener('aifz:supabase-loaded', refresh);
    window.addEventListener('storage', refresh);
  });

  window.AIFSAccess = {
    state: state,
    refresh: refresh,
    isFreeLesson: isFreeLesson,
    isSignedIn: isSignedIn,
    canOpenLesson: canOpenLesson,
    accessLabel: accessLabel,
    isLockedPath: isLockedPath,
    loginHref: loginHref,
    loginHrefFor: loginHrefFor,
    text: text
  };
})();
