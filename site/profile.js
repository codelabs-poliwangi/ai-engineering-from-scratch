(function () {
  var accountStatus = document.getElementById('accountStatus');
  var progressStatus = document.getElementById('progressStatus');
  if (!accountStatus || !progressStatus) return;

  function renderProgress() {
    var summary = window.AIFSAuth
      ? window.AIFSAuth.summarizeProgress()
      : { completed: 0, total: 0, percent: 0, visited: 0, answers: 0, correct: 0, updatedAt: 0 };
    document.getElementById('completedStat').textContent = summary.completed + '/' + summary.total;
    document.getElementById('percentStat').textContent = summary.percent + '%';
    document.getElementById('visitedStat').textContent = String(summary.visited);
    document.getElementById('quizStat').textContent = summary.answers ? Math.round((summary.correct / summary.answers) * 100) + '%' : '0%';
    document.getElementById('progressFill').style.width = summary.percent + '%';
    progressStatus.textContent = summary.updatedAt
      ? 'Last local activity: ' + new Date(summary.updatedAt).toLocaleString()
      : label('profile.noProgress', 'No local progress yet. Start from the catalog or the first lesson.');

    var state = window.AIFSProgress ? window.AIFSProgress.getState() : { lessons: {} };
    var items = Object.keys(state.lessons || {}).map(function (path) {
      var lesson = state.lessons[path] || {};
      return { path: path, t: lesson.completedAt || lesson.visitedAt || 0, done: !!lesson.completedAt };
    }).sort(function (a, b) { return b.t - a.t; }).slice(0, 8);
    document.getElementById('activityList').innerHTML = items.length ? items.map(function (item) {
      return '<li><code>' + item.path + '</code><span>' + (item.done ? label('metric.done', 'done') : label('metric.visited', 'visited')) + '</span></li>';
    }).join('') : '<li><span>' + label('profile.noActivity', 'No activity yet.') + '</span><span>-</span></li>';
  }

  function bootAccount() {
    if (!window.AIFSAuth) {
      accountStatus.textContent = label('profile.authLoading', 'Local progress is ready. Cloud sync is still loading.');
      renderProgress();
      return;
    }
    if (!window.AIFSAuth.isConfigured()) {
      accountStatus.textContent = label('profile.authMissing', 'Account login is not available yet. Local progress is still saved in this browser.');
      renderProgress();
      return;
    }
    window.AIFSAuth.getUser().then(function (user) {
      if (!user) {
        accountStatus.textContent = label('profile.signedOut', 'Not signed in. Open Login to enable cloud sync.');
        renderProgress();
        return;
      }
      document.getElementById('accountEmail').value = user.email || '';
      accountStatus.textContent = label('profile.loadingRemote', 'Signed in. Loading profile and remote progress...');
      window.AIFSAuth.getProfile().then(function (res) {
        if (res && res.data) {
          document.getElementById('profileName').value = res.data.full_name || '';
          var adminLink = document.getElementById('adminLink');
          if (adminLink) adminLink.hidden = !(res.data.role === 'owner' || res.data.role === 'admin');
        }
        return window.AIFSAuth.loadRemoteProgress();
      }).then(function () {
        renderProgress();
        return window.AIFSAuth.syncProgress();
      }).then(function () {
        accountStatus.textContent = label('profile.synced', 'Progress is synced.');
      }).catch(function (err) {
        accountStatus.textContent = err && err.message ? err.message : label('profile.syncFailed', 'Failed to sync profile.');
      });
    });
  }

  document.getElementById('saveProfile').addEventListener('click', function () {
    if (!window.AIFSAuth) {
      accountStatus.textContent = label('profile.authLoading', 'Local progress is ready. Cloud sync is still loading.');
      return;
    }
    var name = document.getElementById('profileName').value.trim();
    accountStatus.textContent = label('profile.saving', 'Saving profile...');
    window.AIFSAuth.upsertProfile(name).then(function (res) {
      if (res && res.error) throw res.error;
      accountStatus.textContent = label('profile.saved', 'Profile saved.');
    }).catch(function (err) {
      accountStatus.textContent = err && err.message ? err.message : label('profile.saveFailed', 'Failed to save profile.');
    });
  });

  document.getElementById('syncNow').addEventListener('click', function () {
    if (!window.AIFSAuth) {
      accountStatus.textContent = label('profile.authLoading', 'Local progress is ready. Cloud sync is still loading.');
      return;
    }
    accountStatus.textContent = label('profile.syncing', 'Syncing progress...');
    window.AIFSAuth.syncProgress().then(function (res) {
      if (res && res.error) throw res.error;
      accountStatus.textContent = res && res.skipped ? label('profile.syncSkipped', 'Not signed in. Sync skipped.') : label('profile.synced', 'Progress is synced.');
    }).catch(function (err) {
      accountStatus.textContent = err && err.message ? err.message : label('profile.syncFailed', 'Failed to sync profile.');
    });
  });

  document.getElementById('signOut').addEventListener('click', function () {
    if (!window.AIFSAuth) {
      window.location.href = 'login.html';
      return;
    }
    window.AIFSAuth.signOut().then(function () {
      window.location.href = 'login.html';
    });
  });

  function start(attempt) {
    try {
      renderProgress();
    } catch (err) {
      progressStatus.textContent = err && err.message ? err.message : label('profile.progressFailed', 'Progress could not be read.');
    }
    if (window.AIFSProgress) window.AIFSProgress.onChange(function () {
      try { renderProgress(); } catch (_) {}
    });
    if (window.AIFSAuth || attempt > 10) {
      bootAccount();
      return;
    }
    setTimeout(function () { start(attempt + 1); }, 100);
  }

  start(0);
  setTimeout(bootAccount, 0);
  window.addEventListener('aifz:supabase-loaded', bootAccount);

  function label(key, fallback) {
    if (!window.AIFSI18n) return fallback;
    var value = window.AIFSI18n.t(key);
    return value && value !== key ? value : fallback;
  }
})();
