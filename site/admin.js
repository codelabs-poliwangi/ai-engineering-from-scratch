(function () {
  var statusEl = document.getElementById('adminStatus');
  var bodyEl = document.getElementById('adminBody');
  var totalUsersEl = document.getElementById('totalUsers');
  var activeUsersEl = document.getElementById('activeUsers');
  var completedEl = document.getElementById('completedLessons');

  if (!statusEl || !bodyEl) return;

  function boot(attempt) {
    if (!window.AIFSAuth && attempt < 20) {
      setTimeout(function () { boot(attempt + 1); }, 100);
      return;
    }
    if (!window.AIFSAuth || !window.AIFSAuth.isConfigured()) {
      statusEl.textContent = 'Admin tools are not available because account login is not ready yet.';
      return;
    }
    window.AIFSAuth.isAdmin().then(function (allowed) {
      if (!allowed) {
        statusEl.textContent = 'Admin access required.';
        return;
      }
      statusEl.textContent = 'Loading learner activity...';
      return window.AIFSAuth.listLearners().then(renderLearners);
    }).catch(function (err) {
      statusEl.textContent = err && err.message ? err.message : 'Failed to load admin data.';
    });
  }

  function renderLearners(res) {
    if (res && res.error) throw res.error;
    var rows = res && res.data ? res.data : [];
    var active = 0;
    var totalCompleted = 0;

    bodyEl.innerHTML = rows.map(function (row) {
      var progressRow = Array.isArray(row.learning_progress) ? row.learning_progress[0] : row.learning_progress;
      var summary = summarize(progressRow && progressRow.progress);
      if (summary.visited > 0 || summary.completed > 0) active++;
      totalCompleted += summary.completed;
      return '<tr>'
        + '<td><strong>' + escapeHtml(row.full_name || '-') + '</strong><span>' + escapeHtml(row.email || '-') + '</span></td>'
        + '<td><span class="role-pill ' + escapeHtml(row.role || 'student') + '">' + escapeHtml(row.role || 'student') + '</span></td>'
        + '<td>' + summary.completed + '</td>'
        + '<td>' + summary.visited + '</td>'
        + '<td>' + summary.accuracy + '%</td>'
        + '<td>' + escapeHtml(formatDate((progressRow && progressRow.updated_at) || row.updated_at)) + '</td>'
        + '</tr>';
    }).join('') || '<tr><td colspan="6" class="empty-row">No learners yet.</td></tr>';

    totalUsersEl.textContent = String(rows.length);
    activeUsersEl.textContent = String(active);
    completedEl.textContent = String(totalCompleted);
    statusEl.textContent = 'Admin dashboard ready.';
  }

  function summarize(progress) {
    var lessons = progress && progress.lessons ? progress.lessons : {};
    var visited = 0;
    var completed = 0;
    var answers = 0;
    var correct = 0;
    Object.keys(lessons).forEach(function (path) {
      var lesson = lessons[path] || {};
      if (lesson.visitedAt) visited++;
      if (lesson.completedAt) completed++;
      Object.keys(lesson.answers || {}).forEach(function (qid) {
        answers++;
        if (lesson.answers[qid] && lesson.answers[qid].correct) correct++;
      });
    });
    return {
      visited: visited,
      completed: completed,
      accuracy: answers ? Math.round((correct / answers) * 100) : 0
    };
  }

  function formatDate(value) {
    if (!value) return '-';
    try { return new Date(value).toLocaleString(); }
    catch (_) { return '-'; }
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str == null ? '' : String(str);
    return div.innerHTML;
  }

  boot(0);
  window.addEventListener('aifz:supabase-loaded', function () { boot(0); });
})();
