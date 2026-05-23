(function () {
  if (window.supabase) {
    window.dispatchEvent(new Event('aifz:supabase-loaded'));
    return;
  }
  fetch('vendor/supabase.js?v=20260523-admin2')
    .then(function (res) { return res.ok ? res.text() : Promise.reject(new Error('supabase-js not found')); })
    .then(function (source) {
      window.supabase = new Function(source + '\nreturn supabase;')();
      window.dispatchEvent(new Event('aifz:supabase-loaded'));
    })
    .catch(function () {
      window.dispatchEvent(new Event('aifz:supabase-load-failed'));
    });
})();
