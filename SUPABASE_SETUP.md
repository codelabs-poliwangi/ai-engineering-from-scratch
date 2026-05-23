# Supabase setup for learner progress

This site can run without Supabase. In that mode, progress stays in browser
localStorage. To enable account login, profile data, and cloud progress sync:

1. Open the Supabase SQL editor for project `crkjukavntvvqlgxtiaf`.
2. Run `supabase/schema.sql`.
3. Enable Email/Password auth in Supabase Authentication providers.
4. Turn on email confirmation in Authentication > Sign In / Providers > Email,
   then configure the production site URL and redirect URLs in Authentication >
   URL Configuration.
5. Configure a production email sender under Authentication > Email / SMTP.
   The built-in sender is only suitable for testing and may reject or rate-limit
   confirmation emails.
6. Put the project publishable or anon public key in `site/supabase-config.js`.
   The current project uses a publishable key, so the production UI does not ask
   learners to paste any key.

For local development, Supabase URL Configuration must include:

- Site URL: `http://localhost:8765`
- Redirect URLs:
  - `http://localhost:8765/site/profile.html`
  - `http://localhost:8765/site/profile.html?verified=1&lang=en`
  - `http://localhost:8765/site/profile.html?verified=1&lang=id`

For production, replace the localhost entries with the production domain and
the same `/site/profile.html` callback path. If the Site URL is still
`http://localhost:3000`, confirmation links will fall back there.

The frontend writes only to these RLS-protected tables:

- `profiles`: learner name, email, role.
- `learning_progress`: versioned progress JSON containing visited lessons,
  completed lessons, and quiz answers.
- `app_roles`: role metadata for `owner`, `admin`, `instructor`, and `student`.
- `admin_invites`: email-based role bootstrap records for owner/admin users.

Current role model:

- `owner`: primary operator, can monitor learners and manage roles.
- `admin`: can monitor learners and manage roles.
- `instructor`: reserved for future class or cohort monitoring.
- `student`: default learner role.

The migration seeds owner invites for the main operator emails. If the production
owner signs up with a different email, add that email to `admin_invites` with
role `owner` before registration, or update the matching `profiles.role` after
registration from the Supabase SQL editor.

Do not put a `service_role` key in the static site. If full lesson content needs
real access control, move the lesson body behind Supabase Storage/RLS, Edge
Functions, or a backend route. A static Markdown file in a public repository is
not secure gated content even when the UI hides it.
