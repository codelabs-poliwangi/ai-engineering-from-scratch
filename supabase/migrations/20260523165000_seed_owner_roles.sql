-- Seed production owner accounts and keep ordinary registrations as students.

insert into public.admin_invites (email, role)
values
  ('sepyan@poliwangi.ac.id', 'owner'),
  ('sepyankristanto@gmail.com', 'owner'),
  ('zirolabs@gmail.com', 'owner'),
  ('vianziro@gmail.com', 'owner')
on conflict (email) do update set role = excluded.role;

update public.profiles
set role = 'owner',
    updated_at = now()
where lower(email) in (
  'sepyan@poliwangi.ac.id',
  'sepyankristanto@gmail.com',
  'zirolabs@gmail.com',
  'vianziro@gmail.com'
);

update public.profiles
set role = 'student',
    updated_at = now()
where role is null
   or role not in ('owner', 'admin', 'instructor', 'student');
