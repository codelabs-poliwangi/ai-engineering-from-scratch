-- AI Engineering from Zero learner accounts.
-- Run this in Supabase SQL editor for project crkjukavntvvqlgxtiaf.

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  role text not null default 'student',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.learning_progress (
  user_id uuid primary key references auth.users(id) on delete cascade,
  progress jsonb not null default '{"lessons":{},"updatedAt":0}'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
alter table public.learning_progress enable row level security;

drop policy if exists "profiles_select_own" on public.profiles;
create policy "profiles_select_own"
  on public.profiles for select
  using (auth.uid() = id);

drop policy if exists "profiles_insert_own" on public.profiles;
create policy "profiles_insert_own"
  on public.profiles for insert
  with check (auth.uid() = id);

drop policy if exists "profiles_update_own" on public.profiles;
create policy "profiles_update_own"
  on public.profiles for update
  using (auth.uid() = id)
  with check (auth.uid() = id);

drop policy if exists "learning_progress_select_own" on public.learning_progress;
create policy "learning_progress_select_own"
  on public.learning_progress for select
  using (auth.uid() = user_id);

drop policy if exists "learning_progress_insert_own" on public.learning_progress;
create policy "learning_progress_insert_own"
  on public.learning_progress for insert
  with check (auth.uid() = user_id);

drop policy if exists "learning_progress_update_own" on public.learning_progress;
create policy "learning_progress_update_own"
  on public.learning_progress for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
-- Roles and admin monitoring for AI Engineering from Zero.

create table if not exists public.app_roles (
  role text primary key,
  label text not null,
  can_monitor_users boolean not null default false,
  can_manage_roles boolean not null default false,
  created_at timestamptz not null default now()
);

insert into public.app_roles (role, label, can_monitor_users, can_manage_roles)
values
  ('owner', 'Owner', true, true),
  ('admin', 'Admin', true, true),
  ('instructor', 'Instructor', true, false),
  ('student', 'Student', false, false)
on conflict (role) do update set
  label = excluded.label,
  can_monitor_users = excluded.can_monitor_users,
  can_manage_roles = excluded.can_manage_roles;

alter table public.profiles
  drop constraint if exists profiles_role_check;

alter table public.profiles
  add constraint profiles_role_check
  check (role in ('owner', 'admin', 'instructor', 'student'));

alter table public.learning_progress
  drop constraint if exists learning_progress_profiles_fk;

alter table public.learning_progress
  add constraint learning_progress_profiles_fk
  foreign key (user_id) references public.profiles(id) on delete cascade;

create table if not exists public.admin_invites (
  email text primary key,
  role text not null references public.app_roles(role),
  created_at timestamptz not null default now()
);

-- Main owner candidates. Adjust or remove these if the production owner email differs.
insert into public.admin_invites (email, role)
values
  ('sepyan@poliwangi.ac.id', 'owner'),
  ('sepyankristanto@gmail.com', 'owner')
on conflict (email) do update set role = excluded.role;

create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
    from public.profiles p
    where p.id = auth.uid()
      and p.role in ('owner', 'admin')
  );
$$;

create or replace function public.handle_new_user_profile()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  invited_role text;
begin
  select ai.role into invited_role
  from public.admin_invites ai
  where lower(ai.email) = lower(new.email)
  limit 1;

  insert into public.profiles (id, email, full_name, role)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data ->> 'full_name', ''),
    coalesce(invited_role, 'student')
  )
  on conflict (id) do update set
    email = excluded.email,
    full_name = coalesce(nullif(public.profiles.full_name, ''), excluded.full_name),
    updated_at = now();

  return new;
end;
$$;

drop trigger if exists on_auth_user_created_profile on auth.users;
create trigger on_auth_user_created_profile
  after insert on auth.users
  for each row execute function public.handle_new_user_profile();

drop policy if exists "profiles_select_admin" on public.profiles;
create policy "profiles_select_admin"
  on public.profiles for select
  using (public.is_admin());

drop policy if exists "profiles_update_admin" on public.profiles;
create policy "profiles_update_admin"
  on public.profiles for update
  using (public.is_admin())
  with check (public.is_admin());

drop policy if exists "learning_progress_select_admin" on public.learning_progress;
create policy "learning_progress_select_admin"
  on public.learning_progress for select
  using (public.is_admin());

drop policy if exists "app_roles_select_authenticated" on public.app_roles;
alter table public.app_roles enable row level security;
create policy "app_roles_select_authenticated"
  on public.app_roles for select
  using (auth.role() = 'authenticated');

drop policy if exists "admin_invites_select_admin" on public.admin_invites;
alter table public.admin_invites enable row level security;
create policy "admin_invites_select_admin"
  on public.admin_invites for select
  using (public.is_admin());

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
