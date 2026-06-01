create table if not exists public.scan_history (
  id uuid primary key,
  user_id uuid,
  filename text not null,
  predicted_class text not null,
  category text not null check (category in ('Organik', 'Anorganik', 'B3', 'Kertas', 'Residu')),
  confidence numeric not null check (confidence >= 0 and confidence <= 1),
  handling_advice text not null,
  raw_predictions jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

alter table public.scan_history
  add column if not exists user_id uuid;

alter table public.scan_history
  drop constraint if exists scan_history_category_check;

alter table public.scan_history
  add constraint scan_history_category_check
  check (category in ('Organik', 'Anorganik', 'B3', 'Kertas', 'Residu'));

create index if not exists scan_history_created_at_idx
  on public.scan_history (created_at desc);

create index if not exists scan_history_user_id_idx
  on public.scan_history (user_id, created_at desc);

create table if not exists public.app_users (
  id uuid primary key,
  email text not null unique,
  name text not null,
  avatar_url text not null default '',
  password_hash text not null,
  created_at timestamptz not null default now()
);

alter table public.app_users
  add column if not exists avatar_url text not null default '';

alter table public.scan_history
  drop constraint if exists scan_history_user_id_fkey;

alter table public.scan_history
  add constraint scan_history_user_id_fkey
  foreign key (user_id) references public.app_users(id) on delete set null;

create index if not exists app_users_email_idx
  on public.app_users (email);

create table if not exists public.weekly_challenges (
  id text primary key,
  title text not null,
  description text not null,
  current integer not null default 0 check (current >= 0),
  target integer not null default 1 check (target > 0),
  reward integer not null default 0 check (reward >= 0),
  ends_at text not null,
  created_at timestamptz not null default now()
);

create index if not exists weekly_challenges_created_at_idx
  on public.weekly_challenges (created_at desc);

create table if not exists public.community_posts (
  id uuid primary key,
  user_id uuid references public.app_users(id) on delete set null,
  author text not null,
  badge text not null default 'Anggota',
  title text not null,
  body text not null,
  type text not null default 'post' check (type in ('post', 'tip')),
  tag text not null default '',
  likes integer not null default 0 check (likes >= 0),
  created_at timestamptz not null default now()
);

create index if not exists community_posts_created_at_idx
  on public.community_posts (created_at desc);

create index if not exists community_posts_type_idx
  on public.community_posts (type);

create table if not exists public.community_comments (
  id uuid primary key,
  post_id uuid not null references public.community_posts(id) on delete cascade,
  parent_id uuid references public.community_comments(id) on delete cascade,
  user_id uuid references public.app_users(id) on delete set null,
  author text not null,
  body text not null,
  created_at timestamptz not null default now()
);

create index if not exists community_comments_post_id_idx
  on public.community_comments (post_id, created_at asc);

create table if not exists public.community_leaderboard (
  user_id uuid primary key references public.app_users(id) on delete cascade,
  name text not null,
  scans integer not null default 0 check (scans >= 0),
  points integer not null default 0 check (points >= 0),
  updated_at timestamptz not null default now()
);

create index if not exists community_leaderboard_points_idx
  on public.community_leaderboard (points desc);

create table if not exists public.eco_trivia (
  id text primary key,
  title text not null,
  text text not null,
  details text not null,
  thumbnail text,
  alt text,
  type text not null default 'organic',
  created_at timestamptz not null default now()
);

insert into public.weekly_challenges (id, title, description, current, target, reward, ends_at)
values (
  'weekly-plastic-10',
  'Scan 10 sampah plastik',
  'Kumpulkan scan plastik bersih minggu ini dan bagikan tips pemilahanmu.',
  6,
  10,
  80,
  'Minggu ini'
)
on conflict (id) do nothing;

insert into public.eco_trivia (id, title, text, details, type)
values
  (
    'trivia-plastic',
    'Fakta Daur Ulang',
    'Botol plastik PET sebaiknya dicuci, dikeringkan, lalu disetor ke bank sampah.',
    'Botol PET yang bersih lebih mudah diterima bank sampah karena tidak mencemari material lain.',
    'plastic'
  ),
  (
    'trivia-organic',
    'Sampah Organik',
    'Sisa sayur dan buah bisa diolah menjadi kompos untuk mengurangi sampah rumah.',
    'Sampah organik seperti kulit buah, sisa sayur, ampas kopi, dan daun kering bisa masuk komposter.',
    'organic'
  )
on conflict (id) do nothing;
