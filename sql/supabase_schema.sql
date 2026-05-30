create table if not exists public.scan_history (
  id uuid primary key,
  filename text not null,
  predicted_class text not null,
  category text not null check (category in ('Organik', 'Anorganik', 'B3')),
  confidence numeric not null check (confidence >= 0 and confidence <= 1),
  handling_advice text not null,
  raw_predictions jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists scan_history_created_at_idx
  on public.scan_history (created_at desc);

create table if not exists public.app_users (
  id uuid primary key,
  email text not null unique,
  name text not null,
  password_hash text not null,
  created_at timestamptz not null default now()
);

create index if not exists app_users_email_idx
  on public.app_users (email);
