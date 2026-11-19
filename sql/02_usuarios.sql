-- =====================================================================
-- PREDICTA RETAIL - Fase B: usuarios que pueden entrar al sistema
-- Ejecutar en: Supabase -> SQL Editor -> New query -> Run
-- ANTES de ejecutar, cambia TU-CORREO@gmail.com por tu correo de Google.
-- =====================================================================

-- Cada correo de Google pertenece a una tienda y tiene un rol
create table if not exists usuarios (
    email      text primary key check (email = lower(email)),
    tienda_id  bigint not null references tiendas(id) on delete cascade,
    nombre     text,
    rol        text not null default 'dueño' check (rol in ('administrador', 'dueño', 'empleado')),
    activo     boolean not null default true,
    creado_en  timestamptz not null default now()
);

-- Bloquear el acceso por la API pública (igual que las demás tablas)
alter table usuarios enable row level security;

-- Tú: administrador del sistema, asignado a la Tienda de prueba (id 1)
insert into usuarios (email, tienda_id, nombre, rol)
values (lower('TU-CORREO@gmail.com'), 1, 'Administrador', 'administrador')
on conflict (email) do nothing;

-- Comprobación: debe mostrar tu correo
select email, tienda_id, rol, activo from usuarios;
