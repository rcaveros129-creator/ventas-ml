-- =====================================================================
-- PREDICTA RETAIL - Fase A: tablas de la base de datos (PostgreSQL / Supabase)
-- Ejecutar UNA sola vez en: Supabase -> SQL Editor -> New query -> Run
-- =====================================================================

-- 1. TIENDAS: cada cliente del sistema (SaaS) es una tienda
create table if not exists tiendas (
    id               bigint generated always as identity primary key,
    nombre           text not null,
    datos_ficticios  boolean not null default true,
    creado_en        timestamptz not null default now()
);

-- 2. PRODUCTOS: catálogo y stock ACTUAL de cada tienda
create table if not exists productos (
    tienda_id              bigint not null references tiendas(id) on delete cascade,
    codigo                 text not null,
    nombre                 text not null,
    categoria              text not null,
    talla                  text not null,
    color                  text not null,
    precio_compra          double precision not null check (precio_compra >= 0),
    precio_venta           double precision not null check (precio_venta >= 0),
    stock_actual           integer not null check (stock_actual >= 0),
    stock_minimo           integer not null check (stock_minimo >= 0),
    tiempo_reposicion_dias integer not null check (tiempo_reposicion_dias >= 0),
    proveedor              text,
    fecha_alta             date,
    primary key (tienda_id, codigo)
);

-- 3. VENTAS: historial de ventas (movimientos de salida)
create table if not exists ventas (
    id             bigint generated always as identity primary key,
    tienda_id      bigint not null references tiendas(id) on delete cascade,
    id_venta       text,
    fecha          date not null,
    codigo         text not null,
    cantidad       integer not null check (cantidad > 0),
    precio_compra  double precision,
    precio_venta   double precision not null check (precio_venta >= 0),
    total_venta    double precision
);
create index if not exists ventas_tienda_fecha on ventas (tienda_id, fecha);

-- 4. COMPRAS: pedidos al proveedor (recibidos y en camino)
create table if not exists compras (
    id               bigint generated always as identity primary key,
    tienda_id        bigint not null references tiendas(id) on delete cascade,
    id_compra        text,
    fecha_pedido     date not null,
    fecha_recepcion  date not null,
    codigo           text not null,
    proveedor        text,
    cantidad         integer not null check (cantidad > 0),
    costo_unitario   double precision not null check (costo_unitario >= 0),
    costo_total      double precision,
    estado           text not null
);
create index if not exists compras_tienda on compras (tienda_id);

-- 5. SEGURIDAD: bloquear el acceso por la API pública de Supabase.
--    El aplicativo se conecta con la contraseña privada de la base de datos,
--    así que nadie más puede leer ni modificar estas tablas desde internet.
alter table tiendas   enable row level security;
alter table productos enable row level security;
alter table ventas    enable row level security;
alter table compras   enable row level security;

-- 6. Primera tienda: la de prueba, con los datos ficticios
insert into tiendas (nombre, datos_ficticios)
select 'Tienda de prueba', true
where not exists (select 1 from tiendas);

-- Comprobación: debe mostrar 1 fila con "Tienda de prueba"
select id, nombre, datos_ficticios from tiendas;
