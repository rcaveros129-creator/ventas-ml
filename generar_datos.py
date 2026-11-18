"""Genera un CSV con datos ficticios de ventas de 1 año (nov-2025 a oct-2026).

Columnas: fecha, producto, categoria, cantidad, precio_unitario
Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# producto: (categoria, precio S/, ventas diarias promedio, tendencia mensual, estacionalidad)
PRODUCTOS = {
    "Arroz Costeño 5kg":        ("Abarrotes", 24.90, 18, 0.01, "plana"),
    "Aceite Primor 1L":         ("Abarrotes", 11.50, 14, 0.00, "plana"),
    "Azúcar Rubia 1kg":         ("Abarrotes",  4.20, 16, 0.00, "plana"),
    "Fideos Don Vittorio 500g": ("Abarrotes",  3.80, 12, 0.01, "invierno"),
    "Atún Florida 170g":        ("Abarrotes",  6.50,  7, 0.02, "plana"),
    "Leche Gloria 400g":        ("Lácteos",    4.10, 22, 0.00, "plana"),
    "Yogurt Laive 1L":          ("Lácteos",    7.90,  6, 0.02, "verano"),
    "Queso Fresco 500g":        ("Lácteos",   12.00,  3, 0.00, "plana"),
    "Inca Kola 1.5L":           ("Bebidas",    7.50, 20, 0.01, "verano"),
    "Agua San Luis 2.5L":       ("Bebidas",    4.50, 15, 0.00, "verano"),
    "Cerveza Pilsen 650ml":     ("Bebidas",    7.00, 10, 0.00, "fiestas"),
    "Jugo Frugos 1L":           ("Bebidas",    5.20,  4, -0.02, "verano"),
    "Galletas Soda Field":      ("Snacks",     1.20, 25, 0.00, "plana"),
    "Papas Lay's 150g":         ("Snacks",     6.90,  5, 0.03, "verano"),
    "Chocolate Sublime":        ("Snacks",     1.50, 13, 0.00, "invierno"),
    "Panetón D'Onofrio 900g":   ("Snacks",    32.90, 0.6, 0.00, "navidad"),
    "Detergente Bolívar 2kg":   ("Limpieza",  22.50,  4, 0.00, "plana"),
    "Lejía Clorox 1L":          ("Limpieza",   4.90,  5, 0.00, "plana"),
    "Papel Higiénico Elite x4": ("Limpieza",   6.80,  8, 0.01, "plana"),
    "Esponja Scotch-Brite":     ("Limpieza",   2.50, 1.5, -0.03, "plana"),
}

fechas = pd.date_range("2025-11-01", "2026-10-31", freq="D")


def factor_estacional(fecha: pd.Timestamp, tipo: str) -> float:
    m = fecha.month
    if tipo == "verano":      # dic-mar en Perú
        return 1.45 if m in (12, 1, 2, 3) else 0.85
    if tipo == "invierno":    # jun-sep
        return 1.35 if m in (6, 7, 8, 9) else 0.9
    if tipo == "fiestas":     # navidad, año nuevo, fiestas patrias
        return 1.6 if m in (12, 1, 7) else 0.95
    if tipo == "navidad":     # casi solo en nov-dic
        return {11: 6.0, 12: 18.0}.get(m, 0.15)
    return 1.0


filas = []
for producto, (cat, precio, base, tendencia, estac) in PRODUCTOS.items():
    for i, f in enumerate(fechas):
        mes_idx = i / 30.4
        lam = base * factor_estacional(f, estac) * (1 + tendencia) ** mes_idx
        lam *= 1.25 if f.dayofweek >= 5 else 1.0          # más ventas fin de semana
        cantidad = rng.poisson(max(lam, 0.01))
        if cantidad == 0:
            continue                                     # no hubo venta ese día
        # pequeñas variaciones de precio (ofertas / subidas)
        p = round(precio * rng.choice([1.0, 1.0, 1.0, 0.95, 1.03]), 2)
        filas.append((f.date(), producto, cat, int(cantidad), p))

df = pd.DataFrame(filas, columns=["fecha", "producto", "categoria", "cantidad", "precio_unitario"])
df.to_csv("ventas.csv", index=False, encoding="utf-8")
print(f"ventas.csv generado: {len(df):,} filas, {df['producto'].nunique()} productos, "
      f"{df['fecha'].min()} a {df['fecha'].max()}")
