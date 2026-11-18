"""Genera un CSV con datos ficticios de ventas de una TIENDA DE ROPA (1 año: nov-2025 a oct-2026).

Columnas: fecha, producto, categoria, cantidad, precio_unitario
Uso:  python generar_datos.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# producto: (categoria, precio S/, ventas diarias promedio, tendencia mensual, temporada)
PRODUCTOS = {
    # Polos y camisas
    "Polo básico algodón":        ("Polos y camisas", 29.90, 14, 0.01, "verano"),
    "Polo estampado":             ("Polos y camisas", 39.90,  9, 0.02, "verano"),
    "Camisa manga larga":         ("Polos y camisas", 79.90,  4, 0.00, "regalo"),
    "Blusa de gasa":              ("Polos y camisas", 59.90,  5, 0.01, "madre"),
    # Pantalones
    "Jean clásico":               ("Pantalones", 119.90, 7, 0.00, "plana"),
    "Jean mom fit":               ("Pantalones", 129.90, 5, 0.03, "plana"),
    "Pantalón de vestir":         ("Pantalones", 99.90,  2, -0.01, "regalo"),
    "Short de verano":            ("Pantalones", 49.90,  6, 0.00, "verano_fuerte"),
    "Jogger deportivo":           ("Pantalones", 69.90,  5, 0.02, "invierno"),
    # Abrigos
    "Casaca impermeable":         ("Abrigos", 189.90, 1.2, 0.00, "invierno_fuerte"),
    "Chompa de lana":             ("Abrigos", 89.90,  2.5, 0.00, "invierno_fuerte"),
    "Polera con capucha":         ("Abrigos", 79.90,  4, 0.02, "invierno"),
    # Vestidos y faldas
    "Vestido casual":             ("Vestidos y faldas", 99.90, 3, 0.01, "verano"),
    "Vestido de fiesta":          ("Vestidos y faldas", 199.90, 0.8, 0.00, "fiestas"),
    "Falda plisada":              ("Vestidos y faldas", 69.90, 2, -0.02, "plana"),
    # Ropa interior y básicos
    "Medias pack x3":             ("Básicos", 19.90, 16, 0.00, "plana"),
    "Boxer pack x2":              ("Básicos", 34.90, 9, 0.00, "regalo"),
    "Pijama polar":               ("Básicos", 69.90, 1.5, 0.00, "invierno_fuerte"),
    # Accesorios
    "Gorra":                      ("Accesorios", 39.90, 3, 0.01, "verano"),
    "Bufanda tejida":             ("Accesorios", 34.90, 0.7, 0.00, "invierno_fuerte"),
    "Correa de cuero":            ("Accesorios", 59.90, 1.5, 0.00, "regalo"),
    "Uniforme escolar (buzo)":    ("Escolar", 89.90, 0.5, 0.00, "escolar"),
}

fechas = pd.date_range("2025-11-01", "2026-10-31", freq="D")


def factor_temporada(f: pd.Timestamp, tipo: str) -> float:
    m = f.month
    if tipo == "verano":           # dic-mar en Perú
        return 1.4 if m in (12, 1, 2, 3) else 0.85
    if tipo == "verano_fuerte":
        return 2.2 if m in (12, 1, 2, 3) else 0.35
    if tipo == "invierno":         # jun-sep
        return 1.4 if m in (5, 6, 7, 8, 9) else 0.8
    if tipo == "invierno_fuerte":
        return 3.0 if m in (6, 7, 8) else (1.6 if m in (5, 9) else 0.2)
    if tipo == "regalo":           # navidad, día del padre (jun)
        return {12: 2.5, 6: 1.8, 11: 1.2}.get(m, 0.9)
    if tipo == "madre":            # día de la madre (may)
        return {5: 2.2, 12: 1.6}.get(m, 0.9)
    if tipo == "fiestas":          # navidad/año nuevo y fiestas patrias
        return {12: 3.5, 1: 1.3, 7: 2.0}.get(m, 0.6)
    if tipo == "escolar":          # campaña escolar feb-mar
        return {2: 12.0, 3: 9.0, 1: 2.0}.get(m, 0.2)
    return 1.0


filas = []
for producto, (cat, precio, base, tendencia, temporada) in PRODUCTOS.items():
    for i, f in enumerate(fechas):
        lam = base * factor_temporada(f, temporada) * (1 + tendencia) ** (i / 30.4)
        lam *= 1.5 if f.dayofweek >= 5 else 1.0                       # fines de semana
        if f.month == 12 and f.day >= 15:                              # recta final navideña
            lam *= 1.4
        cantidad = rng.poisson(max(lam, 0.01))
        if cantidad == 0:
            continue
        # liquidaciones de fin de temporada (marzo y septiembre) y ofertas puntuales
        descuento = 0.7 if (f.month in (3, 9) and f.day >= 15) else rng.choice([1.0, 1.0, 1.0, 0.9])
        filas.append((f.date(), producto, cat, int(cantidad), round(precio * descuento, 2)))

df = pd.DataFrame(filas, columns=["fecha", "producto", "categoria", "cantidad", "precio_unitario"])
df.to_csv("ventas.csv", index=False, encoding="utf-8")
print(f"ventas.csv generado: {len(df):,} filas, {df['producto'].nunique()} productos, "
      f"{df['fecha'].min()} a {df['fecha'].max()}")
