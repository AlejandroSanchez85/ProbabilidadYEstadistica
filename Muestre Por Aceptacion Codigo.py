"""
===========================================================
VALIDACIÓN COMPUTACIONAL DEL MUESTREO POR ACEPTACIÓN
===========================================================

Caso de estudio:
- Lote de N = 10.000 unidades
- Plan de muestreo simple (n, c) = (132, 3)
- AQL = 1%
- LTPD = 5%

Este programa:
1. Calcula probabilidades de aceptación con:
   - distribución binomial
   - distribución hipergeométrica
   - aproximación normal con corrección por continuidad
2. Calcula riesgos del productor y del consumidor
3. Construye tablas comparativas
4. Simula miles de lotes
5. Genera gráficas:
   - Curva OC
   - Curva AOQ
   - ASN
   - Diferencia entre binomial e hipergeométrica
   - Histogramas para varios escenarios
   - Diagrama de caja
   - Barras de aceptados/rechazados
   - Diagrama del flujo operativo
   - Diagrama del sistema de calidad

Dependencias:
- numpy
- pandas
- matplotlib
- scipy

Ejecutar:
    python muestreo_aceptacion.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Usar un backend no interactivo para guardar imágenes sin abrir ventanas.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binom, hypergeom, norm


# ===========================================================
# 1) CONFIGURACIÓN GENERAL
# ===========================================================

# Carpeta base donde se guardarán todas las salidas del programa.
BASE_DIR = Path("salidas_muestreo_aceptacion")

# Subcarpetas para organizar resultados.
FIG_DIR = BASE_DIR / "figuras"
TAB_DIR = BASE_DIR / "tablas"

# Crear carpetas si no existen.
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================
# 2) ESTRUCTURA DEL PLAN DE MUESTREO
# ===========================================================

@dataclass(frozen=True)
class PlanMuestreo:
    """
    Representa un plan simple de muestreo por aceptación.

    Atributos:
        N: tamaño del lote
        n: tamaño de la muestra
        c: número de aceptación
    """
    N: int
    n: int
    c: int


# ===========================================================
# 3) FUNCIONES DE PROBABILIDAD
# ===========================================================

def prob_aceptacion_binomial(n: int, c: int, p: float) -> float:
    """
    Probabilidad de aceptación bajo un modelo binomial.

    Si X ~ Binomial(n, p), entonces:
        Pa(p) = P(X <= c)

    Se usa la función de distribución acumulada (CDF) de scipy,
    que es equivalente a sumar desde x=0 hasta c.

    Args:
        n: tamaño de muestra
        c: número de aceptación
        p: proporción de defectuosos en el lote

    Returns:
        Probabilidad de aceptar el lote.
    """
    return float(binom.cdf(c, n, p))


def prob_aceptacion_hipergeometrica(N: int, D: int, n: int, c: int) -> float:
    """
    Probabilidad exacta de aceptación bajo el modelo hipergeométrico.

    Supone que el lote tiene N unidades, de las cuales D son defectuosas.
    Se extrae una muestra de tamaño n sin reemplazo.

    X ~ Hipergeométrica(N, D, n)

    Pa = P(X <= c)

    Args:
        N: tamaño del lote
        D: número de defectuosos en el lote
        n: tamaño de la muestra
        c: número de aceptación

    Returns:
        Probabilidad exacta de aceptación.
    """
    D = max(0, min(D, N))  # seguridad: D debe estar entre 0 y N
    return float(hypergeom.cdf(c, N, D, n))


def prob_aceptacion_normal_binomial(n: int, c: int, p: float) -> float:
    """
    Aproximación normal de la binomial con corrección por continuidad.

    Si X ~ Binomial(n, p), entonces:
        X ≈ N(mu, sigma^2)
        mu = n*p
        sigma = sqrt(n*p*(1-p))

    Con corrección por continuidad:
        Pa ≈ Phi((c + 0.5 - mu) / sigma)

    Args:
        n: tamaño de muestra
        c: número de aceptación
        p: proporción defectuosa

    Returns:
        Probabilidad aproximada de aceptación.
    """
    mu = n * p
    sigma = np.sqrt(n * p * (1 - p))

    # Caso degenerado: no hay variabilidad.
    if sigma == 0:
        return 1.0 if c >= mu else 0.0

    z = (c + 0.5 - mu) / sigma
    return float(norm.cdf(z))


def prob_aceptacion_normal_hipergeometrica(N: int, D: int, n: int, c: int) -> float:
    """
    Aproximación normal para la hipergeométrica con corrección por continuidad.

    Para la extracción sin reemplazo, la varianza incorpora el factor de
    corrección por población finita:

        mu = n * p
        sigma^2 = n * p * (1-p) * (N-n)/(N-1)

    donde p = D/N.

    Args:
        N: tamaño del lote
        D: defectuosos en el lote
        n: tamaño de la muestra
        c: número de aceptación

    Returns:
        Probabilidad aproximada de aceptación.
    """
    if N <= 1:
        return 1.0

    p = D / N
    mu = n * p
    sigma2 = n * p * (1 - p) * ((N - n) / (N - 1))
    sigma = np.sqrt(max(sigma2, 0.0))

    if sigma == 0:
        return 1.0 if c >= mu else 0.0

    z = (c + 0.5 - mu) / sigma
    return float(norm.cdf(z))


# ===========================================================
# 4) BÚSQUEDA DE PLANES CANDIDATOS
# ===========================================================

def buscar_plan_optimo(
    p_buena: float = 0.01,
    p_mala: float = 0.05,
    alfa_max: float = 0.05,
    beta_max: float = 0.10,
    n_max: int = 300,
    c_max: int = 15,
) -> tuple[int, int, float, float]:
    """
    Busca el primer plan (n, c) que cumpla simultáneamente:

        Pa(p_buena) >= 1 - alfa_max
        Pa(p_mala)   <= beta_max

    La búsqueda se hace por fuerza bruta sobre un rango razonable.

    Returns:
        (n, c, Pa_buena, Pa_mala)
    """
    for n in range(1, n_max + 1):
        for c in range(0, c_max + 1):
            pa_buena = prob_aceptacion_binomial(n, c, p_buena)
            pa_mala = prob_aceptacion_binomial(n, c, p_mala)
            if pa_buena >= 1 - alfa_max and pa_mala <= beta_max:
                return n, c, pa_buena, pa_mala

    raise RuntimeError("No se encontró un plan que cumpla las restricciones.")


# ===========================================================
# 5) TABLAS DE RESULTADOS
# ===========================================================

def tabla_probabilidades(plan: PlanMuestreo, escenarios: list[float]) -> pd.DataFrame:
    """
    Construye una tabla comparando:
    - probabilidad binomial
    - probabilidad hipergeométrica exacta
    - aproximación normal binomial
    - aproximación normal hipergeométrica

    Args:
        plan: plan de muestreo
        escenarios: lista de proporciones defectuosas

    Returns:
        DataFrame con la comparación.
    """
    filas = []

    for p in escenarios:
        D = int(round(plan.N * p))

        filas.append(
            {
                "p": p,
                "Pa_binomial": prob_aceptacion_binomial(plan.n, plan.c, p),
                "Pa_hipergeometrica": prob_aceptacion_hipergeometrica(plan.N, D, plan.n, plan.c),
                "Pa_normal_binomial": prob_aceptacion_normal_binomial(plan.n, plan.c, p),
                "Pa_normal_hipergeometrica": prob_aceptacion_normal_hipergeometrica(plan.N, D, plan.n, plan.c),
            }
        )

    return pd.DataFrame(filas)


def tabla_riesgos(plan: PlanMuestreo, aql: float, ltpd: float) -> pd.DataFrame:
    """
    Calcula indicadores clave del plan:
    - Pa(AQL)
    - Pa(LTPD)
    - alpha = riesgo del productor
    - beta = riesgo del consumidor
    - AOQ en AQL y LTPD
    - ASN (en plan simple es constante e igual a n)

    Returns:
        DataFrame con indicadores resumidos.
    """
    pa_aql = prob_aceptacion_binomial(plan.n, plan.c, aql)
    pa_ltpd = prob_aceptacion_binomial(plan.n, plan.c, ltpd)

    alpha = 1 - pa_aql
    beta = pa_ltpd

    aoq_aql = aql * pa_aql
    aoq_ltpd = ltpd * pa_ltpd

    return pd.DataFrame(
        {
            "indicador": [
                "Pa(AQL)",
                "Pa(LTPD)",
                "alpha",
                "beta",
                "AOQ(AQL)",
                "AOQ(LTPD)",
                "ASN",
            ],
            "valor": [
                pa_aql,
                pa_ltpd,
                alpha,
                beta,
                aoq_aql,
                aoq_ltpd,
                float(plan.n),
            ],
        }
    )


def comparar_planes(planes: list[tuple[int, int]], N: int, aql: float, ltpd: float) -> pd.DataFrame:
    """
    Compara varios planes candidatos para verificar su comportamiento
    en AQL y LTPD tanto por modelo binomial como por hipergeométrico.

    Args:
        planes: lista de tuplas (n, c)
        N: tamaño del lote
        aql: nivel aceptable de calidad
        ltpd: nivel tolerable de calidad deficiente

    Returns:
        DataFrame con la comparación.
    """
    filas = []
    D_aql = int(round(N * aql))
    D_ltpd = int(round(N * ltpd))

    for n, c in planes:
        filas.append(
            {
                "plan": f"({n}, {c})",
                "Pa1_bin": prob_aceptacion_binomial(n, c, aql),
                "Pa5_bin": prob_aceptacion_binomial(n, c, ltpd),
                "Pa1_exact": prob_aceptacion_hipergeometrica(N, D_aql, n, c),
                "Pa5_exact": prob_aceptacion_hipergeometrica(N, D_ltpd, n, c),
                "cumple": (
                    prob_aceptacion_binomial(n, c, aql) >= 0.95
                    and prob_aceptacion_binomial(n, c, ltpd) <= 0.10
                ),
            }
        )

    return pd.DataFrame(filas)


# ===========================================================
# 6) SIMULACIÓN DE LOTES
# ===========================================================

def simular_lotes(
    n: int,
    c: int,
    escenarios: list[float],
    repeticiones: int = 5000,
    semilla: int = 42,
) -> pd.DataFrame:
    """
    Simula lotes usando una binomial para generar el número de defectuosos
    observados en la muestra.

    Para cada escenario de p:
    - genera 'repeticiones' muestras
    - cuenta cuántas son aceptadas y cuántas rechazadas
    - resume estadísticas básicas de la distribución simulada

    Args:
        n: tamaño de muestra
        c: número de aceptación
        escenarios: lista de valores de p
        repeticiones: número de simulaciones por escenario
        semilla: semilla aleatoria para reproducibilidad

    Returns:
        DataFrame resumido.
    """
    rng = np.random.default_rng(semilla)
    filas = []

    for p in escenarios:
        defectuosos = rng.binomial(n, p, size=repeticiones)
        aceptados = int(np.sum(defectuosos <= c))
        rechazados = int(repeticiones - aceptados)

        filas.append(
            {
                "p": p,
                "aceptados": aceptados,
                "rechazados": rechazados,
                "tasa_aceptacion_sim": aceptados / repeticiones,
                "media_defectuosos": float(np.mean(defectuosos)),
                "desv_defectuosos": float(np.std(defectuosos, ddof=1)),
                "q1": float(np.percentile(defectuosos, 25)),
                "mediana": float(np.median(defectuosos)),
                "q3": float(np.percentile(defectuosos, 75)),
            }
        )

    return pd.DataFrame(filas)


# ===========================================================
# 7) GRÁFICAS ESTADÍSTICAS
# ===========================================================

def graficar_oc(n: int, c: int, aql: float, ltpd: float, archivo: Path) -> None:
    """
    Grafica la curva característica de operación (OC).

    La curva OC muestra cómo cambia la probabilidad de aceptación
    según aumenta la proporción defectuosa p.
    """
    p = np.linspace(0, 0.12, 241)
    pa = np.array([prob_aceptacion_binomial(n, c, pi) for pi in p])

    plt.figure(figsize=(8, 5))
    plt.plot(p * 100, pa, linewidth=2, label="P_a binomial")
    plt.scatter(
        [aql * 100, ltpd * 100],
        [prob_aceptacion_binomial(n, c, aql), prob_aceptacion_binomial(n, c, ltpd)],
        zorder=3,
        label="Puntos AQL / LTPD",
    )
    plt.axvline(aql * 100, linestyle="--", label="AQL")
    plt.axvline(ltpd * 100, linestyle="--", label="LTPD")
    plt.axhline(0.95, linestyle=":", label="Referencia 0.95")
    plt.axhline(0.10, linestyle=":", label="Referencia 0.10")
    plt.xlabel("Porcentaje defectuoso p (%)")
    plt.ylabel("Probabilidad de aceptación P_a")
    plt.title(f"Curva OC del plan (n={n}, c={c})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_aoq(n: int, c: int, archivo: Path) -> None:
    """
    Grafica la AOQ (Average Outgoing Quality) aproximada para el plan simple.

    En un plan simple sin rectificación explícita, se suele ilustrar como:
        AOQ(p) ≈ p * P_a(p)

    La interpretación es pedagógica: la calidad de salida depende
    de la calidad de entrada y de la probabilidad de aceptación.
    """
    p = np.linspace(0, 0.12, 241)
    pa = np.array([prob_aceptacion_binomial(n, c, pi) for pi in p])
    aoq = p * pa

    plt.figure(figsize=(8, 5))
    plt.plot(p * 100, aoq * 100, linewidth=2)
    plt.xlabel("Porcentaje defectuoso de entrada p (%)")
    plt.ylabel("AOQ aproximada (%)")
    plt.title(f"Curva AOQ del plan (n={n}, c={c})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_asn(n: int, archivo: Path) -> None:
    """
    Grafica el ASN (Average Sample Number).

    En un plan simple, el tamaño promedio inspeccionado es constante:
        ASN = n
    """
    p = np.linspace(0, 0.12, 241)
    asn = np.full_like(p, n, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(p * 100, asn, linewidth=2)
    plt.xlabel("Porcentaje defectuoso p (%)")
    plt.ylabel("ASN")
    plt.title("ASN constante en un plan simple")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_diferencia_modelos(n: int, c: int, N: int, archivo: Path) -> None:
    """
    Grafica la diferencia absoluta entre la probabilidad de aceptación
    calculada por binomial y por hipergeométrica.

    Esto permite ver qué tan buena es la aproximación binomial.
    """
    p = np.linspace(0.001, 0.12, 241)
    dif = []

    for pi in p:
        D = int(round(N * pi))
        pa_bin = prob_aceptacion_binomial(n, c, pi)
        pa_hg = prob_aceptacion_hipergeometrica(N, D, n, c)
        dif.append(abs(pa_bin - pa_hg))

    plt.figure(figsize=(8, 5))
    plt.plot(p * 100, dif, linewidth=2)
    plt.xlabel("Porcentaje defectuoso p (%)")
    plt.ylabel("Diferencia absoluta |Pa_bin - Pa_hg|")
    plt.title("Diferencia entre modelos binomial e hipergeométrico")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_histogramas(
    n: int,
    escenarios: list[float],
    repeticiones: int,
    carpeta_salida: Path,
    semilla: int = 42,
) -> None:
    """
    Genera un histograma independiente para cada escenario de calidad.

    Cada histograma muestra la distribución del número de defectuosos
    observados en la muestra para un valor dado de p.
    """
    rng = np.random.default_rng(semilla)

    for p in escenarios:
        defectuosos = rng.binomial(n, p, size=repeticiones)

        plt.figure(figsize=(7.5, 4.8))
        bins = range(0, int(defectuosos.max()) + 2)
        plt.hist(defectuosos, bins=bins, edgecolor="black")
        plt.xlabel("Número de defectuosos en la muestra")
        plt.ylabel("Frecuencia")
        plt.title(f"Histograma de defectuosos (p = {p*100:.0f}%)")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        nombre = carpeta_salida / f"histograma_{int(p*100):02d}pct.png"
        plt.savefig(nombre, dpi=200)
        plt.close()


def graficar_boxplot(
    n: int,
    escenarios: list[float],
    repeticiones: int,
    archivo: Path,
    semilla: int = 42,
) -> None:
    """
    Genera un diagrama de caja e intercuartílico para la proporción
    defectuosa muestral en varios escenarios.
    """
    rng = np.random.default_rng(semilla)
    datos = [rng.binomial(n, p, size=repeticiones) / n for p in escenarios]

    plt.figure(figsize=(8, 5))
    plt.boxplot(datos, labels=[f"{int(p*100)}%" for p in escenarios], showmeans=True)
    plt.xlabel("Escenario de defectos reales p")
    plt.ylabel("Proporción defectuosa muestral")
    plt.title("Diagrama de caja de la proporción defectuosa muestral")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_barras_decision(df_sim: pd.DataFrame, archivo: Path) -> None:
    """
    Grafica barras comparando el número de lotes aceptados y rechazados
    en la simulación para cada escenario de p.
    """
    x = np.arange(len(df_sim))
    ancho = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - ancho / 2, df_sim["aceptados"], width=ancho, label="Aceptados")
    plt.bar(x + ancho / 2, df_sim["rechazados"], width=ancho, label="Rechazados")
    plt.xticks(x, [f"{int(p*100)}%" for p in df_sim["p"]])
    plt.xlabel("Escenario de defectos reales p")
    plt.ylabel("Número de lotes simulados")
    plt.title("Lotes aceptados y rechazados en la simulación")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_resumen_riesgos(plan: PlanMuestreo, aql: float, ltpd: float, archivo: Path) -> pd.DataFrame:
    """
    Genera una gráfica resumen con los indicadores:
    - Pa(AQL)
    - Pa(LTPD)
    - alpha
    - beta

    Returns:
        DataFrame con esos valores.
    """
    pa_aql = prob_aceptacion_binomial(plan.n, plan.c, aql)
    pa_ltpd = prob_aceptacion_binomial(plan.n, plan.c, ltpd)

    alpha = 1 - pa_aql
    beta = pa_ltpd

    df = pd.DataFrame(
        {
            "indicador": ["Pa(AQL)", "Pa(LTPD)", "alpha", "beta"],
            "valor": [pa_aql, pa_ltpd, alpha, beta],
        }
    )

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(df["indicador"], df["valor"])
    plt.ylim(0, 1)
    plt.ylabel("Valor")
    plt.title("Resumen de riesgos y probabilidades clave")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()

    return df


# ===========================================================
# 8) DIAGRAMAS DEL PROCESO Y DEL SISTEMA
# ===========================================================

def _dibujar_caja(ax, x: float, y: float, texto: str, width: float = 2.6, height: float = 0.8) -> None:
    """
    Función auxiliar para dibujar una caja con texto en un diagrama.
    """
    from matplotlib.patches import FancyBboxPatch

    caja = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(caja)
    ax.text(
        x + width / 2,
        y + height / 2,
        texto,
        ha="center",
        va="center",
        fontsize=10,
    )


def _dibujar_flecha(ax, x1: float, y1: float, x2: float, y2: float, texto: str | None = None) -> None:
    """
    Función auxiliar para dibujar una flecha entre dos puntos.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )

    if texto:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, texto, ha="center", va="center", fontsize=9)


def graficar_flujo_operativo(archivo: Path) -> None:
    """
    Diagrama del recorrido de la muestra.

    Reproduce la lógica:
        Definir lote -> fijar plan -> muestrear -> contar defectuosos
        -> decidir aceptar/rechazar
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Cajas principales
    _dibujar_caja(ax, 0.4, 4.2, "Inicio")
    _dibujar_caja(ax, 2.0, 4.2, "Definir lote,\nunidad y defecto")
    _dibujar_caja(ax, 4.4, 4.2, "Fijar n, c,\nAQL y LTPD")
    _dibujar_caja(ax, 6.8, 4.2, "Tomar muestra\naleatoria")
    _dibujar_caja(ax, 9.2, 4.2, "Contar\n defectuosos X")

    _dibujar_caja(ax, 4.4, 2.3, "¿X ≤ c?")
    _dibujar_caja(ax, 2.2, 0.7, "Aceptar\nel lote")
    _dibujar_caja(ax, 6.8, 0.7, "Rechazar\nel lote")

    # Flechas horizontales superiores
    _dibujar_flecha(ax, 3.0, 4.6, 4.4, 4.6)
    _dibujar_flecha(ax, 5.6, 4.6, 6.8, 4.6)
    _dibujar_flecha(ax, 8.4, 4.6, 9.2, 4.6)

    # Flechas hacia decisión
    _dibujar_flecha(ax, 10.4, 4.2, 10.4, 2.8)
    _dibujar_flecha(ax, 10.4, 2.8, 7.0, 2.8)
    _dibujar_flecha(ax, 7.0, 2.8, 5.7, 2.8)
    _dibujar_flecha(ax, 5.7, 2.8, 5.7, 3.1)

    # Flechas desde la decisión
    _dibujar_flecha(ax, 5.0, 2.3, 3.5, 1.5, "Sí")
    _dibujar_flecha(ax, 5.8, 2.3, 7.9, 1.5, "No")

    # Flechas finales
    _dibujar_flecha(ax, 3.5, 1.5, 3.5, 1.1)
    _dibujar_flecha(ax, 7.9, 1.5, 7.9, 1.1)

    ax.text(0.65, 4.55, "", fontsize=1)  # evita un borde raro en algunos renders
    plt.title("Flujo operativo del muestreo por aceptación")
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


def graficar_sistema_calidad(archivo: Path) -> None:
    """
    Diagrama del sistema de calidad basado en muestreo por aceptación.

    Presenta la lógica global:
        Producción -> Muestreo -> Decisión -> Acción correctiva / liberación
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _dibujar_caja(ax, 0.6, 3.8, "Producción\ndel lote")
    _dibujar_caja(ax, 3.0, 3.8, "Muestreo\naleatorio")
    _dibujar_caja(ax, 5.4, 3.8, "Decisión:\nAcepta / Rechaza")
    _dibujar_caja(ax, 8.2, 4.8, "Liberación\ndel lote")
    _dibujar_caja(ax, 8.2, 2.6, "Acción correctiva\no reproceso")
    _dibujar_caja(ax, 5.4, 1.2, "Registro estadístico\ny trazabilidad")

    _dibujar_flecha(ax, 3.2, 4.2, 3.0, 4.2)
    _dibujar_flecha(ax, 5.4, 4.2, 5.4, 4.2)
    _dibujar_flecha(ax, 7.7, 4.2, 8.2, 5.2, "Acepta")
    _dibujar_flecha(ax, 7.7, 4.0, 8.2, 3.2, "Rechaza")
    _dibujar_flecha(ax, 6.7, 3.8, 6.7, 2.0)

    plt.title("Esquema funcional del sistema de control de calidad")
    plt.tight_layout()
    plt.savefig(archivo, dpi=200)
    plt.close()


# ===========================================================
# 9) UTILIDADES DE EXPORTACIÓN
# ===========================================================

def guardar_tabla(df: pd.DataFrame, nombre_archivo: str) -> Path:
    """
    Guarda un DataFrame en formato CSV dentro de la carpeta de tablas.
    """
    ruta = TAB_DIR / nombre_archivo
    df.to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def imprimir_titulo(texto: str) -> None:
    """
    Imprime títulos de sección de forma más legible en consola.
    """
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


# ===========================================================
# 10) PROGRAMA PRINCIPAL
# ===========================================================

def main() -> None:
    """
    Ejecuta todo el flujo del trabajo:
    1. Define el caso
    2. Calcula probabilidades y riesgos
    3. Genera tablas
    4. Simula escenarios
    5. Produce gráficas y diagramas
    """
    # -------------------------------------------------------
    # Parámetros del caso de estudio
    # -------------------------------------------------------
    N = 10_000
    aql = 0.01
    ltpd = 0.05

    # Plan seleccionado en el informe
    plan = PlanMuestreo(N=N, n=132, c=3)

    # Escenarios usados en las tablas y simulaciones
    escenarios = [0.01, 0.03, 0.05, 0.08]

    # Número de simulaciones
    repeticiones = 5000

    imprimir_titulo("CÁLCULOS BÁSICOS DEL PLAN")

    # Probabilidades en AQL y LTPD bajo el modelo binomial
    pa_aql_bin = prob_aceptacion_binomial(plan.n, plan.c, aql)
    pa_ltpd_bin = prob_aceptacion_binomial(plan.n, plan.c, ltpd)

    # Probabilidades exactas bajo el modelo hipergeométrico
    D_aql = int(round(N * aql))
    D_ltpd = int(round(N * ltpd))
    pa_aql_exact = prob_aceptacion_hipergeometrica(N, D_aql, plan.n, plan.c)
    pa_ltpd_exact = prob_aceptacion_hipergeometrica(N, D_ltpd, plan.n, plan.c)

    # Riesgos principales
    alpha_bin = 1 - pa_aql_bin
    beta_bin = pa_ltpd_bin

    alpha_exact = 1 - pa_aql_exact
    beta_exact = pa_ltpd_exact

    print(f"Plan de muestreo: N={plan.N}, n={plan.n}, c={plan.c}")
    print(f"AQL = {aql:.2%}")
    print(f"LTPD = {ltpd:.2%}")
    print()
    print(f"Pa binomial en AQL: {pa_aql_bin:.6f}")
    print(f"Pa binomial en LTPD: {pa_ltpd_bin:.6f}")
    print(f"Pa exacta en AQL (hipergeométrica): {pa_aql_exact:.6f}")
    print(f"Pa exacta en LTPD (hipergeométrica): {pa_ltpd_exact:.6f}")
    print()
    print(f"Riesgo del productor alpha (binomial): {alpha_bin:.6f}")
    print(f"Riesgo del consumidor beta (binomial): {beta_bin:.6f}")
    print(f"Riesgo del productor alpha (exacto): {alpha_exact:.6f}")
    print(f"Riesgo del consumidor beta (exacto): {beta_exact:.6f}")

    # -------------------------------------------------------
    # Tablas
    # -------------------------------------------------------
    imprimir_titulo("TABLAS")

    df_prob = tabla_probabilidades(plan, escenarios)
    df_riesgos = tabla_riesgos(plan, aql, ltpd)
    df_sim = simular_lotes(plan.n, plan.c, escenarios, repeticiones=repeticiones)
    df_planes = comparar_planes([(132, 3), (133, 3), (158, 4)], N, aql, ltpd)

    # Guardar tablas en CSV
    ruta_prob = guardar_tabla(df_prob, "tabla_probabilidades.csv")
    ruta_riesgos = guardar_tabla(df_riesgos, "tabla_riesgos.csv")
    ruta_sim = guardar_tabla(df_sim, "tabla_simulacion.csv")
    ruta_planes = guardar_tabla(df_planes, "tabla_planes_candidatos.csv")

    print("Tabla de probabilidades:")
    print(df_prob.to_string(index=False))
    print("\nTabla de riesgos:")
    print(df_riesgos.to_string(index=False))
    print("\nTabla de simulación:")
    print(df_sim.to_string(index=False))
    print("\nTabla de comparación de planes:")
    print(df_planes.to_string(index=False))

    # -------------------------------------------------------
    # Gráficas principales
    # -------------------------------------------------------
    imprimir_titulo("GRÁFICAS")

    graficar_oc(plan.n, plan.c, aql, ltpd, FIG_DIR / "curva_oc.png")
    graficar_aoq(plan.n, plan.c, FIG_DIR / "curva_aoq.png")
    graficar_asn(plan.n, FIG_DIR / "asn_constante.png")
    graficar_diferencia_modelos(plan.n, plan.c, N, FIG_DIR / "diferencia_modelos.png")
    graficar_histogramas(plan.n, escenarios, repeticiones, FIG_DIR / "histogramas")
    graficar_boxplot(plan.n, escenarios, repeticiones, FIG_DIR / "boxplot_proporcion_defectuosa.png")
    graficar_barras_decision(df_sim, FIG_DIR / "barras_aceptacion_rechazo.png")
    df_resumen_riesgos = graficar_resumen_riesgos(plan, aql, ltpd, FIG_DIR / "resumen_riesgos.png")
    ruta_resumen_riesgos = guardar_tabla(df_resumen_riesgos, "tabla_resumen_riesgos.csv")

    # -------------------------------------------------------
    # Diagramas del procedimiento y del sistema
    # -------------------------------------------------------
    imprimir_titulo("DIAGRAMAS")

    graficar_flujo_operativo(FIG_DIR / "flujo_operativo.png")
    graficar_sistema_calidad(FIG_DIR / "sistema_calidad.png")

    # -------------------------------------------------------
    # Resumen final en consola
    # -------------------------------------------------------
    imprimir_titulo("RESUMEN FINAL")

    print("Archivos generados:")
    print(f"- {ruta_prob}")
    print(f"- {ruta_riesgos}")
    print(f"- {ruta_sim}")
    print(f"- {ruta_planes}")
    print(f"- {ruta_resumen_riesgos}")
    print()
    print(f"Figuras guardadas en: {FIG_DIR.resolve()}")
    print(f"Tablas guardadas en:  {TAB_DIR.resolve()}")
    print()
    print("Listo. El script reprodujo los cálculos, las tablas, las simulaciones")
    print("y las gráficas del estudio por aceptación.")


# ===========================================================
# 11) PUNTO DE ENTRADA
# ===========================================================

if __name__ == "__main__":
    main()