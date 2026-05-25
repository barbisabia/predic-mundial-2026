import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from itertools import combinations

st.set_page_config(page_title="Dashboard Prode 2026", layout="wide")

# =====================================================================
#  PARAMETROS DEL MODELO  (un solo lugar para calibrar todo)
# =====================================================================
GOLES_TOTALES = 2.7        # goles totales esperados en un partido parejo (~2.6-2.8 en mundiales)
ESCALA_ELO = 800.0         # sensibilidad del modelo a la diferencia de Elo.
                           # 400 = escala Elo "pura" pero sobreconcentra al favorito en un
                           # torneo de 7 partidos. 800 da prob. de campeon realistas
                           # (favorito ~17%, en linea con predictores publicos). Subir = mas
                           # parejo/sorpresivo; bajar = mas predecible.
MAX_GOLES = 10             # tope de goles de la matriz; la cola perdida es ~0
VENTAJA_PENALES = 0.0      # 0.0 = penales 50/50. Ej 0.05 le da ventaja al favorito en la definicion
UMBRAL_EMPATE = 70         # diff de Elo bajo la cual el prode sugiere empate

# =====================================================================
#  MOTOR UNIFICADO  -  todo el dashboard consume de aca
# =====================================================================
def calcular_xg(elo_a, elo_b):
    """xG de cada equipo. Reparte un total de goles segun la cuota Elo
    logistica, asi las diferencias grandes no explotan a marcadores irreales."""
    p_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / ESCALA_ELO))
    xg_a = max(0.15, GOLES_TOTALES * p_a)
    xg_b = max(0.15, GOLES_TOTALES * (1 - p_a))
    return xg_a, xg_b


def matriz_resultado(elo_a, elo_b):
    """UNICA fuente de verdad: matriz de resultados exactos, normalizada a 1.
    Grupos, eliminatoria, resumen y Monte Carlo salen todos de aca."""
    xg_a, xg_b = calcular_xg(elo_a, elo_b)
    prob_a = poisson.pmf(np.arange(MAX_GOLES + 1), xg_a)
    prob_b = poisson.pmf(np.arange(MAX_GOLES + 1), xg_b)
    matriz = np.outer(prob_a, prob_b)
    matriz /= matriz.sum()
    return matriz, xg_a, xg_b


def probabilidades_1x2(matriz):
    """1X2 para GRUPOS: el empate es un resultado final."""
    gana_a = np.tril(matriz, -1).sum()
    empate = np.trace(matriz)
    gana_b = np.triu(matriz, 1).sum()
    return gana_a, empate, gana_b


def probabilidades_avanza(matriz, ventaja=VENTAJA_PENALES):
    """ELIMINATORIA: si o si avanza alguien. El empate se reparte (penales).
    P(avanza_a) = gana_a + empate*(0.5+ventaja). Suma 1, misma matriz que grupos."""
    gana_a, empate, gana_b = probabilidades_1x2(matriz)
    p_pen_a = 0.5 + ventaja
    return gana_a + empate * p_pen_a, gana_b + empate * (1 - p_pen_a)


def resultado_mas_probable(matriz):
    idx = np.unravel_index(np.argmax(matriz), matriz.shape)
    return int(idx[0]), int(idx[1]), matriz[idx] * 100


def muestrear_partido(matriz, rng):
    """Muestrea un marcador de la MISMA matriz (no una Poisson aparte)."""
    flat = matriz.ravel()
    i = rng.choice(len(flat), p=flat)
    return divmod(i, matriz.shape[1])


def muestrear_partido_cdf(cdf, n_cols, rng):
    """Muestrea marcador usando CDF precomputada (mas rapido en bucles grandes)."""
    i = np.searchsorted(cdf, rng.random(), side="right")
    return divmod(i, n_cols)


def _precomputar_cache_grupo(equipos, elo_dict):
    """Precalcula CDFs por cruce del grupo para acelerar Monte Carlo."""
    cache = {}
    for a, b in combinations(equipos, 2):
        m, _, _ = matriz_resultado(elo_dict[a], elo_dict[b])
        cache[(a, b)] = {"cdf": np.cumsum(m.ravel()), "n_cols": m.shape[1]}
    return cache


def _obtener_prob_avanza(eq_1, eq_2, elo_dict, cache_avanza):
    """Cache lazy de P(avanza eq_1) para cruces de eliminatoria."""
    key = (eq_1, eq_2)
    if key not in cache_avanza:
        m, _, _ = matriz_resultado(elo_dict[eq_1], elo_dict[eq_2])
        p1, _ = probabilidades_avanza(m)
        cache_avanza[key] = p1
    return cache_avanza[key]


def graficar_1x2(ax, etiquetas, valores):
    df = pd.DataFrame({"Resultado": etiquetas, "Prob (%)": valores})
    sns.barplot(x="Prob (%)", y="Resultado", data=df, hue="Resultado",
                palette="mako", legend=False, ax=ax)
    ax.set_xlim(0, 100)
    ax.set_ylabel("")
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.1f}%',
                    (p.get_width() + 2, p.get_y() + p.get_height() / 2), va='center')
    sns.despine(left=True, bottom=True)


def graficar_heatmap(ax, matriz, eq_a, eq_b, top=6):
    sub = matriz[:top, :top] * 100
    sns.heatmap(sub, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel(f"Goles {eq_b}", fontsize=10)
    ax.set_ylabel(f"Goles {eq_a}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)


# =====================================================================
#  ELO EN VIVO  (con aviso explicito si cae al backup)
# =====================================================================
@st.cache_data(ttl=86400)
def obtener_elo_en_vivo():
    elo_backup = {
        "España": 2165, "Argentina": 2113, "Francia": 2081, "Inglaterra": 2020,
        "Brasil": 1984, "Portugal": 1984, "Colombia": 1975, "Países Bajos": 1961,
        "Ecuador": 1933, "Croacia": 1930, "Alemania": 1923, "Noruega": 1912,
        "Japón": 1904, "Turquía": 1902, "Uruguay": 1892, "Suiza": 1889,
        "Senegal": 1878, "Bélgica": 1867, "México": 1860, "Paraguay": 1833,
        "Austria": 1827, "Marruecos": 1821, "Canadá": 1784, "Australia": 1783,
        "Irán": 1760, "Corea del Sur": 1752, "Argelia": 1743, "Estados Unidos": 1721,
        "Suecia": 1719, "República Checa": 1726, "Escocia": 1610, "Túnez": 1636,
        "Egipto": 1689, "Costa de Marfil": 1614, "Uzbekistán": 1727,
        "Bosnia y Herzegovina": 1594, "Panamá": 1737, "Ghana": 1503,
        "Arabia Saudita": 1568, "Qatar": 1425, "Nueva Zelanda": 1585,
        "Sudáfrica": 1524, "Jordania": 1690, "Cabo Verde": 1549, "Irak": 1607,
        "RD Congo": 1655, "Haití": 1532, "Curazao": 1436
    }
    traducciones = {
        "AR": "Argentina", "FR": "Francia", "BR": "Brasil", "ES": "España",
        "EN": "Inglaterra", "BE": "Bélgica", "NL": "Países Bajos", "DE": "Alemania",
        "PT": "Portugal", "UY": "Uruguay", "CO": "Colombia", "HR": "Croacia",
        "MX": "México", "US": "Estados Unidos", "SN": "Senegal", "MA": "Marruecos",
        "JP": "Japón", "DZ": "Argelia", "EC": "Ecuador", "PY": "Paraguay",
        "KR": "Corea del Sur", "CH": "Suiza", "AT": "Austria", "SE": "Suecia",
        "TR": "Turquía", "CZ": "República Checa", "NO": "Noruega", "AU": "Australia",
        "SC": "Escocia", "CA": "Canadá", "TN": "Túnez", "EG": "Egipto",
        "IR": "Irán", "CI": "Costa de Marfil", "UZ": "Uzbekistán",
        "BA": "Bosnia y Herzegovina", "PA": "Panamá", "GH": "Ghana",
        "SA": "Arabia Saudita", "QA": "Qatar", "NZ": "Nueva Zelanda",
        "ZA": "Sudáfrica", "JO": "Jordania", "CV": "Cabo Verde",
        "IQ": "Irak", "CD": "RD Congo", "HT": "Haití", "CW": "Curazao"
    }
    try:
        url = "https://www.eloratings.net/World.tsv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        respuesta = requests.get(url, headers=headers, timeout=10)
        respuesta.raise_for_status()

        elo_descargado = {}
        for linea in respuesta.text.split('\n'):
            columnas = linea.split('\t')
            if len(columnas) >= 4:
                try:
                    elo_descargado[columnas[2].strip()] = int(columnas[3])
                except ValueError:
                    continue

        elo_final, faltantes = {}, 0
        for codigo, nombre in traducciones.items():
            if codigo in elo_descargado:
                elo_final[nombre] = elo_descargado[codigo]
            else:
                elo_final[nombre] = elo_backup[nombre]
                faltantes += 1
        # si descargamos casi nada, no confiamos
        if len(elo_final) - faltantes < 10:
            return elo_backup, "backup"
        return elo_final, ("vivo" if faltantes == 0 else "parcial")

    except Exception:
        return elo_backup, "backup"


equipos_elo, fuente_elo = obtener_elo_en_vivo()

grupos = {
    "Grupo A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "Grupo B": ["Canadá", "Bosnia y Herzegovina", "Qatar", "Suiza"],
    "Grupo C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "Grupo D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "Grupo E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "Grupo F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "Grupo G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "Grupo H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "Grupo I": ["Francia", "Senegal", "Irak", "Noruega"],
    "Grupo J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "Grupo K": ["Portugal", "RD Congo", "Uzbekistán", "Colombia"],
    "Grupo L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}
nombres_equipos = sorted(list(equipos_elo.keys()))


# =====================================================================
#  SIMULACIONES MONTE CARLO  (muestrean de la misma matriz)
# =====================================================================
def _orden_grupo(equipos, elo_dict, rng, con_goles_fav=True, cache_grupo=None):
    """Juega un grupo y devuelve la tabla ordenada con criterios FIFA:
    puntos -> diferencia de gol -> goles a favor."""
    puntos = {eq: 0 for eq in equipos}
    gd = {eq: 0 for eq in equipos}
    gf = {eq: 0 for eq in equipos}
    for i in range(len(equipos)):
        for j in range(i + 1, len(equipos)):
            a, b = equipos[i], equipos[j]
            if cache_grupo is not None:
                data = cache_grupo[(a, b)]
                ga, gb = muestrear_partido_cdf(data["cdf"], data["n_cols"], rng)
            else:
                m, _, _ = matriz_resultado(elo_dict[a], elo_dict[b])
                ga, gb = muestrear_partido(m, rng)
            gd[a] += ga - gb; gd[b] += gb - ga
            gf[a] += ga; gf[b] += gb
            if ga > gb: puntos[a] += 3
            elif gb > ga: puntos[b] += 3
            else: puntos[a] += 1; puntos[b] += 1
    if con_goles_fav:
        tabla = [(puntos[eq], gd[eq], gf[eq], eq) for eq in equipos]
    else:
        tabla = [(puntos[eq], gd[eq], 0, eq) for eq in equipos]
    tabla.sort(reverse=True)
    return tabla


def simular_grupo_montecarlo(equipos_grupo, elo_dict, iteraciones=1000):
    rng = np.random.default_rng()
    cache_grupo = _precomputar_cache_grupo(equipos_grupo, elo_dict)
    resultados = {eq: [0, 0, 0, 0] for eq in equipos_grupo}
    for _ in range(iteraciones):
        tabla = _orden_grupo(equipos_grupo, elo_dict, rng, cache_grupo=cache_grupo)
        for pos, fila in enumerate(tabla):
            resultados[fila[3]][pos] += 1
    df = []
    for eq, c in resultados.items():
        df.append({"Equipo": eq,
                   "1º Lugar": c[0] / iteraciones * 100,
                   "2º Lugar": c[1] / iteraciones * 100,
                   "3º Lugar": c[2] / iteraciones * 100,
                   "4º Lugar": c[3] / iteraciones * 100})
    return pd.DataFrame(df).set_index("Equipo")


def jugar_eliminatoria(eq_1, eq_2, elo_dict, rng, cache_avanza=None):
    """Un cruce de eliminatoria: si o si avanza alguien, via prob. de avanzar."""
    if cache_avanza is None:
        cache_avanza = {}
    avanza_1 = _obtener_prob_avanza(eq_1, eq_2, elo_dict, cache_avanza)
    return eq_1 if rng.random() < avanza_1 else eq_2


def simular_torneo_completo(elo_dict, grupos_dict, iteraciones=1000):
    estadisticas = {eq: {"16avos": 0, "Octavos": 0, "Cuartos": 0,
                         "Semis": 0, "Final": 0, "Campeon": 0} for eq in elo_dict}
    rng = np.random.default_rng()
    cache_grupos = {
        nombre: _precomputar_cache_grupo(equipos, elo_dict)
        for nombre, equipos in grupos_dict.items()
    }
    cache_avanza = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for it in range(iteraciones):
        if it % max(1, iteraciones // 20) == 0:
            progress_bar.progress(int(it / iteraciones * 100))
            status_text.text(f"Simulando... {int(it / iteraciones * 100)}%")

        posiciones, terceros = {}, []
        for nombre, equipos in grupos_dict.items():
            g = nombre.split()[-1]
            tabla = _orden_grupo(equipos, elo_dict, rng, cache_grupo=cache_grupos[nombre])
            posiciones[g] = [fila[3] for fila in tabla]
            terceros.append((tabla[2][0], tabla[2][1], tabla[2][2], tabla[2][3]))

        terceros.sort(reverse=True)
        t = [x[3] for x in terceros[:8]]
        p = posiciones

        partidos_r32 = [
            (p["E"][0], t[0]), (p["I"][0], t[1]),
            (p["A"][1], p["B"][1]), (p["F"][0], p["C"][1]),
            (p["K"][1], p["L"][1]), (p["H"][0], p["J"][1]),
            (p["D"][0], t[2]), (p["G"][0], t[3]),
            (p["C"][0], p["F"][1]), (p["E"][1], p["I"][1]),
            (p["A"][0], t[4]), (p["L"][0], t[5]),
            (p["J"][0], p["H"][1]), (p["D"][1], p["G"][1]),
            (p["B"][0], t[6]), (p["K"][0], t[7])
        ]

        # Cada columna cuenta "LLEGO a esta ronda" (jugo esa instancia).
        # Los 32 clasificados llegan a 16avos:
        clasificados = [eq for par in partidos_r32 for eq in par]
        for eq in clasificados:
            estadisticas[eq]["16avos"] += 1

        # El ganador de cada cruce LLEGA a la ronda siguiente:
        ronda = [jugar_eliminatoria(e1, e2, elo_dict, rng, cache_avanza)
                 for e1, e2 in partidos_r32]
        for clave in ["Octavos", "Cuartos", "Semis", "Final"]:
            for eq in ronda:
                estadisticas[eq][clave] += 1
            ronda = [jugar_eliminatoria(ronda[j], ronda[j + 1], elo_dict, rng, cache_avanza)
                     for j in range(0, len(ronda), 2)]

        # ronda quedo con 1 equipo: el campeon
        estadisticas[ronda[0]]["Campeon"] += 1

    progress_bar.empty(); status_text.empty()

    df = []
    for eq, s in estadisticas.items():
        if s["16avos"] > 0:
            df.append({"Equipo": eq,
                       "16avos": s["16avos"] / iteraciones * 100,
                       "Octavos": s["Octavos"] / iteraciones * 100,
                       "Cuartos": s["Cuartos"] / iteraciones * 100,
                       "Semifinal": s["Semis"] / iteraciones * 100,
                       "Final": s["Final"] / iteraciones * 100,
                       "Campeón": s["Campeon"] / iteraciones * 100})
    return pd.DataFrame(df).set_index("Equipo").sort_values(by="Campeón", ascending=False)


# =====================================================================
#  INTERFAZ
# =====================================================================
st.title("Dashboard Analítico - Mundial 2026")

if fuente_elo == "vivo":
    st.caption("🟢 Elo actualizado en vivo desde eloratings.net")
elif fuente_elo == "parcial":
    st.caption("🟡 Elo en vivo (algunos equipos usan valores de respaldo)")
else:
    st.caption("🔴 Sin conexión a eloratings.net — usando Elo de respaldo (mayo 2026)")

tab_grupos, tab_resumen, tab_libre, tab_llaves, tab_montecarlo = st.tabs([
    "Simulador Grupos", "Resumen Automático", "Simulador Libre",
    "Llaves Eliminatorias", "Simulador Monte Carlo"
])


def render_partido(eq_a, eq_b, key):
    """Vista compartida de un partido (xG + sugerencia + 1X2 + heatmap)."""
    elo_a, elo_b = equipos_elo[eq_a], equipos_elo[eq_b]
    matriz, xg_a, xg_b = matriz_resultado(elo_a, elo_b)
    gana_a, empate, gana_b = probabilidades_1x2(matriz)
    g_a, g_b, _ = resultado_mas_probable(matriz)

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric(f"xG {eq_a}", f"{xg_a:.2f}")
    if abs(elo_a - elo_b) <= UMBRAL_EMPATE:
        sugerencia = f"{g_a} - {g_b} (parejo)"
    else:
        sugerencia = f"{g_a} - {g_b}"
    k2.metric("Sugerencia Prode", sugerencia)
    k3.metric(f"xG {eq_b}", f"{xg_b:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probabilidades (1X2)")
        fig, ax = plt.subplots(figsize=(6, 4))
        graficar_1x2(ax, [f"Gana {eq_a}", "Empate", f"Gana {eq_b}"],
                     [gana_a * 100, empate * 100, gana_b * 100])
        st.pyplot(fig); plt.close(fig)
    with c2:
        st.subheader("Resultados Exactos")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        graficar_heatmap(ax2, matriz, eq_a, eq_b)
        st.pyplot(fig2); plt.close(fig2)


# --- TAB 1: GRUPOS ---
with tab_grupos:
    st.subheader("Simular partido específico por Grupo")
    grupo_elegido = st.selectbox("Elegí la zona:", list(grupos.keys()), key="sel_g1")
    equipos_del_grupo = grupos[grupo_elegido]
    c1, c2 = st.columns(2)
    with c1:
        eq_a = st.selectbox("Equipo Local (Grupo)", equipos_del_grupo, key="ga")
    with c2:
        eq_b = st.selectbox("Equipo Visitante (Grupo)",
                            [e for e in equipos_del_grupo if e != eq_a], key="gb")
    if eq_a and eq_b:
        render_partido(eq_a, eq_b, "g")


# --- TAB 2: RESUMEN ---
with tab_resumen:
    st.subheader("Predicciones Automáticas de la Fase de Grupos")
    grupo_resumen = st.selectbox("Elegí el grupo:", list(grupos.keys()), key="sel_resumen")
    equipos = grupos[grupo_resumen]
    filas = []
    for i in range(len(equipos)):
        for j in range(i + 1, len(equipos)):
            a, b = equipos[i], equipos[j]
            matriz, _, _ = matriz_resultado(equipos_elo[a], equipos_elo[b])
            ga, gb, p_ex = resultado_mas_probable(matriz)
            gana_a, empate, gana_b = probabilidades_1x2(matriz)
            if abs(equipos_elo[a] - equipos_elo[b]) <= UMBRAL_EMPATE:
                pron = f"Parejo ({ga}-{gb})"
            elif ga > gb:
                pron = f"Gana {a} ({ga}-{gb})"
            elif gb > ga:
                pron = f"Gana {b} ({ga}-{gb})"
            else:
                pron = f"Empate ({ga}-{gb})"
            filas.append({"Partido": f"{a} vs {b}",
                          "Pronóstico Exacto": pron,
                          "Prob (Ese resultado)": f"{p_ex:.1f}%",
                          "1X2": f"{gana_a:.0%} / {empate:.0%} / {gana_b:.0%}"})
    st.table(pd.DataFrame(filas).set_index("Partido"))


# --- TAB 3: SIMULADOR LIBRE ---
with tab_libre:
    st.subheader("Simulador Libre (cruces manuales)")
    c1, c2 = st.columns(2)
    with c1:
        eq_a = st.selectbox("Equipo Local", nombres_equipos, key="loc_libre")
    with c2:
        eq_b = st.selectbox("Equipo Visitante", nombres_equipos, index=1, key="vis_libre")
    if eq_a and eq_b and eq_a != eq_b:
        render_partido(eq_a, eq_b, "l")


# --- TAB 4: LLAVES ELIMINATORIAS ---
with tab_llaves:
    st.subheader("Calculadora Visual de Llaves Eliminatorias")
    st.markdown("Acá no hay empate final: el modelo reparte la franja de empate "
                "como definición por penales y calcula quién avanza.")
    c_izq, c_med, c_der = st.columns([2, 1, 2])
    with c_izq:
        eq_1 = st.selectbox("Equipo 1", nombres_equipos, index=0, key="llave_1")
    with c_der:
        eq_2 = st.selectbox("Equipo 2", nombres_equipos, index=1, key="llave_2")

    if eq_1 and eq_2 and eq_1 != eq_2:
        matriz, _, _ = matriz_resultado(equipos_elo[eq_1], equipos_elo[eq_2])
        gana_1, empate, gana_2 = probabilidades_1x2(matriz)
        avanza_1, avanza_2 = probabilidades_avanza(matriz)

        with c_med:
            st.write(""); st.write("")
            st.markdown("<h3 style='text-align:center;'>VS</h3>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### Predicción de Clasificación")
        # mostramos de donde sale el numero: 90 min + definicion
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("En los 90' (1X2)")
            st.write(f"**{eq_1}** {gana_1:.0%} · Empate {empate:.0%} · **{eq_2}** {gana_2:.0%}")
        with cc2:
            st.caption("Probabilidad de AVANZAR (con definición)")
            st.write(f"**{eq_1}** {avanza_1:.0%} · **{eq_2}** {avanza_2:.0%}")

        if avanza_1 >= avanza_2:
            st.success(f"Avanza {eq_1} (Probabilidad: {avanza_1*100:.1f}%)")
        else:
            st.success(f"Avanza {eq_2} (Probabilidad: {avanza_2*100:.1f}%)")
        st.progress(int(round(np.clip(avanza_1 * 100, 0, 100))))
        st.caption(f"⟵ {eq_1}  |  {eq_2} ⟶")


# --- TAB 5: MONTE CARLO ---
with tab_montecarlo:
    st.subheader("Motor de Simulación Predictiva")
    tipo = st.radio("Tipo de análisis:",
                    ["Fase de Grupos (posiciones)", "Torneo Completo (llave FIFA)"],
                    horizontal=True)
    c_izq, c_der = st.columns([1, 2])
    with c_izq:
        n_sim = st.slider("Cantidad de simulaciones:", 1000, 20000, 2000, 1000)
        if n_sim > 5000:
            st.warning("Más de 5.000 simulaciones puede demorar unos segundos.")
        if "Fase de Grupos" in tipo:
            grupo_mc = st.selectbox("Grupo a simular:", list(grupos.keys()), key="sel_mc_g")
            btn_txt = f"Simular grupo {n_sim} veces"
        else:
            btn_txt = f"Correr {n_sim} Mundiales"
        btn = st.button(btn_txt, type="primary", use_container_width=True)
    with c_der:
        if btn:
            if "Fase de Grupos" in tipo:
                with st.spinner(f'Simulando el grupo {n_sim} veces...'):
                    df_g = simular_grupo_montecarlo(grupos[grupo_mc], equipos_elo, n_sim)
                st.success("¡Análisis completado!")
                st.dataframe(df_g.style.background_gradient(cmap='Greens', axis=None)
                             .format("{:.1f}%"), use_container_width=True)
            else:
                with st.spinner(f'Procesando {n_sim} torneos...'):
                    df_t = simular_torneo_completo(equipos_elo, grupos, n_sim)
                st.success(f"¡{n_sim} escenarios completados!")
                st.dataframe(df_t.style.background_gradient(cmap='Blues', axis=None)
                             .format("{:.1f}%"), use_container_width=True)
