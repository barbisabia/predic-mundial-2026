import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.set_page_config(page_title="Dashboard Prode 2026", layout="wide")

@st.cache_data(ttl=86400)
def obtener_elo_en_vivo():
# Diccionario de respaldo actualizado con los datos del archivo World.tsv (Mayo 2026)
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
    
    try:
        url = "https://www.eloratings.net/World.tsv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        respuesta = requests.get(url, headers=headers, timeout=10)
        respuesta.raise_for_status()
        
        lineas = respuesta.text.split('\n')
        elo_descargado = {}
        
        for linea in lineas:
            columnas = linea.split('\t')
            # En el TSV: [2] es el Código del País, [3] es el puntaje Elo
            if len(columnas) >= 4:
                codigo_pais = columnas[2].strip()
                try:
                    rating = int(columnas[3])
                    elo_descargado[codigo_pais] = rating
                except ValueError:
                    continue
                    
        # Mapeo exacto de códigos ISO/TSV a los nombres en tu dashboard
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
        
        elo_final = {}
        
        for codigo, eq_esp in traducciones.items():
            if codigo in elo_descargado:
                elo_final[eq_esp] = elo_descargado[codigo]
            else:
                elo_final[eq_esp] = elo_backup[eq_esp]
                
        return elo_final

    except Exception as e:
        return elo_backup

equipos_elo = obtener_elo_en_vivo()

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

def calcular_probabilidades(elo_a, elo_b):
    xg_a = max(0.1, 1.0 + (elo_a - elo_b) / 200)
    xg_b = max(0.1, 1.0 + (elo_b - elo_a) / 200)
    
    prob_a = [poisson.pmf(i, xg_a) for i in range(6)]
    prob_b = [poisson.pmf(i, xg_b) for i in range(6)]
    
    matriz = np.outer(prob_a, prob_b)
    gana_a = np.sum(np.tril(matriz, -1))
    empate = np.trace(matriz)
    gana_b = np.sum(np.triu(matriz, 1))
    
    max_idx = np.unravel_index(np.argmax(matriz), matriz.shape)
    goles_a_exacto, goles_b_exacto = max_idx
    prob_exacta = matriz[goles_a_exacto][goles_b_exacto] * 100
    
    return xg_a, xg_b, gana_a, empate, gana_b, matriz, goles_a_exacto, goles_b_exacto, prob_exacta

def simular_grupo_montecarlo(equipos_grupo, elo_dict, iteraciones=1000):
    resultados = {eq: [0, 0, 0, 0] for eq in equipos_grupo} 
    
    for _ in range(iteraciones):
        puntos = {eq: 0 for eq in equipos_grupo}
        goles_dif = {eq: 0 for eq in equipos_grupo}
        goles_fav = {eq: 0 for eq in equipos_grupo}
        
        for i in range(len(equipos_grupo)):
            for j in range(i+1, len(equipos_grupo)):
                eq_a = equipos_grupo[i]
                eq_b = equipos_grupo[j]
                
                xg_a = max(0.1, 1.0 + (elo_dict[eq_a] - elo_dict[eq_b]) / 200)
                xg_b = max(0.1, 1.0 + (elo_dict[eq_b] - elo_dict[eq_a]) / 200)
                
                goles_a = np.random.poisson(xg_a)
                goles_b = np.random.poisson(xg_b)
                
                goles_dif[eq_a] += (goles_a - goles_b)
                goles_dif[eq_b] += (goles_b - goles_a)
                goles_fav[eq_a] += goles_a
                goles_fav[eq_b] += goles_b
                
                if goles_a > goles_b: puntos[eq_a] += 3
                elif goles_b > goles_a: puntos[eq_b] += 3
                else:
                    puntos[eq_a] += 1
                    puntos[eq_b] += 1
        
        tabla = [(puntos[eq], goles_dif[eq], goles_fav[eq], eq) for eq in equipos_grupo]
        tabla.sort(reverse=True)
        
        for pos, data in enumerate(tabla):
            eq_name = data[3]
            resultados[eq_name][pos] += 1
            
    df_res = []
    for eq, pos_counts in resultados.items():
        df_res.append({
            "Equipo": eq,
            "1º Lugar": (pos_counts[0] / iteraciones) * 100,
            "2º Lugar": (pos_counts[1] / iteraciones) * 100,
            "3º Lugar": (pos_counts[2] / iteraciones) * 100,
            "4º Lugar": (pos_counts[3] / iteraciones) * 100
        })
    
    return pd.DataFrame(df_res).set_index("Equipo")

def simular_torneo_completo(elo_dict, grupos_dict, iteraciones=1000):
    estadisticas = {eq: {"16avos": 0, "Octavos": 0, "Cuartos": 0, "Semis": 0, "Final": 0, "Campeon": 0} for eq in elo_dict.keys()}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(iteraciones):
        if i % max(1, (iteraciones // 10)) == 0:
            progreso = int((i / iteraciones) * 100)
            progress_bar.progress(progreso)
            status_text.text(f"Simulando algoritmo... {progreso}%")
            
        posiciones_grupos = {}
        todos_los_terceros = []
        
        for nombre, equipos in grupos_dict.items():
            g_letra = nombre.split()[-1] 
            puntos = {eq: 0 for eq in equipos}
            goles_dif = {eq: 0 for eq in equipos}
            
            for m in range(len(equipos)):
                for n in range(m+1, len(equipos)):
                    eq_a, eq_b = equipos[m], equipos[n]
                    xg_a = max(0.1, 1.0 + (elo_dict[eq_a] - elo_dict[eq_b]) / 200)
                    xg_b = max(0.1, 1.0 + (elo_dict[eq_b] - elo_dict[eq_a]) / 200)
                    
                    goles_a = np.random.poisson(xg_a)
                    goles_b = np.random.poisson(xg_b)
                    
                    goles_dif[eq_a] += (goles_a - goles_b)
                    goles_dif[eq_b] += (goles_b - goles_a)
                    
                    if goles_a > goles_b: puntos[eq_a] += 3
                    elif goles_b > goles_a: puntos[eq_b] += 3
                    else:
                        puntos[eq_a] += 1
                        puntos[eq_b] += 1
                        
            tabla_grupo = [(puntos[eq], goles_dif[eq], eq) for eq in equipos]
            tabla_grupo.sort(reverse=True)
            
            posiciones_grupos[g_letra] = [tabla_grupo[0][2], tabla_grupo[1][2], tabla_grupo[2][2], tabla_grupo[3][2]]
            todos_los_terceros.append((tabla_grupo[2][0], tabla_grupo[2][1], tabla_grupo[2][2]))
            
        todos_los_terceros.sort(reverse=True)
        mejores_terceros = [t[2] for t in todos_los_terceros[:8]]
        
        p = posiciones_grupos
        t = mejores_terceros
        
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
        
        def jugar_match(eq_1, eq_2):
            xg_1 = max(0.1, 1.0 + (elo_dict[eq_1] - elo_dict[eq_2]) / 200)
            xg_2 = max(0.1, 1.0 + (elo_dict[eq_2] - elo_dict[eq_1]) / 200)
            prob_1 = xg_1 / (xg_1 + xg_2)
            return eq_1 if np.random.random() < prob_1 else eq_2

        ganadores_r32 = []
        for eq1, eq2 in partidos_r32:
            ganador = jugar_match(eq1, eq2)
            ganadores_r32.append(ganador)
            estadisticas[ganador]["16avos"] += 1
            
        ganadores_r16 = []
        for j in range(0, len(ganadores_r32), 2):
            ganador = jugar_match(ganadores_r32[j], ganadores_r32[j+1])
            ganadores_r16.append(ganador)
            estadisticas[ganador]["Octavos"] += 1
            
        ganadores_qf = []
        for j in range(0, len(ganadores_r16), 2):
            ganador = jugar_match(ganadores_r16[j], ganadores_r16[j+1])
            ganadores_qf.append(ganador)
            estadisticas[ganador]["Cuartos"] += 1
            
        ganadores_sf = []
        for j in range(0, len(ganadores_qf), 2):
            ganador = jugar_match(ganadores_qf[j], ganadores_qf[j+1])
            ganadores_sf.append(ganador)
            estadisticas[ganador]["Semis"] += 1
            
        for eq in ganadores_sf:
            estadisticas[eq]["Final"] += 1
            
        campeon = jugar_match(ganadores_sf[0], ganadores_sf[1])
        estadisticas[campeon]["Campeon"] += 1

    progress_bar.empty()
    status_text.empty()
    
    df_res = []
    for eq, stats in estadisticas.items():
        if stats["16avos"] > 0:
            df_res.append({
                "Equipo": eq,
                "16avos": (stats["16avos"] / iteraciones) * 100,
                "Octavos": (stats["Octavos"] / iteraciones) * 100,
                "Cuartos": (stats["Cuartos"] / iteraciones) * 100,
                "Semifinal": (stats["Semis"] / iteraciones) * 100,
                "Final": (stats["Final"] / iteraciones) * 100,
                "Campeón": (stats["Campeon"] / iteraciones) * 100
            })
            
    return pd.DataFrame(df_res).set_index("Equipo").sort_values(by="Campeón", ascending=False)

st.title("Dashboard Analítico - Mundial 2026")

tab_grupos, tab_resumen, tab_libre, tab_llaves, tab_montecarlo = st.tabs([
    "Simulador Grupos", "Resumen Automático", "Simulador Libre", "Llaves Eliminatorias", "Simulador Monte Carlo"
])

# --- PESTAÑA 1: SIMULADOR GRUPOS ---
with tab_grupos:
    st.subheader("Simular partido específico por Grupo")
    grupo_elegido = st.selectbox("Elegí la zona:", list(grupos.keys()), key="sel_g1")
    equipos_del_grupo = grupos[grupo_elegido]
    
    col1, col2 = st.columns(2)
    with col1:
        equipo_a_g = st.selectbox("Equipo Local (Grupo)", equipos_del_grupo)
    with col2:
        equipo_b_g = st.selectbox("Equipo Visitante (Grupo)", [e for e in equipos_del_grupo if e != equipo_a_g])

    if equipo_a_g and equipo_b_g:
        elo_a, elo_b = equipos_elo[equipo_a_g], equipos_elo[equipo_b_g]
        xg_a, xg_b, gana_a, empate, gana_b, matriz, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)

        st.divider()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(f"xG {equipo_a_g}", f"{xg_a:.2f}")
        
        if abs(elo_a - elo_b) <= 70:
            resultado_prode_g = "1 - 1 (Empate Técnico)"
        else:
            resultado_prode_g = f"{g_a} - {g_b}"
            
        kpi2.metric("Sugerencia Prode", resultado_prode_g)
        kpi3.metric(f"xG {equipo_b_g}", f"{xg_b:.2f}")

        col_grafico, col_heatmap = st.columns([1, 1])

        with col_grafico:
            st.subheader("Probabilidades Agrupadas (1X2)")
            df_probs = pd.DataFrame({
                "Resultado": [f"Gana {equipo_a_g}", "Empate", f"Gana {equipo_b_g}"],
                "Prob (%)": [gana_a * 100, empate * 100, gana_b * 100]
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Prob (%)", y="Resultado", data=df_probs, palette="mako", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_ylabel("")
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 2, p.get_y() + 0.5), va='center')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)

        with col_heatmap:
            st.subheader("Resultados Exactos")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.heatmap(matriz * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2)
            ax2.set_xlabel(f"Goles {equipo_b_g}", fontsize=10)
            ax2.set_ylabel(f"Goles {equipo_a_g}", fontsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            st.pyplot(fig2)

# --- PESTAÑA 2: RESUMEN AUTOMÁTICO ---
with tab_resumen:
    st.subheader("Predicciones Automáticas de la Fase de Grupos")
    grupo_resumen = st.selectbox("Elegí el grupo para ver la tabla completa:", list(grupos.keys()), key="sel_resumen")
    
    equipos = grupos[grupo_resumen]
    resultados_grupo = []
    
    for i in range(len(equipos)):
        for j in range(i + 1, len(equipos)):
            eq_a, eq_b = equipos[i], equipos[j]
            elo_a, elo_b = equipos_elo[eq_a], equipos_elo[eq_b]
            _, _, _, _, _, _, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)
            
            if abs(elo_a - elo_b) <= 70:
                pronostico = "Empate (1-1)"
            elif g_a > g_b:
                pronostico = f"Gana {eq_a} ({g_a}-{g_b})"
            elif g_b > g_a:
                pronostico = f"Gana {eq_b} ({g_a}-{g_b})"
            else:
                pronostico = f"Empate ({g_a}-{g_b})"
            
            resultados_grupo.append({
                "Partido": f"{eq_a} vs {eq_b}",
                "Pronóstico Exacto": pronostico,
                "Prob (Ese resultado)": f"{p_exacta:.1f}%"
            })
    
    st.table(pd.DataFrame(resultados_grupo).set_index("Partido"))

# --- PESTAÑA 3: SIMULADOR LIBRE ---
with tab_libre:
    st.subheader("Simulador Libre (Para cruces manuales)")
    col1, col2 = st.columns(2)
    with col1:
        equipo_a_l = st.selectbox("Equipo Local", nombres_equipos, key="loc_libre")
    with col2:
        equipo_b_l = st.selectbox("Equipo Visitante", nombres_equipos, index=1, key="vis_libre")

    if equipo_a_l and equipo_b_l and equipo_a_l != equipo_b_l:
        elo_a, elo_b = equipos_elo[equipo_a_l], equipos_elo[equipo_b_l]
        xg_a, xg_b, gana_a, empate, gana_b, matriz, g_a, g_b, p_exacta = calcular_probabilidades(elo_a, elo_b)

        st.divider()
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(f"xG {equipo_a_l}", f"{xg_a:.2f}")
        
        if abs(elo_a - elo_b) <= 70:
            resultado_prode_l = "1 - 1 (Empate Técnico)"
        else:
            resultado_prode_l = f"{g_a} - {g_b}"
            
        kpi2.metric("Sugerencia Prode", resultado_prode_l)
        kpi3.metric(f"xG {equipo_b_l}", f"{xg_b:.2f}")

        col_grafico, col_heatmap = st.columns([1, 1])

        with col_grafico:
            st.subheader("Probabilidades Agrupadas (1X2)")
            df_probs = pd.DataFrame({
                "Resultado": [f"Gana {equipo_a_l}", "Empate", f"Gana {equipo_b_l}"],
                "Prob (%)": [gana_a * 100, empate * 100, gana_b * 100]
            })
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Prob (%)", y="Resultado", data=df_probs, palette="mako", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_ylabel("")
            for i, p in enumerate(ax.patches):
                ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 2, p.get_y() + 0.5), va='center')
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)

        with col_heatmap:
            st.subheader("Resultados Exactos")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.heatmap(matriz * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2)
            ax2.set_xlabel(f"Goles {equipo_b_l}", fontsize=10)
            ax2.set_ylabel(f"Goles {equipo_a_l}", fontsize=10)
            ax2.tick_params(axis='both', which='major', labelsize=10)
            st.pyplot(fig2)

# --- PESTAÑA 4: LLAVES ELIMINATORIAS ---
with tab_llaves:
    st.subheader("Calculadora Visual de Llaves Eliminatorias")
    st.markdown("Armá tu cruce. Acá no hay empates: el modelo evalúa quién avanza de ronda.")
    
    col_izq, col_med, col_der = st.columns([2, 1, 2])
    
    with col_izq:
        eq_llave_1 = st.selectbox("Equipo 1", nombres_equipos, index=0, key="llave_1")
    with col_der:
        eq_llave_2 = st.selectbox("Equipo 2", nombres_equipos, index=1, key="llave_2")
        
    if eq_llave_1 and eq_llave_2 and eq_llave_1 != eq_llave_2:
        elo_1, elo_2 = equipos_elo[eq_llave_1], equipos_elo[eq_llave_2]
        
        _, _, gana_1, empate, gana_2, _, _, _, _ = calcular_probabilidades(elo_1, elo_2)
        
        total_victoria = gana_1 + gana_2
        prob_avanza_1 = (gana_1 / total_victoria) * 100
        prob_avanza_2 = (gana_2 / total_victoria) * 100
        
        with col_med:
            st.write("")
            st.write("")
            st.markdown("<h3 style='text-align: center;'>VS</h3>", unsafe_allow_html=True)
            
        st.divider()
        st.markdown("### Predicción de Clasificación")
        
        if prob_avanza_1 > prob_avanza_2:
            st.success(f"Avanza {eq_llave_1} (Probabilidad: {prob_avanza_1:.1f}%)")
        else:
            st.success(f"Avanza {eq_llave_2} (Probabilidad: {prob_avanza_2:.1f}%)")
            
        st.progress(int(prob_avanza_1))
        st.caption(f"<- {eq_llave_1} | {eq_llave_2} ->")

# --- PESTAÑA 5: SIMULADOR MONTE CARLO ---
with tab_montecarlo:
    st.subheader("Motor de Simulación Predictiva")
    
    tipo_simulacion = st.radio(
        "Seleccioná el tipo de análisis:", 
        ["Fase de Grupos (Detalle de posiciones)", "Torneo Completo (Llave oficial FIFA)"],
        horizontal=True
    )
    
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        num_simulaciones = st.slider(
            "Cantidad de escenarios paralelos:", 
            min_value=1000, 
            max_value=20000, 
            value=1000, 
            step=1000
        )
        
        if num_simulaciones > 5000:
            st.warning("Aviso: Procesar más de 5.000 universos paralelos puede demorar unos segundos.")
            
        if "Fase de Grupos" in tipo_simulacion:
            grupo_mc = st.selectbox("Elegí el grupo a simular:", list(grupos.keys()), key="sel_mc_g")
            btn_texto = f"Simular Grupo {num_simulaciones} veces"
        else:
            btn_texto = f"Correr {num_simulaciones} Mundiales"
            
        btn_simular = st.button(btn_texto, type="primary", use_container_width=True)
        
    with col_der:
        if btn_simular:
            if "Fase de Grupos" in tipo_simulacion:
                with st.spinner(f'Simulando el grupo {num_simulaciones} veces...'):
                    df_mc_grupo = simular_grupo_montecarlo(grupos[grupo_mc], equipos_elo, iteraciones=num_simulaciones)
                    st.success("¡Análisis completado!")
                    st.dataframe(
                        df_mc_grupo.style.background_gradient(cmap='Greens', axis=None).format("{:.1f}%"),
                        use_container_width=True
                    )
            else:
                with st.spinner(f'Procesando {num_simulaciones} torneos en la llave oficial...'):
                    df_torneo = simular_torneo_completo(equipos_elo, grupos, iteraciones=num_simulaciones)
                    st.success(f"¡Simulación de {num_simulaciones} escenarios completada con éxito!")
                    st.dataframe(
                        df_torneo.style.background_gradient(cmap='Blues', axis=None).format("{:.1f}%"),
                        use_container_width=True
                    )
