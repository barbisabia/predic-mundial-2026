import streamlit as st
import numpy as np
from scipy.stats import poisson
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Prode 2026", layout="wide")

equipos_elo = {
    "Argentina": 2140, "Francia": 2110, "Brasil": 2080, "España": 2040, 
    "Inglaterra": 2030, "Bélgica": 2000, "Países Bajos": 1980, "Alemania": 1970, 
    "Portugal": 1960, "Uruguay": 1940, "Colombia": 1920, "Croacia": 1880, 
    "México": 1850, "Estados Unidos": 1830, "Senegal": 1780, "Marruecos": 1770, 
    "Japón": 1760, "Argelia": 1750, "Ecuador": 1740, "Paraguay": 1700,
    "Corea del Sur": 1690, "Suiza": 1680, "Austria": 1670, "Suecia": 1660,
    "Turquía": 1650, "República Checa": 1640, "Noruega": 1630, "Australia": 1620,
    "Escocia": 1610, "Canadá": 1600, "Túnez": 1590, "Egipto": 1580,
    "Irán": 1570, "Costa de Marfil": 1560, "Uzbekistán": 1550, 
    "Bosnia y Herzegovina": 1540, "Panamá": 1530, "Ghana": 1520, 
    "Arabia Saudita": 1510, "Qatar": 1500, "Nueva Zelanda": 1490, 
    "Sudáfrica": 1480, "Jordania": 1470, "Cabo Verde": 1460, 
    "Irak": 1450, "RD Congo": 1440, "Haití": 1430, "Curazao": 1420
}

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
                
                if goles_a > goles_b:
                    puntos[eq_a] += 3
                elif goles_b > goles_a:
                    puntos[eq_b] += 3
                else:
                    puntos[eq_a] += 1
                    puntos[eq_b] += 1
        
        tabla = []
        for eq in equipos_grupo:
            tabla.append((puntos[eq], goles_dif[eq], goles_fav[eq], eq))
        
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
    
    df = pd.DataFrame(df_res).set_index("Equipo")
    return df

def simular_torneo_completo(elo_dict, grupos_dict, iteraciones=1000):
    estadisticas = {eq: {"Octavos": 0, "Cuartos": 0, "Semis": 0, "Final": 0, "Campeon": 0} for eq in elo_dict.keys()}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(iteraciones):
        if i % max(1, (iteraciones // 10)) == 0:
            progreso = int((i / iteraciones) * 100)
            progress_bar.progress(progreso)
            status_text.text(f"Simulando torneo... {progreso}%")
            
        clasificados_grupos = []
        
        for nombre, equipos in grupos_dict.items():
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
                        
            tabla = [(puntos[eq], goles_dif[eq], eq) for eq in equipos]
            tabla.sort(reverse=True)
            clasificados_grupos.extend([tabla[0][2], tabla[1][2]])
            
        faltantes = 32 - len(clasificados_grupos)
        terceros_disponibles = [eq for eq in elo_dict.keys() if eq not in clasificados_grupos][:faltantes]
        llave_32 = clasificados_grupos + terceros_disponibles
        
        def jugar_fase(equipos_activos):
            siguiente_ronda = []
            for j in range(0, len(equipos_activos), 2):
                eq_1 = equipos_activos[j]
                eq_2 = equipos_activos[j+1]
                xg_1 = max(0.1, 1.0 + (elo_dict[eq_1] - elo_dict[eq_2]) / 200)
                xg_2 = max(0.1, 1.0 + (elo_dict[eq_2] - elo_dict[eq_1]) / 200)
                
                prob_1 = xg_1 / (xg_1 + xg_2)
                if np.random.random() < prob_1:
                    siguiente_ronda.append(eq_1)
                else:
                    siguiente_ronda.append(eq_2)
            return siguiente_ronda

        octavos = jugar_fase(llave_32)
        for eq in octavos: estadisticas[eq]["Octavos"] += 1
            
        cuartos = jugar_fase(octavos)
        for eq in cuartos: estadisticas[eq]["Cuartos"] += 1
            
        semis = jugar_fase(cuartos)
        for eq in semis: estadisticas[eq]["Semis"] += 1
            
        finalistas = jugar_fase(semis)
        for eq in finalistas: estadisticas[eq]["Final"] += 1
            
        campeon = jugar_fase(finalistas)
        estadisticas[campeon[0]]["Campeon"] += 1

    progress_bar.empty()
    status_text.empty()
    
    df_res = []
    for eq, stats in estadisticas.items():
        if stats["Octavos"] > 0:
            df_res.append({
                "Equipo": eq,
                "Octavos": (stats["Octavos"] / iteraciones) * 100,
                "Cuartos": (stats["Cuartos"] / iteraciones) * 100,
                "Semifinal": (stats["Semis"] / iteraciones) * 100,
                "Final": (stats["Final"] / iteraciones) * 100,
                "Campeón": (stats["Campeon"] / iteraciones) * 100
            })
            
    df = pd.DataFrame(df_res).set_index("Equipo").sort_values(by="Campeón", ascending=False)
    return df

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
    st.markdown("Armá tu cruce. Acá no hay empates: el modelo evalúa quién avanza de ronda (por victoria en los 90 minutos, alargue o penales).")
    
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
    st.subheader("Motor de Simulación Monte Carlo")
    
    tipo_simulacion = st.radio(
        "Seleccioná el tipo de simulación:", 
        ["Fase de Grupos (Detalle de posiciones)", "Torneo Completo (Probabilidades de campeonato)"],
        horizontal=True
    )
    
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        num_simulaciones = st.slider(
            "Cantidad de simulaciones:", 
            min_value=1000, 
            max_value=20000, 
            value=1000, 
            step=1000
        )
        
        if num_simulaciones > 5000:
            st.warning("Aviso: Ejecutar más de 5.000 simulaciones puede tardar dependiendo de la capacidad del servidor.")
            
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
                    st.success("¡Simulación completada!")
                    st.dataframe(
                        df_mc_grupo.style.background_gradient(cmap='Greens', axis=None).format("{:.1f}%"),
                        use_container_width=True
                    )
            else:
                df_torneo = simular_torneo_completo(equipos_elo, grupos, iteraciones=num_simulaciones)
                st.success(f"¡Simulación de {num_simulaciones} escenarios completada!")
                st.dataframe(
                    df_torneo.style.background_gradient(cmap='Blues', axis=None).format("{:.1f}%"),
                    use_container_width=True
                )