import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import random
import os
from datetime import datetime

# NOMBRE DEL ARCHIVO MAESTRO
FILENAME = 'historico_bolita_MASTER.csv'

# ==========================================
# 1. MAPA DE ORDEN CRONOLÓGICO (Base ET)
# ==========================================
PRIORIDAD_TIEMPO_REAL = {
    ('TN', 'Morning'): 1, ('GA', 'Midday'): 2, ('NJ', 'Midday'): 3,
    ('TN', 'Midday'): 4, ('FL', 'Midday'): 5, ('NY', 'Midday'): 6,
    ('GA', 'Evening'): 7, ('TN', 'Evening'): 8, ('FL', 'Evening'): 9,
    ('NY', 'Evening'): 10, ('NJ', 'Evening'): 11, ('GA', 'Night'): 12
}

# ==========================================
# 2. CONFIGURACIÓN
# ==========================================
CONFIG_LOTERIAS = [
    {"estado": "TN", "url_p3": "https://loteria.guru/resultados-loteria-estados-unidos/us-cash-3-tn/resultados-anteriores-cash-3-tn-us", "url_p4": "https://loteria.guru/resultados-loteria-estados-unidos/us-cash-4-tn/resultados-anteriores-cash-4-tn-us"},
    {"estado": "GA", "url_p3": "https://loteria.guru/resultados-loteria-estados-unidos/us-cash-3-ga/resultados-anteriores-cash-3-ga-us", "url_p4": "https://loteria.guru/resultados-loteria-estados-unidos/us-cash-4-ga/resultados-anteriores-cash-4-ga-us"},
    {"estado": "NJ", "url_p3": "https://loteria.guru/resultados-loteria-estados-unidos/us-pick-3-nj/resultados-anteriores-pick-3-nj-us", "url_p4": "https://loteria.guru/resultados-loteria-estados-unidos/us-pick-4-nj/resultados-anteriores-pick-4-nj-us"},
    {"estado": "FL", "url_p3": "https://loteria.guru/resultados-loteria-estados-unidos/us-pick-3-fl/resultados-anteriores-pick-3-fl-us", "url_p4": "https://loteria.guru/resultados-loteria-estados-unidos/us-pick-4-fl/resultados-anteriores-pick-4-fl-us"},
    {"estado": "NY", "url_p3": "https://loteria.guru/resultados-loteria-estados-unidos/us-numbers-ny/resultados-anteriores-numbers-ny-us", "url_p4": "https://loteria.guru/resultados-loteria-estados-unidos/us-win-4-ny/resultados-anteriores-win-4-ny-us"}
]

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15'
]

MESES = {
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'ago': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dic': 12,
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

def get_random_header():
    return {'User-Agent': random.choice(USER_AGENTS)}

def parsear_fecha_final(texto_crudo):
    if not texto_crudo: return None
    txt = texto_crudo.lower().replace('.', ' ').replace(',', ' ').replace(' de ', ' ')
    try:
        match = re.search(r'(\d{1,2})\s+([a-z]{3,})\s*(\d{4})', txt)
        if match:
            dia, mes_txt, anio = match.groups()
            mes_num = MESES.get(mes_txt[:4], 0) 
            if mes_num == 0: mes_num = MESES.get(mes_txt[:3], 0)
            if mes_num > 0: return f"{anio}-{mes_num:02d}-{int(dia):02d}"
    except: pass
    return None

def solicitar_pagina_con_retry(url, max_retries=3):
    for intento in range(max_retries):
        try:
            response = requests.get(url, headers=get_random_header(), timeout=20)
            if response.status_code == 200: return response
            elif response.status_code == 404: return None
            else: time.sleep(5)
        except: time.sleep(5)
    return None

def scrapear_con_limite(base_url, game_type, fecha_limite_dt=None):
    """
    Si fecha_limite_dt existe, el scraper se detendrá al encontrar una fecha igual o menor (más vieja).
    """
    resultados = []
    page = 1
    stop_signal = False
    
    print(f"      >> Buscando nuevos datos {base_url.split('/')[-2]}...")
    
    while not stop_signal:
        url = f"{base_url}?page={page}"
        response = solicitar_pagina_con_retry(url)
        
        if not response: break
        
        soup = BeautifulSoup(response.content, 'html.parser')
        main_block = soup.find('div', class_='lg-lottery-older-results')
        if not main_block: break
            
        filas = main_block.find_all('div', class_='lg-line')
        if not filas: break

        items_en_pagina = 0
        
        for fila in filas:
            # Fecha
            dates = fila.find_all('div', class_='lg-date')
            fecha_str = None
            for d in dates:
                parsed = parsear_fecha_final(d.get_text(strip=True))
                if parsed:
                    fecha_str = parsed
                    break
            
            if not fecha_str: continue 
            
            # --- LÓGICA DE ACTUALIZACIÓN INCREMENTAL ---
            if fecha_limite_dt:
                fecha_actual_dt = datetime.strptime(fecha_str, "%Y-%m-%d")
                # Si la fecha que leemos es MAS VIEJA o IGUAL a la que ya tenemos, paramos.
                if fecha_actual_dt < fecha_limite_dt:
                    stop_signal = True
                    break # Salir del bucle for
            # -------------------------------------------

            # Números
            labels = fila.find_all('div', class_='lg-label')
            for label in labels:
                horario_txt = label.get_text(strip=True)
                if any(x in horario_txt for x in ['Bote', 'Jackpot', 'Prize', 'Cash 5']): continue
                
                numeros_div = label.find_next_sibling('div', class_='lg-center-flex')
                if numeros_div:
                    bolas = [li.get_text(strip=True) for li in numeros_div.find_all('li', class_='lg-number')]
                    bolas_digits = [b for b in bolas if b.isdigit()]
                    
                    if len(bolas_digits) >= game_type:
                        numero_final = "".join(bolas_digits[:game_type])
                        resultados.append({
                            'Fecha': fecha_str,
                            'Horario': horario_txt,
                            f'Result_{game_type}': numero_final
                        })
                        items_en_pagina += 1
        
        # Feedback reducido
        if page % 5 == 0: print(f"         ... escaneando pág {page}")
        
        if stop_signal:
            print(f"         ✅ Fecha límite alcanzada. Deteniendo descarga.")
            break
            
        if items_en_pagina == 0 and page > 1: # Si pagina vacia y no es la 1
            break
        
        page += 1
        time.sleep(random.uniform(1.0, 1.5)) # Pausa rápida

    return pd.DataFrame(resultados)

def gestionar_actualizacion():
    dfs_nuevos = []
    fecha_corte = None
    df_existente = pd.DataFrame()
    
    # 1. VERIFICAR SI EXISTE ARCHIVO
    if os.path.exists(FILENAME):
        print(f"📂 Archivo '{FILENAME}' detectado.")
        try:
            df_existente = pd.read_csv(FILENAME)
            if not df_existente.empty:
                # Ordenar por fecha para encontrar la última real
                df_existente['Fecha_DT'] = pd.to_datetime(df_existente['Fecha'])
                ultima_fecha = df_existente['Fecha_DT'].max()
                fecha_corte = ultima_fecha
                print(f"📅 Última fecha registrada: {ultima_fecha.strftime('%Y-%m-%d')}")
                print("⚡ MODALIDAD: ACTUALIZACIÓN (Solo bajando lo nuevo)")
        except Exception as e:
            print(f"⚠️ Error leyendo archivo existente: {e}. Se bajará todo de nuevo.")
    else:
        print("⚡ MODALIDAD: DESCARGA COMPLETA (Histórico 5 años)")

    # 2. SCRAPING (Con o sin límite)
    print("-------------------------------------------------------")
    
    for config in CONFIG_LOTERIAS:
        estado = config['estado']
        print(f"\n🌍 ESTADO: {estado}")
        
        # Buscamos datos nuevos hasta la fecha de corte (menos unos días de margen por seguridad)
        fecha_margen = None
        if fecha_corte:
            # Restamos 1 día al corte para asegurar que no se pierda nada por horas
            # (Los duplicados se eliminan después)
            fecha_margen = fecha_corte 

        df3 = scrapear_con_limite(config['url_p3'], 3, fecha_margen)
        df4 = scrapear_con_limite(config['url_p4'], 4, fecha_margen)
        
        if not df3.empty and not df4.empty:
            df_merged = pd.merge(df3, df4, on=['Fecha', 'Horario'], how='inner')
            if not df_merged.empty:
                df_merged['Estado'] = estado
                dfs_nuevos.append(df_merged)
                print(f"   ✅ {len(df_merged)} nuevos registros encontrados.")
            else:
                print("   ⚠️ Sin coincidencias nuevas.")
        else:
            print("   💤 Nada nuevo o error de conexión.")
        
        time.sleep(2)

    # 3. UNIFICACIÓN Y GUARDADO
    if not dfs_nuevos and df_existente.empty:
        print("\n❌ No hay datos para guardar.")
        return

    print("\n🔮 Procesando y Reordenando Base de Datos...")
    
    # Unir lo nuevo con lo viejo
    if dfs_nuevos:
        df_new_master = pd.concat(dfs_nuevos, ignore_index=True)
        # Calcular columnas calculadas para lo nuevo
        df_new_master['Centena'] = df_new_master['Result_3'].apply(lambda x: x[0])
        df_new_master['Fijo'] = df_new_master['Result_3'].apply(lambda x: x[1:])
        df_new_master['Corrido_1'] = df_new_master['Result_4'].apply(lambda x: x[:2])
        df_new_master['Corrido_2'] = df_new_master['Result_4'].apply(lambda x: x[2:])
        
        # Concatenar con el existente
        if not df_existente.empty:
            # Asegurarse que df_existente no tenga columna Fecha_DT auxiliar
            if 'Fecha_DT' in df_existente.columns:
                df_existente = df_existente.drop(columns=['Fecha_DT'])
            
            master = pd.concat([df_existente, df_new_master], ignore_index=True)
        else:
            master = df_new_master
    else:
        master = df_existente
        if 'Fecha_DT' in master.columns: master = master.drop(columns=['Fecha_DT'])
        print("✅ No hubo datos nuevos web, pero reordenaremos el archivo local.")

    # 4. LIMPIEZA DE DUPLICADOS (Crucial en actualizaciones)
    total_antes = len(master)
    master = master.drop_duplicates(subset=['Fecha', 'Estado', 'Horario'])
    print(f"🧹 Duplicados eliminados: {total_antes - len(master)}")

    # 5. ORDENAMIENTO INVERTIDO (VIEJO ARRIBA -> NUEVO ABAJO)
    master['Ranking_Tiempo'] = master.apply(
        lambda row: PRIORIDAD_TIEMPO_REAL.get((row['Estado'], row['Horario']), 99), axis=1
    )
    
    # AQUÍ ESTÁ EL CAMBIO: ascending=[True, True]
    # Fecha True = 2020... 2021... 2025 (Cronológico)
    master = master.sort_values(by=['Fecha', 'Ranking_Tiempo'], ascending=[True, True])
    
    cols = ['Fecha', 'Estado', 'Horario', 'Centena', 'Fijo', 'Corrido_1', 'Corrido_2']
    final_df = master[cols]
    
    final_df.to_csv(FILENAME, index=False)
    
    print("\n" + "="*50)
    print(f"✅ BASE DE DATOS ACTUALIZADA EXITOSAMENTE")
    print(f"📂 Archivo: {FILENAME}")
    print(f"📊 Total Registros: {len(final_df)}")
    print(f"📅 Rango: {final_df.iloc[0]['Fecha']} -> {final_df.iloc[-1]['Fecha']}")
    print("="*50)
    print("Muestra (Últimos registros - Lo más nuevo al final):")
    print(final_df.tail(15).to_string(index=False))

if __name__ == "__main__":
    gestionar_actualizacion()
