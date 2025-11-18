import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from supabase import create_client
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from datetime import datetime, timezone
from uuid import uuid4

# Env vars de Vercel
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carga el modelo una vez (global para perf)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Función para test conexión (opcional)
def test_supabase():
    try:
        response = supabase.table('profiles').select('*').limit(1).execute()
        return True
    except Exception as e:
        raise Exception(f"Error Supabase: {e}")

# Tus funciones originales (corregidas)
def cargar_jovenes() -> pd.DataFrame:
    response = supabase.table('profiles').select('*', 'profiles_detail!inner(*)').eq('role', 'joven').execute()
    data = response.data or []
    flattened = []
    for row in data:
        detail = row.get('profiles_detail', {})
        flat_row = row.copy()
        flat_row.update({f'detail_{k}': v for k, v in detail.items() if k != 'user_id'})
        flattened.append(flat_row)
    return pd.DataFrame(flattened)

def cargar_oportunidades() -> pd.DataFrame:
    response = supabase.table('opportunities').select('*').eq('active', True).execute()
    data = response.data or []
    opps_list = []
    for opp in data:
        flat_opp = opp.copy()
        if opp.get('owner_type') == 'municipio' and opp.get('municipality_id'):
            mun_resp = supabase.table('municipalities').select('*').eq('id', opp['municipality_id']).execute()
            mun = mun_resp.data[0] if mun_resp.data else {}
            flat_opp.update({f'mun_{k}': v for k, v in mun.items() if k != 'id'})
            flat_opp['mun_name'] = mun.get('name', 'NA')
        elif opp.get('owner_type') == 'empresa' and opp.get('owner_id'):
            emp_resp = supabase.table('empresas').select('*').eq('id', opp['owner_id']).execute()
            emp = emp_resp.data[0] if emp_resp.data else {}
            flat_opp.update({f'emp_{k}': v for k, v in emp.items() if k != 'id'})
            flat_opp['emp_name'] = emp.get('name', 'NA')
        opps_list.append(flat_opp)
    return pd.DataFrame(opps_list)

def preprocesar_joven(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['experiencia_embedding'] = df['detail_experiencia'].apply(lambda x: model.encode(str(x)) if pd.notna(x) else np.zeros(384))
    df['motivaciones_embedding'] = df['detail_motivaciones'].apply(lambda x: model.encode(str(x)) if pd.notna(x) else np.zeros(384))
    df['ingresos_num'] = df['detail_ingresos_mensuales'].map(lambda x: {'bajo': 500, 'medio': 1000, 'alto': 2000}.get(str(x).lower(), 750))
    df['regiones_set'] = df['detail_preferencias_ubicacion'].apply(lambda x: set(x) if isinstance(x, list) else set())
    df['idiomas_set'] = df['detail_idiomas'].apply(lambda x: set(x) if isinstance(x, list) else set())
    df['situaciones_set'] = df['detail_situacion_personal'].apply(lambda x: set(x) if isinstance(x, list) else set())
    niveles_map = {'secundaria': 1, 'tecnico': 2, 'pregrado': 4, 'posgrado': 5}
    df['nivel_num'] = df['detail_nivel_estudios'].map(niveles_map).fillna(0)
    return df

def preprocesar_oportunidades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['desc_embedding'] = df['description'].apply(lambda x: model.encode(str(x)) if pd.notna(x) else np.zeros(384))
    df['regiones'] = df.apply(lambda row: [row.get('mun_province', '')] if row['owner_type'] == 'municipio' else [row.get('province', '')], axis=1)
    df['regiones_set'] = df['regiones'].apply(lambda x: set(x) if isinstance(x, list) else set())
    df['sectores_set'] = df.apply(lambda row: set(row.get('emp_sectors', [])) if row['owner_type'] == 'empresa' else set(row.get('mun_main_economic_sectors', [])), axis=1)
    df['nivel_requerido'] = df.apply(lambda row: 'tecnico' if 'agricultura' in str(row.get('emp_activity') or row.get('mun_main_economic_sectors', [])) else 'secundaria', axis=1)
    if 'accompaniment_services' in df.columns:
        df['apoyos_set'] = df['accompaniment_services'].apply(lambda x: set(str(x).lower().split()) if pd.notna(x) else set())
    else:
        df['apoyos_set'] = [set() for _ in range(len(df))]
    return df

def similitud_coseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def match_score_heuristic(usuario: Dict, oportunidad: Dict) -> Tuple[float, str, str, str, List[str]]:
    score = 0.0
    razones = []

    # Regiones (10%)
    user_regs = usuario.get('regiones_set', set())
    opp_regs = oportunidad.get('regiones_set', set())
    if user_regs and not (user_regs & opp_regs):
        return 0.0, oportunidad.get('title', 'Sin título'), 'descartado', '❌ No hay coincidencia geográfica', []
    jaccard = len(user_regs & opp_regs) / len(user_regs | opp_regs) if opp_regs else 0
    score += 0.10 * jaccard
    if jaccard > 0.5:
        razones.append(f"📍 Ubicación en {', '.join(opp_regs)} coincide con tu preferencia.")

    # Idioma (filtro duro)
    if 'ingles' in oportunidad.get('idiomas_requeridos', []) and 'ingles' not in usuario.get('idiomas_set', set()):
        return 0.0, oportunidad.get('title', 'Sin título'), 'descartado', '❌ Requiere inglés', []

    # Similitud semántica (50%)
    sim = similitud_coseno(usuario['experiencia_embedding'], oportunidad['desc_embedding'])
    score += 0.5 * sim
    if sim > 0.7:
        razones.append(f"🧠 Tu experiencia se alinea ({sim:.2f}).")

    # Requisitos (15%)
    req_text = f"{', '.join(oportunidad.get('requirements', []))} {oportunidad.get('description', '')}".lower()
    exp_text = f"{usuario.get('detail_experiencia', '')} {usuario.get('detail_titulacion', '')}".lower()
    fuzzy_sim = fuzz.ratio(exp_text, req_text) / 100.0
    if fuzzy_sim > 0.5:
        score += 0.15
        razones.append(f"✅ Cumples requisitos ({fuzzy_sim:.2f}).")

    # Formación (15%)
    niveles_map = {'secundaria': 1, 'tecnico': 2, 'pregrado': 4, 'posgrado': 5}
    user_nivel = niveles_map.get(usuario.get('detail_nivel_estudios', ''), 0)
    req_nivel = niveles_map.get(oportunidad.get('nivel_requerido', 'secundaria'), 1)
    if user_nivel >= req_nivel:
        score += 0.15
        razones.append(f"🎓 Nivel de estudios cumple.")

    # Ingresos (10%)
    user_ing = usuario.get('ingresos_num', 0)
    opp_econ = oportunidad.get('salario_o_precio', 0)
    if opp_econ >= user_ing * 1.2:
        score += 0.10
        razones.append(f"💰 Ingreso atractivo.")
    elif opp_econ >= user_ing * 0.8:
        score += 0.05
        razones.append(f"💰 Ingreso viable.")

    # Apoyos (5%)
    if usuario.get('situaciones_set', set()) & {'desempleado', 'vulnerabilidad'} and oportunidad.get('apoyos_set', set()) & {'mentorizacion', 'formacion'}:
        score += 0.05
        razones.append("🧩 Incluye apoyos útiles.")

    # Tipo de vínculo
    owner_type = oportunidad.get('owner_type', 'desconocido')
    mun_name = oportunidad.get('mun_name', 'NA')
    emp_name = oportunidad.get('emp_name', 'NA')
    if owner_type == 'municipio':
        tipo = 'municipio'
        razones.append(f"🏘️ Municipio {mun_name}.")
    elif owner_type == 'empresa':
        tipo = 'empresa' if not oportunidad.get('municipality_id') else 'combo'
        razones.append(f"🏢 Empresa {emp_name}.")
    else:
        tipo = 'desconocido'

    razon_final = '\n'.join(razones)
    return min(score, 1.0), oportunidad.get('title', 'Sin título'), tipo, razon_final, razones

# Handler para Vercel (responde a GET /api/match?user_id=...)
def main(request):
    try:
        # Test conexión
        test_supabase()

        user_id = request.query_params.get('user_id')
        if not user_id:
            return json.dumps({'error': 'user_id requerido'}), 400, {'Content-Type': 'application/json'}

        # Carga y procesa
        df_jovenes = preprocesar_joven(cargar_jovenes())
        df_opps = preprocesar_oportunidades(cargar_oportunidades())

        if user_id not in df_jovenes['id'].values:
            return json.dumps({'error': 'User no encontrado'}), 404, {'Content-Type': 'application/json'}

        usuario = df_jovenes[df_jovenes['id'] == user_id].iloc[0].to_dict()
        matches = []

        for _, opp_row in df_opps.iterrows():
            opp_dict = opp_row.to_dict()
            score, nombre_opp, tipo_match, razon_final, razones_lista = match_score_heuristic(usuario, opp_dict)

            if score > 0.2:
                match_data = {
                    'id': str(uuid4()),
                    'user_id': user_id,
                    'opportunity_id': opp_dict['id'],
                    'score': round(score, 3),
                    'status': 'pending',
                    'notes': razon_final,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                matches.append(match_data)

        # Inserta en Supabase
        if matches:
            supabase.table('matches').insert(matches).execute()

        return json.dumps({'matches': len(matches), 'data': matches}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}

# Export para Vercel
if __name__ == '__main__':
    # Para testing local: python api/match.py
    print("Handler listo para testing.")