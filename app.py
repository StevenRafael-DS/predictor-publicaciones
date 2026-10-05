from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title='Predictor de publicaciones', page_icon='📻', layout='centered')
st.title('Predictor de publicaciones')
st.caption('Estimaciones a 14 días · Proyecto académico con datos sintéticos')

@st.cache_resource
def cargar_modelos(ruta):
    return joblib.load(ruta)

ruta = Path(__file__).parent / 'modelos.joblib'
if not ruta.exists():
    st.info('Coloca modelos.joblib junto a app.py. Genera ese archivo en Colab usando exportar_colab.py.')
    st.stop()

bundle = cargar_modelos(str(ruta))
features = bundle['features']
opciones = bundle['opciones']
ejemplo = bundle['ejemplo']
transformadas = {'log_duracion', 'log_longitud_copy', 'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos', 'dia_semana_sin', 'dia_semana_cos'}
entradas = [c for c in features if c not in transformadas]
for derivada, original in [('log_duracion','duracion'),('log_longitud_copy','longitud_copy'),('hora_sin','hora'),('hora_cos','hora'),('dia_sin','dia_semana'),('dia_cos','dia_semana'),('dia_semana_sin','dia_semana'),('dia_semana_cos','dia_semana')]:
    if derivada in features and original not in entradas:
        entradas.append(original)

labels = {
    'radio_nombre_corto':'Radio', 'contenido_grupo':'Contenido', 'personalidad_grupo':'Artista o personalidad',
    'tipo_de_personalidad':'Tipo de personalidad', 'sub_tipo_personalidad':'Subtipo de personalidad',
    'longitud_copy':'Longitud del copy (caracteres)', 'duracion':'Duración (unidad usada al entrenar)',
    'hora':'Hora (0–23)', 'dia_semana':'Día de semana (lunes=0, domingo=6)',
    'mes_numero':'Mes (1–12)', 'horas_desde_publicacion_anterior':'Horas desde la publicación anterior'
}
limits = {'hora':(0,23), 'dia_semana':(0,6), 'mes_numero':(1,12)}
float_inputs = {'duracion','horas_desde_publicacion_anterior'}

st.write('Completa las características de la publicación antes de publicarla.')
with st.form('publicacion'):
    valores = {}
    columnas = st.columns(2)
    for idx,c in enumerate(entradas):
        label = labels.get(c, c.replace('_',' ').capitalize())
        with columnas[idx % 2]:
            if c in opciones:
                cats = opciones[c]
                valor = str(ejemplo.get(c,cats[0]))
                valores[c] = st.selectbox(label,cats,index=cats.index(valor) if valor in cats else 0)
            else:
                valor = ejemplo.get(c,0)
                valor = 0 if valor is None or pd.isna(valor) else float(valor)
                if c in limits:
                    lo,hi=limits[c]
                    valores[c]=st.number_input(label,min_value=lo,max_value=hi,value=int(np.clip(valor,lo,hi)),step=1)
                elif c in bundle.get('binarias',[]):
                    valores[c]=int(st.checkbox(label,value=bool(valor)))
                elif c in float_inputs:
                    valores[c]=st.number_input(label,min_value=0.0,value=max(0.0,valor),step=0.1)
                else:
                    valores[c]=st.number_input(label,min_value=0,value=max(0,int(valor)),step=1)
    enviado = st.form_submit_button('Predecir',type='primary',use_container_width=True)

if enviado:
    datos = pd.DataFrame([valores])
    # Las variables derivadas siguen exactamente las fórmulas del entrenamiento.
    if 'log_duracion' in features: datos['log_duracion']=np.log1p(datos['duracion'])
    if 'log_longitud_copy' in features: datos['log_longitud_copy']=np.log1p(datos['longitud_copy'])
    for sufijo,func in [('sin',np.sin),('cos',np.cos)]:
        if f'hora_{sufijo}' in features: datos[f'hora_{sufijo}']=func(2*np.pi*datos['hora']/24)
        for pref in ['dia','dia_semana']:
            if f'{pref}_{sufijo}' in features: datos[f'{pref}_{sufijo}']=func(2*np.pi*datos['dia_semana']/7)
    if 'formato' in datos and datos.loc[0,'formato'] in ['Imagen','Carrusel','Texto']:
        if 'duracion' in datos: datos['duracion']=0
        if 'log_duracion' in datos: datos['log_duracion']=0
    X=datos[features]
    try:
        score=float(bundle['clasificador'].predict_proba(X)[0,1])
        clasificacion='Viral' if score >= bundle['umbral'] else 'Normal'
        st.subheader('Resultado estimado')
        st.metric('Clasificación',clasificacion)
        st.caption(f'Score del modelo: {score:.3f} · Umbral: {bundle["umbral"]:.3f}. No es una probabilidad calibrada.')
        nombres={'ingreso_meta':'Monetización (USD)','t_visualizaciones':'Visualizaciones','interacciones':'Interacciones','alcance_meta':'Alcance'}
        resultados={}
        cols=st.columns(2)
        for idx,(objetivo,modelo) in enumerate(bundle['regresores'].items()):
            pred=max(0,float(modelo.predict(X)[0]))
            resultados[nombres.get(objetivo,objetivo)]=pred
            cols[idx%2].metric(nombres.get(objetivo,objetivo),f'USD {pred:,.2f}' if objetivo=='ingreso_meta' else f'{pred:,.0f}')
        st.caption('Estimaciones puntuales: no representan garantías ni intervalos de predicción.')
    except Exception as exc:
        st.error(f'No se pudo predecir. Revisa que las features y los modelos correspondan al mismo entrenamiento. Detalle: {exc}')
