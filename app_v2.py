from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Radar de publicaciones',page_icon='📻',layout='wide')
st.markdown("""<style>
.stApp {background:#f3f6fb;color:#15243b}
.block-container {padding-top:1rem;padding-bottom:.5rem;max-width:1600px}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stVerticalBlock"] {gap:.6rem}
[data-testid="stMetric"] {background:white;border:1px solid #dce5f0;border-radius:12px;padding:16px;min-height:105px}
[data-testid="stMetricValue"] {font-size:clamp(1.25rem,2.2vw,2rem);color:#143b76}
[data-testid="stMetricLabel"] {font-size:.95rem;color:#526580}
[data-testid="stForm"] {background:white;border:1px solid #dce5f0;border-radius:14px;padding:16px}
[data-testid="stWidgetLabel"] p {font-size:.875rem;line-height:1.2}
.stButton>button,.stFormSubmitButton>button {border-radius:9px;min-height:40px}
.banner {padding:20px;border-radius:14px;background:#112b50;color:white;margin:0 0 12px}
.banner.viral {background:#0b5148}
.banner h2 {color:white;margin:0 0 8px;font-size:1.5rem}
.banner p {color:#e2edf9;margin:0;font-size:.95rem;line-height:1.4}
.header {margin-bottom:12px}
.header h1 {font-size:1.8rem;margin:0 0 3px}
.header p {font-size:.95rem;color:#607590;margin:0}
@media (max-width:900px) {
.block-container {padding-top:1rem;padding-left:1rem;padding-right:1rem}
.header h1 {font-size:1.5rem}
}
@media (min-width:1100px) and (max-height:800px) {
.block-container {padding-top:.6rem}
[data-testid="stVerticalBlock"] {gap:.35rem}
[data-testid="stForm"] {padding:12px}
[data-testid="stForm"] [data-testid="stVerticalBlock"] {gap:.3rem}
}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def cargar_modelos(ruta):
    return joblib.load(ruta)


def preparar_entrada(valores,features):
    datos=pd.DataFrame([valores])
    if 'formato' in datos and datos.loc[0,'formato'] in ['Imagen','Carrusel','Texto']:
        datos['duracion']=0
    for derivada,original in [('log_duracion','duracion'),('log_longitud_copy','longitud_copy')]:
        if derivada in features: datos[derivada]=np.log1p(datos[original].clip(lower=0))
    for sufijo,func in [('sin',np.sin),('cos',np.cos)]:
        if f'hora_{sufijo}' in features:datos[f'hora_{sufijo}']=func(2*np.pi*datos['hora']/24)
        for prefijo in ['dia','dia_semana']:
            if f'{prefijo}_{sufijo}' in features:datos[f'{prefijo}_{sufijo}']=func(2*np.pi*datos['dia_semana']/7)
    return datos[features]

st.markdown('<div class="header"><h1>Radar de publicaciones</h1><p>Planificación editorial · Estimaciones a 14 días</p></div>',unsafe_allow_html=True)

ruta=Path(__file__).parent/'modelos.joblib'
if not ruta.exists():
    st.info('Falta modelos.joblib. Colócalo en la misma carpeta que app.py para activar las predicciones.')
    st.stop()
try:
    bundle=cargar_modelos(str(ruta))
except Exception as exc:
    st.error('No se pudieron cargar los modelos. Comprueba las versiones de requirements.txt.')
    with st.expander('Detalle del error'):st.code(str(exc))
    st.stop()

features=bundle['features'];opciones=bundle['opciones'];ejemplo=bundle['ejemplo']
transformadas={'log_duracion','log_longitud_copy','hora_sin','hora_cos','dia_sin','dia_cos','dia_semana_sin','dia_semana_cos'}
entradas=[c for c in features if c not in transformadas]
for derivada,original in [('log_duracion','duracion'),('log_longitud_copy','longitud_copy'),('hora_sin','hora'),('hora_cos','hora'),('dia_sin','dia_semana'),('dia_cos','dia_semana'),('dia_semana_sin','dia_semana'),('dia_semana_cos','dia_semana')]:
    if derivada in features and original not in entradas:entradas.append(original)

labels={'radio_nombre_corto':'Radio','contenido_grupo':'Tipo de contenido','personalidad_grupo':'Artista o personalidad','tipo_de_personalidad':'Tipo de personalidad','sub_tipo_personalidad':'Subtipo de personalidad','longitud_copy':'Longitud del texto (caracteres)','duracion':'Duración','hora':'Hora de publicación','dia_semana':'Día de publicación','mes_numero':'Mes','horas_desde_publicacion_anterior':'Horas desde la publicación anterior','tiene_texto':'Incluye texto','es_colaboracion':'Publicación en colaboración','tema_en_tendencia':'Tema en tendencia','gancho_fuerte':'Gancho fuerte','cantidad_imagenes':'Cantidad de imágenes','tiene_texto_visual':'Texto en imagen o video','tiene_rostro':'Aparecen rostros','tiene_subtitulos':'Incluye subtítulos','tiene_musica':'Incluye música','cantidad_personalidades':'Cantidad de personalidades','calidad_visual':'Calidad visual','tipo_gancho':'Tipo de gancho','tipo_visual':'Tipo de imagen o video','tema_especifico':'Tema específico'}
primarias={'radio_nombre_corto','formato','fuente','objetivo','hora','dia_semana','mes_numero','horas_desde_publicacion_anterior'}
contenido={'contenido_grupo','contenido','personalidad_grupo','personalidad','tipo_de_personalidad','sub_tipo_personalidad','genero','longitud_copy','tiene_texto','tema_especifico','tipo_gancho','tema_en_tendencia','gancho_fuerte'}
grupos=[('Publicación',[c for c in entradas if c in primarias]),('Contenido',[c for c in entradas if c in contenido]),('Imagen y video',[c for c in entradas if c not in primarias|contenido])]
valores={}

izquierda,derecha=st.columns([1.55,1],gap='large')

with izquierda:
    with st.form('publicacion'):
        st.markdown('**Características de la propuesta**')
        tabs=st.tabs([nombre for nombre,_ in grupos])
        for tab,(nombre_grupo,campos) in zip(tabs,grupos):
            with tab:
                ncols=3 if nombre_grupo=='Contenido' else 2
                cols=st.columns(ncols)
                for idx,c in enumerate(campos):
                    with cols[idx%ncols]:
                        label=labels.get(c,c.replace('_',' ').capitalize())
                        if c in opciones:
                            cats=opciones[c];v=str(ejemplo.get(c,cats[0]))
                            valores[c]=st.selectbox(label,cats,index=cats.index(v) if v in cats else 0,key='campo_'+c)
                        else:
                            v=ejemplo.get(c,0);v=0 if v is None or pd.isna(v) else float(v)
                            if c=='dia_semana':
                                valores[c]=st.selectbox(label,list(range(7)),index=int(np.clip(v,0,6)),format_func=lambda x:['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][x])
                            elif c=='mes_numero':
                                valores[c]=st.selectbox(label,list(range(1,13)),index=int(np.clip(v,1,12))-1,format_func=lambda x:['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][x-1])
                            elif c=='hora':
                                valores[c]=st.selectbox(label,list(range(24)),index=int(np.clip(v,0,23)),format_func=lambda x:f'{x:02d}:00')
                            elif c in bundle.get('binarias',[]):
                                valores[c]=int(st.checkbox(label,value=bool(v)))
                            elif c in ['duracion','horas_desde_publicacion_anterior']:
                                valores[c]=st.number_input(label,min_value=0.0,value=max(0.0,v),step=0.1,help='Usa la misma unidad del dataset con el que entrenaste.' if c=='duracion' else None)
                            else:
                                valores[c]=st.number_input(label,min_value=0,value=max(0,int(v)),step=1)
        st.caption('En Imagen, Carrusel y Texto la duración se considera 0 automáticamente.')
        enviado=st.form_submit_button('Evaluar publicación',type='primary',use_container_width=True)

    if enviado:
        try:
            X=preparar_entrada(valores,features)
            with st.spinner('Calculando estimaciones…'):
                score=float(bundle['clasificador'].predict_proba(X)[0,1])
                if not np.isfinite(score):raise ValueError('El clasificador devolvió un valor no válido.')
                resultados={}
                for objetivo,modelo in bundle['regresores'].items():
                    pred=float(modelo.predict(X)[0])
                    if not np.isfinite(pred):raise ValueError(f'Predicción no válida para {objetivo}.')
                    resultados[objetivo]=max(0,pred)
            st.session_state['resultado']={'score':score,'predicciones':resultados,'valores':valores.copy()}
        except Exception as exc:
            st.session_state.pop('resultado',None)
            st.error('No se pudo evaluar la publicación. Revisa las entradas y la compatibilidad de los modelos.')
            with st.expander('Detalle del error'):st.code(str(exc))

with derecha:
    st.markdown('**Resultado de la evaluación**')
    r=st.session_state.get('resultado')
    nombres={'ingreso_meta':'Monetización','t_visualizaciones':'Visualizaciones','interacciones':'Interacciones','alcance_meta':'Alcance'}
    if r:
        viral=r['score']>=float(bundle['umbral']);v=r['valores']
        titulo='Viral' if viral else 'Normal'
        contexto=' · '.join(str(v.get(c,'')) for c in ['radio_nombre_corto','formato','contenido_grupo'] if v.get(c))
        estilo='banner viral' if viral else 'banner'
        st.markdown(f'<div class="{estilo}"><h2>{titulo}</h2><p>{html.escape(contexto)}</p></div>',unsafe_allow_html=True)
        st.caption(f"Score: {r['score']:.3f} · Umbral: {float(bundle['umbral']):.3f}")
    else:
        st.markdown('<div class="banner"><h2>Por evaluar</h2><p>Completa las tres pestañas y pulsa Evaluar publicación.</p></div>',unsafe_allow_html=True)
    st.markdown('**Rendimiento estimado a 14 días**')
    objetivos=list(nombres)
    for inicio in [0,2]:
        cols=st.columns(2)
        for col,objetivo in zip(cols,objetivos[inicio:inicio+2]):
            pred=r['predicciones'].get(objetivo) if r else None
            valor='—' if pred is None else (f'USD {pred:,.2f}' if objetivo=='ingreso_meta' else f'{pred:,.0f}')
            col.metric(nombres[objetivo],valor)
    if r:
        st.caption('Última evaluación. Pulsa Evaluar publicación después de cambiar las entradas.')
        export={**r['valores'],**{f'pred_{k}':p for k,p in r['predicciones'].items()},'clasificacion':'Viral' if viral else 'Normal','score_viralidad':r['score']}
        st.download_button('Descargar estimación',pd.DataFrame([export]).to_csv(index=False).encode('utf-8-sig'),file_name='estimacion_publicacion.csv',mime='text/csv',use_container_width=True)
    st.caption('Proyecto académico con datos sintéticos. Estimaciones puntuales; el score no es una probabilidad calibrada.')
