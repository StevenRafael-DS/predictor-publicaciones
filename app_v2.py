from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Radar de publicaciones',page_icon='📻',layout='wide')
st.markdown('''<style>
.stApp {background:#f3f6fb;color:#15243b}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1250px}
h1,h2,h3 {letter-spacing:-.025em}
[data-testid="stMetric"] {background:white;border:1px solid #dce5f0;border-radius:14px;padding:20px;min-height:135px}
[data-testid="stMetricValue"] {font-size:2rem;color:#143b76}
[data-testid="stMetricLabel"] {font-size:1rem;color:#526580}
[data-testid="stForm"] {background:white;border:1px solid #dce5f0;border-radius:16px;padding:24px}
.stButton>button,.stFormSubmitButton>button {border-radius:10px;min-height:46px}
.banner {padding:24px;border-radius:16px;background:#112b50;color:white;margin:10px 0 24px}
.banner h2 {color:white;margin:0 0 8px;font-size:1.6rem}
.banner p {color:#dbeafe;margin:0;font-size:1rem}
.badge {display:inline-block;background:#e8eff9;color:#234b80;border-radius:20px;padding:6px 12px;font-size:.9rem;margin-bottom:12px}
.eyebrow {color:#607590;font-size:.9rem;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px}
</style>''',unsafe_allow_html=True)

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

st.markdown('<div class="eyebrow">Planificación editorial</div>',unsafe_allow_html=True)
st.title('Radar de publicaciones')
st.write('Evalúa una propuesta y estima su rendimiento a los 14 días.')

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

with st.sidebar:
    st.subheader('Cómo usarlo')
    st.write('1. Completa las tres secciones.\n\n2. Pulsa **Evaluar publicación**.\n\n3. Revisa el potencial y las estimaciones.')
    st.caption('Los valores iniciales corresponden a una publicación de ejemplo. Revísalos antes de evaluar tu propuesta.')
    st.divider()
    st.caption('Proyecto académico · Modelos entrenados con datos sintéticos. Las estimaciones no garantizan resultados reales.')

with st.form('publicacion'):
    st.subheader('Características de la propuesta')
    tabs=st.tabs([nombre for nombre,_ in grupos])
    for tab,(_,campos) in zip(tabs,grupos):
        with tab:
            cols=st.columns(2)
            for idx,c in enumerate(campos):
                with cols[idx%2]:
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

if 'resultado' in st.session_state:
    r=st.session_state['resultado'];viral=r['score']>=float(bundle['umbral']);v=r['valores']
    titulo='Potencial viral' if viral else 'Rendimiento normal'
    contexto=' · '.join(str(v.get(c,'')) for c in ['radio_nombre_corto','formato','contenido_grupo'] if v.get(c))
    st.markdown(f'<div class="banner"><h2>{titulo}</h2><p>{html.escape(contexto)} · Estimación a 14 días</p></div>',unsafe_allow_html=True)
    nombres={'ingreso_meta':'Monetización','t_visualizaciones':'Visualizaciones','interacciones':'Interacciones','alcance_meta':'Alcance'}
    cols=st.columns(4)
    for idx,objetivo in enumerate(nombres):
        pred=r['predicciones'].get(objetivo)
        cols[idx].metric(nombres[objetivo],'—' if pred is None else (f'USD {pred:,.2f}' if objetivo=='ingreso_meta' else f'{pred:,.0f}'))
    st.caption('Resultados de la última evaluación. Si modificas el formulario, pulsa Evaluar publicación nuevamente.')
    with st.expander('Detalle de la estimación'):
        st.write(f"Score de viralidad: **{r['score']:.3f}** · Umbral de decisión: **{float(bundle['umbral']):.3f}**")
        st.write('El score no es una probabilidad calibrada. Las cantidades son estimaciones puntuales, no intervalos ni garantías.')
    export={**v,**{f'pred_{k}':p for k,p in r['predicciones'].items()},'clasificacion':'Viral' if viral else 'Normal','score_viralidad':r['score']}
    st.download_button('Descargar estimación',pd.DataFrame([export]).to_csv(index=False).encode('utf-8-sig'),file_name='estimacion_publicacion.csv',mime='text/csv')
else:
    st.markdown('<div class="banner"><h2>Evalúa tu próxima publicación</h2><p>Completa el formulario para ver el potencial de viralidad y las cuatro estimaciones.</p></div>',unsafe_allow_html=True)
