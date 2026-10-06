# Ejecuta esta celda en Colab DESPUÉS del entrenamiento y selección de los modelos.
import joblib
import numpy as np
import pandas as pd
import sklearn
import sys

# Opción predeterminada: regresión logística seleccionada en clasificación.
clasificador_app = modelos['Regresión logística']
umbral_app = float(umbral_final)
# Para usar XGBoost de clasificación en su lugar, sustituye las dos líneas anteriores por:
# clasificador_app = modelo_xgb
# umbral_app = float(umbral_xgb)

# Estos son los regresores ya seleccionados por objetivo.
regresores_app = modelos_finales_regresion

categoricas_app = train[features].select_dtypes(include=['object','string','category']).columns.tolist()
opciones = {c:sorted(train[c].dropna().astype(str).unique().tolist()) for c in categoricas_app}
# Ejemplo solo con entradas del formulario; no se exporta el dataset.
campos = list(dict.fromkeys(features + ['duracion','longitud_copy','hora','dia_semana']))
ejemplo = {}
for c in campos:
    if c not in train: continue
    v = train.iloc[0][c]
    if pd.isna(v): ejemplo[c] = None
    elif isinstance(v,np.generic): ejemplo[c] = v.item()
    else: ejemplo[c] = v
binarias = [c for c in features if c not in categoricas_app and set(train[c].dropna().unique()).issubset({0,1})]

bundle = {'features':list(features),'opciones':opciones,'ejemplo':ejemplo,
          'binarias':binarias,'clasificador':clasificador_app,
          'umbral':umbral_app,'regresores':regresores_app}
# Validar que todos los pipelines reciben la misma tabla.
X_comprobar = train[features].head(2)
clasificador_app.predict_proba(X_comprobar)
for modelo in regresores_app.values(): modelo.predict(X_comprobar)
joblib.dump(bundle,'modelos.joblib',compress=3)

# Fijar versiones para que la app use las mismas dependencias que Colab.
from importlib.metadata import version, PackageNotFoundError
req=['streamlit>=1.40,<2']
for paquete in ['numpy','pandas','scikit-learn','joblib','scipy']:
    req.append(f'{paquete}=={version(paquete)}')
try:
    req.append(f'xgboost=={version("xgboost")}')
except PackageNotFoundError:
    pass
with open('requirements.txt','w') as f: f.write('\n'.join(req)+'\n')
print('Python usado:',sys.version.split()[0])
from google.colab import files
files.download('modelos.joblib')
files.download('requirements.txt')
