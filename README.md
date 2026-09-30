# Mantenimiento predictivo de rodamientos con series temporales (PHM IEEE 2012)

## Problema

En máquinas rotativas, un rodamiento que falla sin aviso provoca una parada no planificada: pérdida de
producción, daños en ejes y reparaciones urgentes. El mantenimiento por calendario, a su vez, cambia piezas
que todavía sirven. El proyecto está pensado para equipos de mantenimiento de plantas industriales.

## Objetivo

A partir de la **serie temporal de vibración** (una captura cada 10 s), clasificar si el rodamiento ya entró
en su **zona de mantenimiento** (último 10 % de su vida útil) para programar el cambio antes de la falla.
**Métrica de éxito: F1** de la clase mantenimiento (equilibra detectar la zona de riesgo y no generar
falsas alarmas; el recall solo se puede "inflar" alarmando siempre).

## Datos

[PHM IEEE 2012 Data Challenge – PRONOSTIA](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset):
17 rodamientos llevados hasta la falla (`Learning\_set` + `Full\_Test\_Set`) en 3 condiciones de operación
(C1: 1800 rpm/4000 N, C2: 1650 rpm/4200 N, C3: 1500 rpm/5000 N). Cada captura: 2560 muestras de aceleración
horizontal y vertical a 25,6 kHz. El notebook descarga los datos solo (\~2 GB, requiere `git`) y resume cada
captura en una fila → 24 889 filas. `datos\_muestra.csv` = 100 filas de esa tabla (70 normales + 30 de
mantenimiento, de los rodamientos de test).

## Modelo

1. **Indicadores por captura** (por eje): RMS, pico, curtosis, asimetría, factor de cresta.
2. **Features de serie temporal** (por rodamiento, solo con el pasado): medias móviles de 1 y 5 min, valor
relativo al propio estado sano, tendencia, desvío en 5 min y horas de operación.
3. **Pipeline**: `ColumnTransformer` (`StandardScaler` + `OneHotEncoder` de la condición) → clasificador.
4. **Split por rodamiento**: 12 para train, 5 para test (`GroupShuffleSplit`, semilla 42), nunca se
mezclan capturas de un mismo rodamiento.
5. **Selección del modelo** con validación cruzada `GroupKFold` (5 folds) solo sobre train:

|Modelo (mismo pipeline)|F1 (CV)|Recall|Precisión|
|-|-|-|-|
|RandomForestClassifier (elegido)|**0.513**|0.620|0.437|
|HistGradientBoostingClassifier|0.473|0.655|0.370|
|LogisticRegression|0.448|0.746|0.320|
|RandomForest *sin* features temporales|0.291|0.425|0.222|
|DummyClassifier (azar)|0.089|0.089|0.089|

Las features temporales casi duplican el F1 del mismo modelo (0.29 → 0.51).

## Resultado (test: Bearing1\_1, 1\_2, 1\_6, 2\_5, 3\_2)

|Clase|Precisión|Recall|F1|
|-|-|-|-|
|normal|0.97|0.90|0.93|
|mantenimiento|0.46|0.77|0.57|

Con una alarma al superar 0,5 durante 30 s, los 5 rodamientos de test reciben aviso antes de fallar:
3 con 1,5–2,4 h de anticipación y 2 con solo 7–8 min. La feature más importante son las horas de operación,
seguida del RMS y el pico relativos al estado sano.

**ROI estimado** (100 rodamientos/año, costos supuestos en USD, sistema 30 000 USD/año, inspección por falsa
alarma 100 USD):

|Escenario|Costo por falla no planificada|Ahorro|ROI|
|-|-|-|-|
|Conservador|2 000|76 119|254 %|
|Base|5 000|306 614|1 022 %|
|Optimista|10 000|690 772|2 303 %|

**Próximos pasos:** features de frecuencia (FFT/envolvente), ajustar el umbral de alarma según el costo real
de una falsa alarma y regresión de vida útil remanente (RUL) con modelos secuenciales (LSTM).

## Cómo ejecutar

```bash
pip install scikit-learn==1.8.0 pandas numpy scipy matplotlib joblib streamlit jupyter
jupyter nbconvert --to notebook --execute notebook\_final.ipynb   # o abrirlo y "Run All" (\~2 min + descarga)
streamlit run app.py                                              # demo
```

Archivos: `notebook\_final.ipynb` (pipeline completo), `model.pkl` (pipeline entrenado, `joblib`),
`datos\_muestra.csv` (100 filas), `app.py` (demo Streamlit: filas de la muestra o capturas crudas `acc\_\*.csv`).

## Video Explicativo

Un breve video explicando de donde surge todo: [https://youtu.be/Jrh0KvwlCis](https://youtu.be/Jrh0KvwlCis)
