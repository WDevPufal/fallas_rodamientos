"""App de demo: streamlit run app.py  (requiere model.pkl y datos_muestra.csv en la misma carpeta)"""
import re
import numpy as np, pandas as pd, joblib, streamlit as st
from scipy.stats import kurtosis, skew

SENALES = ["rms_h", "rms_v", "pico_h", "pico_v", "curtosis_h", "curtosis_v", "factor_cresta_h", "factor_cresta_v"]
NUM = [f"{c}_{s}" for s in ("m6", "m30", "rel", "tend", "std30") for c in SENALES] + ["t_horas"]

modelo = joblib.load("model.pkl")
st.title("Mantenimiento predictivo de rodamientos")
st.caption("Clasifica si el rodamiento entró en su último 10 % de vida (zona de mantenimiento).")


def indicadores(x):
    rms = np.sqrt(np.mean(x**2)); pico = np.max(np.abs(x))
    return {"rms": rms, "pico": pico, "curtosis": kurtosis(x), "asimetria": skew(x), "factor_cresta": pico / rms}


def features_desde_capturas(archivos, condicion):
    """Mismas features temporales que el notebook, calculadas sobre las capturas subidas."""
    filas = []
    for f in sorted(archivos, key=lambda a: a.name):
        sep = ";" if ";" in f.getvalue()[:200].decode() else ","
        x = pd.read_csv(f, header=None, sep=sep).iloc[:, [4, 5]].to_numpy()
        num = int(re.search(r"(\d+)", f.name).group(1))          # acc_00001 → captura 1
        r = {"t_horas": (num - 1) * 10 / 3600}                   # una captura cada 10 s
        for eje, k in ((0, "h"), (1, "v")):
            r.update({f"{n}_{k}": v for n, v in indicadores(x[:, eje]).items()})
        filas.append(r)
    df = pd.DataFrame(filas)
    for c in SENALES:
        df[f"{c}_m6"] = df[c].rolling(6, min_periods=1).mean()
        df[f"{c}_m30"] = df[c].rolling(30, min_periods=1).mean()
        df[f"{c}_rel"] = df[f"{c}_m6"] / df[f"{c}_m30"].cummin()
        df[f"{c}_tend"] = df[f"{c}_m6"].diff(30).fillna(0)
        df[f"{c}_std30"] = df[c].rolling(30, min_periods=2).std().fillna(0)
    df["condicion"] = condicion
    return df


def mostrar(prob):
    st.metric("Probabilidad de zona de mantenimiento", f"{prob:.0%}")
    if prob > 0.5:
        st.error("⚠️ Programar mantenimiento: el rodamiento está cerca del final de su vida.")
    else:
        st.success("✅ Operación normal.")


tab1, tab2 = st.tabs(["Datos de muestra", "Archivos crudos acc_*.csv"])

with tab1:
    datos = pd.read_csv("datos_muestra.csv")
    i = st.selectbox("Fila de datos_muestra.csv", datos.index,
                     format_func=lambda k: f"{k}: {datos.rodamiento[k]} — {datos.vida_frac[k]:.0%} de vida")
    fila = datos.loc[[i]]
    mostrar(modelo.predict_proba(fila[NUM + ["condicion"]])[0, 1])
    st.write(f"Etiqueta real: **{'mantenimiento' if fila.mantenimiento.iloc[0] else 'normal'}** "
             f"(faltaban {fila.rul_min.iloc[0]:.0f} min para la falla)")

with tab2:
    cond = st.selectbox("Condición de operación", ["C1", "C2", "C3"],
                        help="C1: 1800 rpm/4000 N · C2: 1650 rpm/4200 N · C3: 1500 rpm/5000 N")
    archivos = st.file_uploader("Sube capturas consecutivas (ideal ≥ 30) de un mismo rodamiento",
                                type="csv", accept_multiple_files=True)
    if archivos:
        feats = features_desde_capturas(archivos, cond)
        probs = modelo.predict_proba(feats[NUM + ["condicion"]])[:, 1]
        mostrar(probs[-1])
        st.line_chart(pd.DataFrame({"probabilidad": probs}))
