import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from prophet import Prophet # Importar Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from pandas.errors import ParserError
import io
import psycopg2
import traceback # Para mostrar errores detallados

# --- Configuración de Página ---
st.set_page_config(layout="wide")
st.title('Aplicación Interactiva de Análisis y Pronóstico de Series de Tiempo')

# --- Funciones Auxiliares ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True).encode('utf-8')

def load_data_from_file(uploaded_file):
    data = None; error_msg = None
    try:
        name = uploaded_file.name
        if name.endswith('.csv'):
            try: data = pd.read_csv(uploaded_file)
            except: uploaded_file.seek(0); data = pd.read_csv(uploaded_file, sep=';')
        elif name.endswith('.xlsx'): data = pd.read_excel(uploaded_file)
        elif name.endswith('.txt'):
            try: data = pd.read_csv(uploaded_file, sep=r'\s+')
            except:
                 uploaded_file.seek(0); stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                 try: data = pd.read_csv(stringio, sep=',')
                 except: uploaded_file.seek(0); data = pd.read_csv(stringio, sep=';')
        if data is None and name.endswith('.txt'): error_msg = "No se pudo interpretar TXT."
    except Exception as e: error_msg = f"Error al leer archivo: {e}"
    return data, error_msg

def load_data_from_db(db_params, query):
    data = None; error_msg = None; conn = None
    try:
        conn = psycopg2.connect(**db_params)
        data = pd.read_sql_query(query, conn)
    except Exception as e: error_msg = f"Error BD: {e}"
    finally:
        if conn: conn.close()
    return data, error_msg

def plot_forecast(original_data, forecast_df, value_col, title, color='blue', line_style='lines'):
    fig = go.Figure()
    line_mode = line_style if '+' not in line_style else 'lines+markers' # Ajuste para plotly
    marker_mode = 'markers' if '+' in line_style else None

    fig.add_trace(go.Scatter(x=original_data.index, y=original_data[value_col], mode='lines', name='Histórico', line=dict(color='grey')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Pronóstico'], mode=line_mode, name='Pronóstico', line=dict(color=color)))
    if 'Límite Inferior (95%)' in forecast_df.columns and 'Límite Superior (95%)' in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Límite Superior (95%)'], mode='lines', line=dict(width=0, color=color), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Límite Inferior (95%)'], mode='lines', line=dict(width=0, color=color), fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)', fill='tonexty', showlegend=False))
    fig.update_layout(title=title, xaxis_title='Fecha', yaxis_title=value_col)
    st.plotly_chart(fig, use_container_width=True)

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return np.nan, np.nan # Evitar errores si los arrays no coinciden o están vacíos
    # Asegurarse de que no haya ceros en y_true para MAPE
    y_true_safe = np.where(y_true == 0, 1e-6, y_true) # Reemplazar 0 con valor pequeño
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)
    return rmse, mape

def create_features(df, target_col, n_lags=3):
    df_feat = df.copy()
    for i in range(1, n_lags + 1): df_feat[f'lag_{i}'] = df_feat[target_col].shift(i)
    df_feat = df_feat.dropna(); return df_feat

# --- Barra Lateral: Controles ---
st.sidebar.header("1. Fuente y Selección de Datos")
data_source = st.sidebar.radio("Fuente de Datos", ["Archivo", "Base de Datos PostgreSQL"])

# Variables globales de datos
data_loaded = None # DataFrame original cargado
data_filtered = None # DataFrame después de aplicar filtros
data_processed = None # DataFrame listo para modelar (con BoxCox si aplica)
original_data_display = None # Datos filtrados originales para graficar
date_col = None
value_col = None
filter_col = None
lmbda = None

# Carga de Datos
if data_source == "Archivo":
    uploaded_file = st.sidebar.file_uploader("Carga CSV, XLSX o TXT", type=["csv", "xlsx", "txt"])
    if uploaded_file: data_loaded, error_msg = load_data_from_file(uploaded_file)
elif data_source == "Base de Datos PostgreSQL":
    # ... (código de conexión a BD igual que antes) ...
    st.sidebar.subheader("Parámetros de Conexión")
    db_host = st.sidebar.text_input("Host", "localhost"); db_port = st.sidebar.text_input("Puerto", "5432")
    db_name = st.sidebar.text_input("Nombre BD", "database_name"); db_user = st.sidebar.text_input("Usuario", "user")
    db_password = st.sidebar.text_input("Contraseña", type="password"); db_query = st.sidebar.text_area("Consulta SQL", "SELECT fecha, valor, categoria FROM mi_tabla ORDER BY fecha")
    if st.sidebar.button("Conectar y Cargar Datos"):
        db_params = {'host': db_host, 'port': db_port, 'dbname': db_name, 'user': db_user, 'password': db_password}
        with st.spinner("Consultando base de datos..."):
            data_loaded, error_msg = load_data_from_db(db_params, db_query)

# Mensaje de error si falla la carga
if 'error_msg' in locals() and error_msg: st.error(error_msg); data_loaded = None
elif data_source == "Base de Datos PostgreSQL" and data_loaded is None and st.sidebar.button("Conectar y Cargar Datos"): st.error("La consulta no devolvió datos.")


# Selección de Columnas y Filtros (si hay datos cargados)
if data_loaded is not None:
    st.sidebar.subheader("Selección de Columnas")
    available_cols = data_loaded.columns.tolist()
    date_col_default = next((c for c in available_cols if 'fecha' in c.lower() or 'date' in c.lower() or 'ds' in c.lower()), available_cols[0] if available_cols else None)
    value_col_default = next((c for c in available_cols if 'valor' in c.lower() or 'value' in c.lower() or 'y' in c.lower() or 'count' in c.lower()), available_cols[1] if len(available_cols) > 1 else None)
    date_col = st.sidebar.selectbox("Columna de Fecha/Tiempo ('ds')", available_cols, index=available_cols.index(date_col_default) if date_col_default else 0)
    value_col = st.sidebar.selectbox("Columna de Valor ('y')", available_cols, index=available_cols.index(value_col_default) if value_col_default else 1 if len(available_cols) > 1 else 0)

    st.sidebar.subheader("Filtro Opcional")
    filter_col_options = [None] + [col for col in available_cols if col not in [date_col, value_col] and data_loaded[col].nunique() < 100] # Columnas categóricas/id
    filter_col = st.sidebar.selectbox("Filtrar por Columna (Opcional)", filter_col_options)

    data_filtered = data_loaded.copy() # Empezar con datos cargados
    if filter_col:
        unique_values = data_loaded[filter_col].unique()
        selected_value = st.sidebar.selectbox(f"Selecciona Valor para '{filter_col}'", unique_values)
        data_filtered = data_loaded[data_loaded[filter_col] == selected_value].copy()
        st.sidebar.success(f"Filtrando por {filter_col} = {selected_value}")

    # Procesamiento Común (Fecha, Índice, Frecuencia, Limpieza)
    if date_col and value_col:
        try:
            df_proc = data_filtered[[date_col, value_col]].copy()
            df_proc.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True) # Renombrar para Prophet
            df_proc['ds'] = pd.to_datetime(df_proc['ds'])
            df_proc = df_proc.set_index('ds').sort_index()
            inferred_freq = pd.infer_freq(df_proc.index)
            if inferred_freq: df_proc = df_proc.asfreq(inferred_freq); st.sidebar.write(f"Frecuencia: {inferred_freq}")
            else: st.sidebar.warning("No se pudo inferir frecuencia.")
            df_proc = df_proc[['y']].dropna()

            if df_proc.empty: st.error("No quedan datos después de filtrar y limpiar."); data_processed = None
            else: st.success("Datos listos."); data_processed = df_proc; original_data_display = data_processed.copy() # Guardar para graficar
        except Exception as e: st.error(f"Error procesando columnas: {e}"); data_processed = None
    else: st.warning("Selecciona columnas Fecha y Valor."); data_processed = None

# --- Sección Principal ---
if data_processed is not None:
    st.header("2. Análisis Exploratorio y Visualización")
    st.subheader("Serie de Tiempo (Filtrada/Original)")
    # Opciones de Visualización
    st.sidebar.header("Ajustes de Visualización")
    plot_color = st.sidebar.color_picker("Color del Pronóstico", "#FF0000") # Rojo por defecto
    plot_style = st.sidebar.selectbox("Estilo de Línea", ['lines', 'markers', 'lines+markers'])
    # Graficar datos originales (filtrados pero antes de BoxCox)
    fig_original = px.line(original_data_display, x=original_data_display.index, y='y', title='Serie Original (después de filtros)')
    st.plotly_chart(fig_original, use_container_width=True)

    # Preprocesamiento
    st.sidebar.header("3. Preprocesamiento")
    apply_boxcox = st.sidebar.checkbox("Aplicar Transformación Box-Cox")
    default_period = 12 if data_processed.index.freq and 'M' in data_processed.index.freq.freqstr else 7 if data_processed.index.freq and 'D' in data_processed.index.freq.freqstr else 4
    periodo_estacional = st.sidebar.number_input("Período Estacional", min_value=2, value=default_period)

    data_model = data_processed.copy() # Datos para modelar
    if apply_boxcox:
        if (data_model['y'] <= 0).any():
            st.warning("Box-Cox requiere y > 0. Sumando constante."); min_val = data_model['y'].min(); shift = abs(min_val) + 1e-6 if min_val <= 0 else 1e-6
            data_model['y'] += shift
        try:
            data_model['y'], lmbda = boxcox(data_model['y']); st.sidebar.write(f"Box-Cox (Lambda={lmbda:.4f})")
            st.subheader("Serie Transformada (Box-Cox)"); fig_boxcox = px.line(data_model, x=data_model.index, y='y'); st.plotly_chart(fig_boxcox)
        except Exception as e: st.error(f"Error Box-Cox: {e}"); lmbda = None; apply_boxcox = False; data_model = data_processed.copy()
    else: lmbda = None

    # Descomposición
    st.subheader("Descomposición"); decomp_model = st.sidebar.selectbox("Modelo Descomposición", ['additive', 'multiplicative'])
    try:
        if len(data_model) > 2 * periodo_estacional:
            decomp = seasonal_decompose(data_model['y'].dropna(), model=decomp_model, period=periodo_estacional)
            st.plotly_chart(px.line(x=decomp.trend.index, y=decomp.trend, title='Tendencia')); st.plotly_chart(px.line(x=decomp.seasonal.index, y=decomp.seasonal, title='Estacionalidad')); st.plotly_chart(px.line(x=decomp.resid.index, y=decomp.resid, title='Residual'))
        else: st.warning(f"Datos insuficientes para descomponer (Período={periodo_estacional}).")
    except Exception as e: st.warning(f"No se pudo descomponer: {e}.")

    # --- Modelos ---
    st.header("4. Modelos de Pronóstico")
    st.sidebar.header("4. Configuración del Modelo")
    model_options = ["Prophet", "ARIMA", "SARIMA", "Suavizado Exponencial Simple", "Suavizado Exponencial Doble (Holt)", "Suavizado Exponencial Triple (Holt-Winters)", "Regresión Lineal (con Lags)"]
    model_type = st.sidebar.selectbox("Selecciona Modelo", model_options)
    forecast_steps = st.sidebar.number_input('Pasos a Pronosticar', min_value=1, value=periodo_estacional)

    model_params = {}; forecast_df = None

    # Parámetros específicos
    if model_type == "Prophet":
        model_params['seasonality_mode'] = st.sidebar.selectbox("Modo Estacionalidad", ('additive', 'multiplicative'))
        model_params['changepoint_prior_scale'] = st.sidebar.slider('Sensibilidad Changepoint', 0.001, 0.5, 0.05, step=0.001, format="%.3f")
        model_params['seasonality_prior_scale'] = st.sidebar.slider('Sensibilidad Estacionalidad', 0.01, 10.0, 10.0)
        model_params['yearly_seasonality'] = st.sidebar.checkbox("Estacionalidad Anual", value=True)
        model_params['weekly_seasonality'] = st.sidebar.checkbox("Estacionalidad Semanal", value=True)
        model_params['daily_seasonality'] = st.sidebar.checkbox("Estacionalidad Diaria", value=False)
    elif model_type == "ARIMA": # ... (igual que antes) ...
        model_params['p'] = st.sidebar.slider('p', 0, 5, 1); model_params['d'] = st.sidebar.slider('d', 0, 2, 1); model_params['q'] = st.sidebar.slider('q', 0, 5, 0)
    elif model_type == "SARIMA": # ... (igual que antes) ...
        model_params['p'] = st.sidebar.slider('p', 0, 5, 1); model_params['d'] = st.sidebar.slider('d', 0, 2, 1); model_params['q'] = st.sidebar.slider('q', 0, 5, 0)
        model_params['P'] = st.sidebar.slider('P', 0, 5, 1); model_params['D'] = st.sidebar.slider('D', 0, 2, 1); model_params['Q'] = st.sidebar.slider('Q', 0, 5, 0)
        model_params['s'] = periodo_estacional; st.sidebar.write(f"s = {model_params['s']}")
    elif "Suavizado Exponencial" in model_type: # ... (igual que antes) ...
        model_params['trend'] = st.sidebar.selectbox("Tendencia", [None, "add", "mul"], index=1 if "Doble" in model_type or "Triple" in model_type else 0)
        model_params['seasonal'] = st.sidebar.selectbox("Estacionalidad", [None, "add", "mul"], index=1 if "Triple" in model_type else 0)
        model_params['seasonal_periods'] = periodo_estacional if model_params['seasonal'] else None
        model_params['use_boxcox'] = st.sidebar.checkbox("Usar Box-Cox en HW", value=False) if lmbda is None else False
        if model_type == "Suavizado Exponencial Triple (Holt-Winters)": model_params['remove_bias'] = st.sidebar.checkbox("Remover Bias (HW)", value=False)
        model_params['initialization_method'] = 'estimated'
    elif model_type == "Regresión Lineal (con Lags)": # ... (igual que antes) ...
        model_params['n_lags'] = st.sidebar.slider('Lags', 1, periodo_estacional*2 if periodo_estacional else 12, 3)

    # Entrenar y Pronosticar
    if st.button(f"Entrenar {model_type} y Pronosticar"):
        try:
            with st.spinner(f'Procesando {model_type}...'):
                # Preparar datos (resetear índice para Prophet)
                model_data_run = data_model.copy()
                if model_type == "Prophet": model_data_run = model_data_run.reset_index() # Prophet necesita columna 'ds'

                # División train/test (secuencial)
                train_size = int(len(model_data_run) * 0.8)
                train = model_data_run[:train_size]
                test = model_data_run[train_size:]

                model_fit = None; forecast_result = None; test_predictions = None; future_forecast_df = None

                # --- Lógica de Modelado ---
                if model_type == "Prophet":
                    # Asegurarse que las columnas se llamen 'ds' y 'y'
                    train_prophet = train.rename(columns={train.columns[0]: 'ds', train.columns[1]: 'y'})
                    test_prophet = test.rename(columns={test.columns[0]: 'ds', test.columns[1]: 'y'})

                    model = Prophet(**{k: v for k, v in model_params.items() if k in ['seasonality_mode', 'changepoint_prior_scale', 'seasonality_prior_scale', 'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality']})
                    model.fit(train_prophet)
                    # Crear dataframe futuro para predicción
                    future_dates = model.make_future_dataframe(periods=len(test) + forecast_steps, freq=data_processed.index.freq)
                    forecast_result = model.predict(future_dates)
                    # Separar predicciones de test y futuro
                    test_predictions = forecast_result.iloc[train_size:len(model_data_run)][['ds', 'yhat']].set_index('ds')['yhat']
                    future_forecast_raw = forecast_result.iloc[len(model_data_run):][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
                    # Renombrar para consistencia
                    future_forecast_df = future_forecast_raw.rename(columns={'yhat': 'Pronóstico', 'yhat_lower': 'Límite Inferior (95%)', 'yhat_upper': 'Límite Superior (95%)'})

                # ... (Lógica para ARIMA, SARIMA, HW, Regresión Lineal similar a la versión anterior,
                #      asegurándose de usar 'train' y 'test' correctamente y generando
                #      'test_predictions' y 'future_forecast_df' con las columnas esperadas) ...

                elif model_type == "ARIMA":
                    model = ARIMA(train['y'], order=(model_params['p'], model_params['d'], model_params['q']))
                    model_fit = model.fit()
                    forecast_result = model_fit.get_forecast(steps=len(test) + forecast_steps)
                    test_predictions = forecast_result.predicted_mean[:len(test)]
                    future_preds = forecast_result.predicted_mean[len(test):]
                    conf_int = forecast_result.conf_int(alpha=0.05)[len(test):]
                    future_forecast_df = pd.DataFrame({'Pronóstico': future_preds, 'Límite Inferior (95%)': conf_int.iloc[:, 0], 'Límite Superior (95%)': conf_int.iloc[:, 1]}, index=pd.date_range(start=test.index[-1], periods=forecast_steps + 1, freq=data_processed.index.freq)[1:])

                # ... [Añadir lógica similar para SARIMA, HW, Linear Regression aquí, adaptando la salida a future_forecast_df] ...
                # Ejemplo simplificado para HW Triple:
                elif model_type == "Suavizado Exponencial Triple (Holt-Winters)":
                     model = ExponentialSmoothing(train['y'], trend=model_params['trend'], seasonal=model_params['seasonal'], seasonal_periods=model_params.get('seasonal_periods'), initialization_method=model_params['initialization_method'])
                     model_fit = model.fit(optimized=True, remove_bias=model_params.get('remove_bias', False))
                     test_predictions = model_fit.predict(start=test.index[0], end=test.index[-1])
                     future_preds = model_fit.forecast(steps=forecast_steps)
                     # HW no da intervalos de confianza por defecto
                     future_forecast_df = pd.DataFrame({'Pronóstico': future_preds}, index=pd.date_range(start=test.index[-1], periods=forecast_steps + 1, freq=data_processed.index.freq)[1:])


                st.success(f"Modelo {model_type} entrenado.")

                # --- Métricas ---
                if test_predictions is not None:
                    y_test_aligned = test.set_index('ds')['y'] if model_type == "Prophet" else test['y'] # Alinear y_test
                    y_test_aligned = y_test_aligned.reindex(test_predictions.index) # Asegurar mismo índice

                    y_test_eval = y_test_aligned
                    test_preds_eval = test_predictions
                    if apply_boxcox and lmbda is not None: # Revertir BoxCox para métricas
                        y_test_eval = inv_boxcox(y_test_aligned.dropna().to_numpy(), lmbda)
                        test_preds_eval = inv_boxcox(test_predictions.dropna().to_numpy(), lmbda)
                        if (original_data_display['y'] <= 0).any(): # Ajustar shift
                            shift = abs(original_data_display['y'].min()) + 1e-6 if original_data_display['y'].min() <=0 else 1e-6
                            y_test_eval -= shift; test_preds_eval -= shift

                    rmse, mape = calculate_metrics(y_test_eval, test_preds_eval)
                    st.subheader("Métricas (sobre conjunto de prueba)")
                    col1, col2 = st.columns(2); col1.metric("RMSE", f"{rmse:.4f}"); col2.metric("MAPE", f"{mape:.4%}")

                # --- Resultados y Visualización ---
                if future_forecast_df is not None:
                    forecast_df = future_forecast_df # Asignar para el resto del script
                    # Revertir BoxCox para el pronóstico final si aplica
                    if apply_boxcox and lmbda is not None:
                        forecast_df['Pronóstico'] = inv_boxcox(forecast_df['Pronóstico'].to_numpy(), lmbda)
                        if 'Límite Inferior (95%)' in forecast_df.columns: forecast_df['Límite Inferior (95%)'] = inv_boxcox(forecast_df['Límite Inferior (95%)'].to_numpy(), lmbda)
                        if 'Límite Superior (95%)' in forecast_df.columns: forecast_df['Límite Superior (95%)'] = inv_boxcox(forecast_df['Límite Superior (95%)'].to_numpy(), lmbda)
                        if (original_data_display['y'] <= 0).any(): # Ajustar shift
                            shift = abs(original_data_display['y'].min()) + 1e-6 if original_data_display['y'].min() <=0 else 1e-6
                            forecast_df -= shift
                        st.write("Pronóstico revertido a escala original.")

                    st.subheader(f"Pronóstico Futuro ({model_type})")
                    # Tabla Interactiva
                    st.dataframe(forecast_df) # Streamlit hace la tabla interactiva por defecto

                    # Botón Descarga
                    csv_export = convert_df_to_csv(forecast_df)
                    st.download_button("Descargar Pronóstico CSV", csv_export, f'pronostico_{model_type}.csv', 'text/csv')

                    # Visualización
                    plot_forecast(original_data_display, forecast_df, 'y', f'Pronóstico {model_type}', color=plot_color, line_style=plot_style)
                else:
                    st.warning("No se generó pronóstico futuro.")

        except Exception as e:
            st.error(f"Error modelando ({model_type}): {e}")
            st.error(traceback.format_exc())

else:
    st.info("Carga o conecta datos y selecciona columnas para comenzar.")