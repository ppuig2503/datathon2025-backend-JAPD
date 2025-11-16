# Datathon 2025 Backend - JAPD

Backend API para predicción y explicabilidad de oportunidades de venta usando Machine Learning (LightGBM).

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
  - [Modo Desarrollo (Swagger)](#modo-desarrollo-swagger)
  - [Conexión con Frontend](#conexión-con-frontend)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Endpoints Disponibles](#endpoints-disponibles)
- [Flujo de Uso](#flujo-de-uso)

## 📝 Descripción

Esta API REST proporciona:
- **Predicción**: Predicción binaria (GANADA/PERDIDA) de oportunidades de venta
- **Explicabilidad Local**: LIME y SHAP local para explicar predicciones individuales
- **Explicabilidad Global**: SHAP global y Partial Dependence Plots (PDP)
- **Generación de Textos**: Explicaciones en lenguaje natural usando OpenAI GPT-4

## 🔧 Requisitos Previos

- **Python 3.8+**
- **pip** (gestor de paquetes de Python)
- **API Key de OpenAI** (para endpoints de explicabilidad con IA)

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd datathon2025-backend-JAPD
```

### 2. Instalar dependencias

Ejecuta el script de instalación de dependencias:

**Linux / Mac / Git Bash:**
```bash
bash install_deps.sh
```

**Windows (PowerShell):**
```powershell
bash install_deps.sh
```

**Windows (cmd) - instalación manual:**
```cmd
pip install fastapi uvicorn pydantic pandas joblib scikit-learn lightgbm openai httpx python-dotenv scikit-image lime shap pdpbox
```

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_clave_de_openai_aqui
```

> **Nota**: Solicita la API key de OpenAI al equipo o crea una en [platform.openai.com](https://platform.openai.com)

## 🚀 Ejecución

### Modo Desarrollo (Swagger)

Para probar los endpoints directamente desde la interfaz Swagger:

**Windows (cmd):**
```cmd
python main.py
```

**Windows (PowerShell) / Linux / Mac:**
```bash
python main.py
```

La API estará disponible en:
- **Servidor local**: `http://localhost:8000`
- **Documentación Swagger**: `http://localhost:8000/docs`
- **Documentación alternativa (ReDoc)**: `http://localhost:8000/redoc`

### Conexión con Frontend

#### Configuración CORS

El backend ya está configurado para aceptar peticiones desde el frontend. Por defecto permite conexiones desde:
- `http://localhost:3000` (React/Next.js por defecto)
- `http://localhost:5173` (Vite por defecto)
- `http://localhost:4200` (Angular por defecto)

Si tu frontend corre en otro puerto, edita `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:4200",
        "http://localhost:TU_PUERTO_AQUI"  # Añadir tu puerto
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📁 Estructura del Proyecto

```
datathon2025-backend-JAPD/
├── app/
│   ├── controlers/
│   │   ├── mlControler.py              # Endpoint de predicción
│   │   ├── openAiControler.py          # Endpoints con OpenAI
│   │   └── explainability/
│   │       └── lightGBM/
│   │           ├── limeControler.py    # Explicabilidad LIME
│   │           ├── shapControler.py    # Explicabilidad SHAP
│   │           └── pdpControler.py     # Partial Dependence Plots
│   └── types/
│       ├── mlTypes.py                  # Modelos Pydantic para ML
│       └── openAiTypes.py              # Modelos Pydantic para OpenAI
├── data/
│   ├── data.py                         # Almacenamiento en memoria
│   └── dataset.csv                     # Dataset de entrenamiento
├── models/
│   └── lgbm/
│       ├── lgbm_classifier.joblib      # Modelo entrenado
│       ├── X_train_sample.joblib       # Muestra de entrenamiento
│       └── X_test.joblib               # Datos de test
├── main.py                             # Punto de entrada de la aplicación
├── install_deps.sh                     # Script de instalación de dependencias
├── .env                                # Variables de entorno (crear)
└── README.md                           # Este archivo
```

## 🔌 Endpoints Disponibles

### Machine Learning (`/ml`)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/ml/predict` | Realizar predicción (devuelve 0 o 1) |
| POST | `/ml/explain_lime` | Explicación LIME local |
| POST | `/ml/explain_shap_local` | Explicación SHAP local |
| POST | `/ml/explain_shap_global` | Importancias SHAP globales |
| POST | `/ml/explain_pdp` | Partial Dependence Plot |

### OpenAI (`/ai`)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/ai/welcome` | Mensaje de bienvenida explicando LGBM |
| POST | `/ai/text/local` | Explicación textual de predicción local |
| POST | `/ai/text/global` | Explicación textual de patrones globales |
| POST | `/ai/pdp_sentence` | Resumen de gráfico PDP en una frase |
| POST | `/ai/answer` | Chatbot para responder dudas |

## 🔄 Flujo de Uso

### 1. Realizar Predicción
```
POST /ml/predict
```
Guarda automáticamente los datos de entrada y predicción en memoria.

### 2. Obtener Explicabilidad Local
```
POST /ml/explain_lime
O
POST /ml/explain_shap_local
```
Usa los datos guardados del paso 1. No requiere parámetros.

### 3. Obtener Explicabilidad Global
```
POST /ml/explain_shap_global
O
POST /ml/explain_pdp?feature_to_analyze=nombre_feature
```

### 4. Generar Textos Explicativos
```
POST /ai/welcome          # Bienvenida inicial
POST /ai/text/local       # Explicación de predicción individual
POST /ai/text/global      # Explicación de patrones del modelo
POST /ai/pdp_sentence     # Resumen de PDP
```

### 5. Chatbot Interactivo
```
POST /ai/answer
Body: { "question": "¿Por qué se predijo ganada?" }
```

## 🐛 Solución de Problemas

### Error: "Model not loaded"
- Verifica que existe `models/lgbm/lgbm_classifier.joblib`
- Asegúrate de haber entrenado y guardado el modelo correctamente

### Error: "No prediction data available"
- Debes llamar primero a `/ml/predict` antes de los endpoints de explicabilidad

### Error: "OpenAI API error"
- Verifica que tu `OPENAI_API_KEY` en `.env` es correcta
- Confirma que tienes créditos disponibles en tu cuenta de OpenAI

### CORS error desde el frontend
- Añade el puerto de tu frontend en la configuración CORS de `main.py`
- Verifica que el frontend está usando la URL correcta: `http://localhost:8000`

## 📊 Ejemplo Completo (usando curl)

```bash
# 1. Realizar predicción
curl -X POST "http://localhost:8000/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.5, "feature2": 0.8, ...}'

# 2. Obtener explicación LIME
curl -X POST "http://localhost:8000/ml/explain_lime"

# 3. Obtener texto explicativo
curl -X POST "http://localhost:8000/ai/text/local"

# 4. Hacer pregunta al chatbot
curl -X POST "http://localhost:8000/ai/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué variables son más importantes?"}'
```

## 👥 Equipo

Proyecto desarrollado por el equipo JAPD para el Datathon 2025.