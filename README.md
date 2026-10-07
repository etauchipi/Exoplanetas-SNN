# Detección de Exoplanetas con SNN Bioinspirada en el Conectoma de la Mosca

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-1.0-orange)](https://snntorch.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Una exploración en la intersección entre neurociencia computacional y astrofísica:
> ¿Puede una red neuronal de impulsos inspirada en el sistema visual de la *Drosophila*
> detectar exoplanetas a partir de curvas de luz de Kepler?

---

## 📖 Resumen

Este proyecto explora si una **Red Neuronal de Impulsos (SNN)** inspirada en el sistema visual de la mosca de la fruta puede clasificar planetas confirmados usando únicamente curvas de luz de la misión Kepler.

Se construyeron **4 experimentos** con datasets crecientes en tamaño y realismo, culminando en un dataset de **2800 estrellas reales** (1400 CONFIRMED + 1400 Sin KOI) descargadas del archivo MAST de la NASA.

**Resultado principal:** la SNN **no logra distinguir** CONFIRMED de Sin KOI con solo la curva de luz (AUC test = 0.56, cerca del azar). Esto confirma lo que la literatura astronómica ya sabe: la detección de exoplanetas requiere features adicionales (período, duración, profundidad, centroide, espectroscopía).

El proyecto tiene valor como **resultado negativo documentado** y como **infraestructura reutilizable** para futuros proyectos de SNN en astrofísica.

---

## 🎯 Motivación

En 2024, Google y FlyWire publicaron el **primer conectoma completo del sistema nervioso de una mosca adulta**: 140,000 neuronas y 50 millones de sinapsis. Este logro abrió la puerta a aplicaciones que van más allá de la neurociencia: inspirar nuevas arquitecturas de redes neuronales.

La pregunta que guio este proyecto fue: **¿puede el sistema visual de la mosca — optimizado por evolución para detectar movimiento contra ruido — ayudar a encontrar planetas que bloquean una fracción mínima de la luz de su estrella?**

---

## 📊 Experimentos Realizados

| ID  | Dataset               | Clases                | N    | TARGET | AUC test | F1 test | Conclusión |
|-----|-----------------------|-----------------------|------|--------|----------|---------|------------|
| v1  | Kaggle Kepler         | CONFIRMED vs FP       | 5087 | 3197   | 0.65     | 0.57    | Baseline público |
| v3  | Kepler real (KOI)     | CONFIRMED vs FP       | 300  | 3197   | 0.65     | 0.68    | FP son binarias eclipsantes |
| v4b | Kepler real           | CONFIRMED vs Sin KOI  | 200  | 1001   | 0.59     | 0.64    | Downsampling a 1001 |
| **v5** | **Kepler real grande** | **CONFIRMED vs Sin KOI** | **2800** | **1001** | **0.56** | **0.67** | **Definitivo** |

Ver [`EXPERIMENTS.md`](EXPERIMENTS.md) para el detalle completo de cada uno.

---

## 🔬 Hallazgos Científicos

### 1. La curva de luz sola no basta
Con 2800 estrellas reales a 1001 puntos, la SNN colapsa a predecir "planeta" para todas las muestras (TN=0, FP=280 en test).

### 2. Los tránsitos de CONFIRMED son invisibles tras downsampling
Los tránsitos de planetas confirmados por Kepler tienen profundidades del **0.1–0.5 %**, mientras que las binarias eclipsantes tienen del **3–5 %**. Al comprimir 50,000 puntos a 1,001 (factor 50×), los tránsitos someros caben en menos de 1 punto y se diluyen.

### 3. Sesgo de selección en CONFIRMED
Las estrellas CONFIRMED son estadísticamente **más ruidosas** que las Sin KOI (std intra-curva 0.0016 vs 0.0013). Las estrellas activas tienen más probabilidad de tener planetas detectables.

### 4. La tarea requiere features adicionales
La literatura moderna (AstroNet, ExoMiner, ExoMiner++) usa:
- **Features de tránsito**: período, duración, profundidad, forma (U vs V)
- **Features estelares**: radio, masa, temperatura, metalicidad
- **Centroide del tránsito**: descarta binarias de fondo
- **Espectroscopía**: confirma la naturaleza del objeto

---

## 🛠️ Stack Técnico

| Categoría | Herramientas |
|-----------|-------------|
| **Lenguaje** | Python 3.12 |
| **Deep Learning** | PyTorch 2.x, snnTorch 1.0 |
| **Astronomía** | lightkurve, astroquery |
| **ML clásico** | scikit-learn |
| **Datos** | pandas, numpy, scipy |
| **Visualización** | matplotlib |
| **Entorno** | Jupyter Notebook en VS Code |

**Hardware utilizado:** Intel Core i7 (8ª gen), 20 GB RAM, sin GPU (entrenamiento en CPU).

---

## 📂 Estructura del Proyecto
exoplanetas-snn/
├── data/ # Datasets (no versionados, ver .gitignore)
├── models/ # Modelos entrenados (no versionados)
├── notebooks/ # Jupyter notebooks del proyecto
│ ├── 01_exploracion.ipynb
│ ├── 02_preprocesamiento.ipynb
│ ├── 03_snn_training.ipynb
│ ├── 04_kepler_real.ipynb
│ ├── 05_snn_kepler_v3.ipynb
│ ├── 06_snn_v4_sin_koi.ipynb
│ └── 07_snn_v5_1001.ipynb
├── src/ # Código modular
│ ├── init.py
│ ├── data_loader.py
│ ├── model.py
│ └── train.py
├── outputs/ # Resultados (gráficas, JSON)
│ ├── resumen_proyecto.png
│ ├── v5_resultados.json
│ └── proyecto_resultados.json
├── README.md # Este archivo
├── EXPERIMENTS.md # Detalle de cada experimento
├── LESSONS_LEARNED.md # Aprendizajes técnicos
├── requirements.txt # Dependencias
└── LICENSE # MIT


---

## 🚀 Cómo Reproducir

### 1. Clonar e instalar

```bash
git clone https://github.com/TU-USUARIO/exoplanetas-snn.git
cd exoplanetas-snn
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

2. Descargar datos
Ejecuta los notebooks en orden:

01_exploracion.ipynb → descarga dataset de Kaggle

04_kepler_real.ipynb → descarga curvas de luz de Kepler

3. Entrenar
03_snn_training.ipynb → baseline con dataset Kaggle

07_snn_v5_1001.ipynb → experimento definitivo con 2800 estrellas

📚 Documentación Adicional
EXPERIMENTS.md — Detalle de los 4 experimentos con métricas completas

LESSONS_LEARNED.md — Aprendizajes técnicos y trampas encontradas

🔮 Próximos Pasos
Este proyecto cierra con resultado negativo. La infraestructura queda disponible para:

Features engineered + XGBoost — enfoque de la industria

Reto 2: Fulguraciones solares — problema intrínsecamente temporal donde las SNN son competitivas

Reto 3: Evasión de basura espacial — control reactivo con SNN ligera

🙏 Agradecimientos
FlyWire Consortium por publicar el conectoma de la Drosophila

NASA Kepler Mission por los datos públicos

MAST Archive por el acceso a las curvas de luz

snnTorch por hacer las SNN accesibles en PyTorch

📜 Licencia
MIT — ver LICENSE.

📬 Contacto
Proyecto desarrollado como exploración personal en la intersección entre neurociencia computacional, machine learning y astrofísica.

"El éxito consiste en ir de fracaso en fracaso sin perder el entusiasmo."
— Winston Churchill
