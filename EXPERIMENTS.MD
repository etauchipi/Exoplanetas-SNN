
---

## 📄 Archivo 2: `EXPERIMENTS.md`

```markdown
# Experimentos — Detalle Completo

Este documento registra los 4 experimentos realizados durante el proyecto, con sus
métricas, configuraciones y conclusiones. Su objetivo es ser una bitácora reproducible
y honesta de la investigación.

---

## Índice

- [Experimento v1 — Baseline con Kaggle](#experimento-v1)
- [Experimento v3 — Kepler real (CONFIRMED vs FP)](#experimento-v3)
- [Experimento v4b — Kepler real (CONFIRMED vs Sin KOI, 200)](#experimento-v4b)
- [Experimento v5 — Kepler real (CONFIRMED vs Sin KOI, 2800)](#experimento-v5)
- [Tabla Comparativa](#tabla-comparativa)

---

## Experimento v1

**Dataset:** Kaggle Labelled Time Series Data (Kepler)
**Clases:** CONFIRMED (1) vs FALSE POSITIVE (2) — remapeados a 1 y 0
**Total:** 5,087 train + 570 test
**Target Length:** 3,197 puntos temporales
**Split:** Estratificado 80/20, luego 80/20 para train/val interno

### Arquitectura
- `FlyInspiredSNN` original (Conv1D 16→32 canales)
- Parámetros: **189,378**
- `num_steps=25`, `beta=0.9`, `threshold=1.0`
- Dropout=0.4

### Entrenamiento
- Épocas: 50
- LR inicial: 1e-3 con `ReduceLROnPlateau(patience=3)`
- Weight decay: 1e-4
- Batch size: 32
- `WeightedRandomSampler` en train

### Resultados
| Métrica | Valor |
|---------|-------|
| F1 val (mejor) | 0.600 |
| AUC val (mejor) | 0.852 |
| F1 test | 0.571 |
| AUC test | 0.65 |

### Conclusión
Baseline razonable, pero el modelo **overfiteó severamente**: train loss bajó a 0.0002
mientras val loss subía. La clase negativa (FP) contiene **binarias eclipsantes con
tránsitos profundos (3-5%)**, más fáciles de detectar que los tránsitos de planetas
confirmados (0.5-2%).

**Aprendizaje:** El problema real no es "hay tránsito vs no hay tránsito", sino
"tránsito es planeta vs tránsito es falso positivo". Esto es *vetting*, un problema
mucho más difícil.

---

## Experimento v3

**Dataset:** Kepler real (NASA Exoplanet Archive KOI)
**Clases:** CONFIRMED vs FALSE POSITIVE
**Total:** 300 estrellas (150 + 150)
**Target Length:** 3,197 puntos
**Split:** 80/20 estratificado (240 train / 60 test), luego 80/20 interno (192/48)

### Pipeline de Datos
- Descarga con `lightkurve` (todos los trimestres, `stitch()`)
- Preprocessing propio: `remove_nans` → `normalize` → `remove_outliers` → `flatten(window=401)`
- Alineación por interpolación a 3,197 puntos

### Arquitectura
- `FlyInspiredSNN` reducida (Conv1D 8→16 canales)
- Parámetros: **47,458**
- Dropout=0.4

### Entrenamiento
- Épocas: 50
- LR inicial: 1e-3 con `ReduceLROnPlateau(patience=7)`
- Batch size: 32
- `WeightedRandomSampler` en train
- Augmentation: factor=5 con ruido gaussiano (σ=0.05)

### Resultados
| Métrica | Valor |
|---------|-------|
| F1 val (mejor) | 0.667 |
| AUC val (mejor) | 0.852 |
| F1 test | 0.684 |
| AUC test | 0.654 |
| Threshold óptimo | 0.05 |
| TN / FP / FN / TP | 10 / 20 / 4 / 26 |

### Conclusión
El modelo **apenas distinguía** las clases. El threshold óptimo cayó al mínimo (0.05),
lo que indica que el modelo no separa bien las distribuciones. La matriz de confusión
mostró **20 falsos positivos** de 30 negativos: el modelo sobre-predice "planeta".

**Aprendizaje:** Confirmado el diagnóstico de v1. La clase FP no es un buen negativo.

---

## Experimento v4b

**Dataset:** Kepler real (CONFIRMED + Sin KOI)
**Clases:** CONFIRMED vs Sin KOI (estrellas sin ningún candidato)
**Total:** 200 estrellas (100 + 100)
**Target Length:** 1,001 puntos (downsampling menos agresivo)
**Split:** 80/20 estratificado (160 train / 40 test), luego 80/20 interno (128/32)

### Pipeline de Datos
- Descarga con `lightkurve`
- **Preprocessing v2**: preserva tránsitos
  - Elimina NaNs
  - Normaliza por mediana
  - **Sigma-clipping unilateral**: solo elimina outliers positivos (no los tránsitos)
  - **Sin flatten**: la señal del tránsito no se atenúa
- Alineación a 1,001 puntos

### Arquitectura
- `FlyInspiredSNN(input_len=1001)`
- Parámetros: **16,226**
- `flat_dim=448`

### Entrenamiento
- Épocas: 30
- LR: 5e-4 fijo
- Batch size: 32, `shuffle=True` (sin sampler, clases ya balanceadas)
- **Sin augmentation**

### Resultados
| Métrica | Valor |
|---------|-------|
| F1 val (mejor) | 0.750 |
| AUC val (mejor) | 0.773 |
| F1 test | 0.642 |
| AUC test | 0.590 |
| Threshold óptimo | 0.49 |
| TN / FP / FN / TP | 4 / 16 / 3 / 17 |

### Conclusión
El cambio de clase negativa (Sin KOI) y la reducción de TARGET_LENGTH **mejoraron
marginalmente** el AUC val (0.77 vs 0.66). Pero el test siguió siendo pobre (0.59)
porque **200 muestras son demasiado pocas** para un test confiable (solo 16 positivos
en val).

**Aprendizaje:** Con 200 muestras el AUC tiene un error de ±0.15. Se necesitan más datos.

---

## Experimento v5

**Dataset:** Kepler real grande
**Clases:** CONFIRMED vs Sin KOI
**Total:** 2,800 estrellas (1,400 + 1,400)
**Target Length:** 1,001 puntos
**Split:** 80/20 estratificado (2,240 train / 560 test), luego 80/20 interno (1,792/448)

### Descarga
- **4.7 horas** de descarga incremental con checkpoints cada 50 estrellas
- Pipeline: `search_lightcurve` → `download_all` → `stitch`
- Preprocessing v2 + alineación a 1,001 puntos

### Arquitectura
- `FlyInspiredSNN(input_len=1001)`
- Parámetros: **16,226**

### Entrenamiento
- Épocas: 30
- LR: 5e-4 fijo
- Batch size: 32, `shuffle=True`
- Sin augmentation (clases ya balanceadas)

### Resultados
| Métrica | Valor |
|---------|-------|
| F1 val (mejor) | 0.651 |
| AUC val (mejor) | 0.662 |
| F1 test | 0.667 |
| **AUC test** | **0.559** |
| Threshold óptimo | 0.05 |
| TN / FP / FN / TP | **0 / 280 / 0 / 280** |

### Conclusión
**El modelo colapsó a predecir "planeta" para todas las muestras de test.** El
umbral óptimo cayó al mínimo (0.05), la matriz de confusión muestra 0 verdaderos
negativos, y el AUC de 0.559 está cerca del azar (0.5).

Curiosamente, el val loss **subió** de 0.69 a 0.81 durante el entrenamiento, mientras
el train loss bajaba de 0.69 a 0.36. Overfitting severo con solo 16K parámetros.

**Aprendizaje definitivo:** La tarea no es aprendible con solo la curva de luz
downsampleada. Se necesitan features adicionales.

---

## Tabla Comparativa

| Experimento | N total | Target | Params | F1 val | AUC val | F1 test | AUC test |
|-------------|---------|--------|--------|--------|---------|---------|----------|
| v1 | 5,087 | 3,197 | 189,378 | 0.600 | 0.852 | 0.571 | 0.650 |
| v3 | 300 | 3,197 | 47,458 | 0.667 | 0.852 | 0.684 | 0.654 |
| v4b | 200 | 1,001 | 16,226 | 0.750 | 0.773 | 0.642 | 0.590 |
| **v5** | **2,800** | **1,001** | **16,226** | **0.651** | **0.662** | **0.667** | **0.559** |

### Observaciones

1. **AUC val > AUC test siempre**: la validación era demasiado pequeña para estimar bien.
2. **Más datos no mejoró el AUC**: v5 con 2,800 estrellas tiene AUC test 0.56 vs v3 con 300 y AUC 0.65.
3. **El mejor AUC test fue v3 (0.65)**: aunque con clase negativa "fácil" (FP).
4. **La arquitectura reducida (16K params) es estable**: no explota con pocos datos, pero tampoco aprende mucho.

### Conclusión Global

Con la curva de luz sola (a cualquier resolución probada), la SNN **no puede** distinguir
planetas confirmados de estrellas sin señal. Esto es consistente con la literatura:
la detección de exoplanetas requiere features astronómicas adicionales (período,
duración, profundidad, centroide, espectroscopía).