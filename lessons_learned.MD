# Lessons Learned — Aprendizajes Técnicos

Este documento recopila los aprendizajes técnicos, trampas encontradas y decisiones
de diseño tomadas durante el proyecto. Sirve como referencia para futuros proyectos
que usen SNN en dominios científicos.

---

## 1. Sobre los Datos

### 1.1. Los datos grandes no van a Git
Los `.npy`, `.csv`, `.pt` pueden pesar cientos de MB o GB. Usar `.gitignore` desde
el primer commit. Solo subir código, notebooks, README y outputs pequeños (JSON, PNG).

### 1.2. Checkpointing incremental es obligatorio para descargas largas
Descargar 2,800 estrellas tarda 4.7 horas. Si algo falla a mitad, no quieres perder
todo. La solución:
- Guardar cada 50 estrellas en `.npy`.
- Al reiniciar, detectar el checkpoint y continuar desde ahí.
- Esto convierte una descarga frágil en una robusta.

**Código clave:**
```python
if os.path.exists(output_path):
    df_existing = pd.read_csv(output_path)
    processed_kepids = set(df_existing['KEPID'])
else:
    processed_kepids = set()

for kepid in kepids:
    if kepid in processed_kepids:
        continue  # Skip, ya procesado
    # ... procesar
    if len(data) % 50 == 0:
        save_checkpoint(data)  # Cada 50


1.3. Los .npy sobreviven reinicios de kernel, las variables no
Si reinicias el kernel de Jupyter, X_train, model, etc. desaparecen de memoria.
Los .npy en disco sobreviven. Siempre guarda los arrays procesados como .npy
al final de cada etapa.

1.4. Valida el dataset con visualizaciones antes de entrenar
En v4b y v5, el preprocessing "funcionaba" (no crasheaba) pero generaba curvas
centradas en 0.0, 1.0 o 0.8 según la estrella. Esto se descubrió solo al graficar
las curvas lado a lado. Sin la visualización, hubiéramos entrenado con datos
inconsistentes durante horas.

Regla: cada vez que hagas preprocessing, grafica 6 muestras (3 de cada clase)
antes de continuar.

2. Sobre el Preprocessing
2.1. El flatten() de lightkurve es inconsistente
lightkurve's flatten(window_length=401) puede centrar las curvas en 0 o en 1
según la estrella. Esto rompe el modelo porque aprende a distinguir por escala, no
por forma.

Solución: no usar flatten(). Hacer el aplanado manualmente con median_filter
y dividir (no restar) por la tendencia.

2.2. El sigma-clipping bilateral elimina los tránsitos
Un sigma_clip estándar (elimina |x - mean| > k·std) borra tanto outliers
positivos (rayos cósmicos) como negativos (tránsitos). Como los tránsitos son la
señal, hay que preservarlos.

Solución: sigma-clipping unilateral. Solo eliminar x > mean + k·std.

python
# MAL: elimina tránsitos
mask = np.abs(flux - mean) < 5 * std

# BIEN: preserva tránsitos
mask = flux < (mean + 5 * std)
2.3. El downsampling agresivo destruye señales débiles
Kepler-1: tránsitos del 1.4%, duran 3 horas → visibles tras comprimir 16×.

CONFIRMED típicas: tránsitos del 0.1–0.5%, duran 2–10 horas → invisibles tras
comprimir 16×.

Lección: si la señal es más corta que duración_original / factor_compresión,
se pierde. Para tránsitos de 3h sobre 50,000 puntos, comprimir a 3,197 (factor 16×)
da ~8 horas por punto: el tránsito cabe en menos de un punto.

2.4. Normalización robusta > normalización estándar
Con outliers masivos (valores de millones), la normalización estándar
(x - media) / std infla la desviación estándar y diluye la señal. La normalización
robusta (x - mediana) / MAD es resistente a outliers.

python
def robust_normalize(X):
    median = np.median(X, axis=1, keepdims=True)
    mad    = np.median(np.abs(X - median), axis=1, keepdims=True)
    mad    = np.maximum(mad, 0.05 * X.std(axis=1, keepdims=True))  # Piso
    return (X - median) / (1.4826 * mad + 1e-8)
El piso (max(mad, 0.05·std)) evita divisiones por cero en curvas muy planas.

3. Sobre el Modelado
3.1. El doble balanceo causa gradientes inestables
Usar al mismo tiempo WeightedRandomSampler y CrossEntropyLoss(weight=...) es
sobre-corregir: cada batch ya está balanceado, y encima el loss penaliza 134× los
errores de la clase minoritaria. Los gradientes explotan.

Regla: usa solo uno de los dos. Preferiblemente el sampler (más estable).

3.2. ReduceLROnPlateau con patience < 7 baja el LR demasiado pronto
En v3, con patience=3, el LR bajó de 1e-3 a 5e-7 en 20 épocas. El modelo dejó de
aprender en la época ~10.

Solución: patience=7 o más. O usar un scheduler cosmético más agresivo
(CosineAnnealing) en lugar de uno reactivo.

3.3. El augmentation con factor alto sobre pocos datos invierte el balance
Con 64 CONFIRMED y factor=5, se generan 320 muestras sintéticas. Ahora la clase
minoritaria se vuelve mayoritaria (320 vs 64). El modelo aprende un sesgo invertido.

Regla: no usar augmentation agresivo si el dataset tiene < 500 muestras por clase.
Es mejor conseguir más datos reales.

3.4. La SNN no es competitiva en clasificación de señales débiles en ruido
Las SNN son excelentes para:

Detección de eventos en series temporales (spikes, transitorios).

Control reactivo de baja latencia.

Procesamiento neuromórfico de bajo consumo.

Las SNN no son adecuadas para:

Clasificación de señales débiles sepultadas en ruido (como exoplanetas).

Tareas donde la señal es estática (imágenes, tablas).

Para exoplanetas: mejor usar CNNs profundas (AstroNet, ExoMiner) o Gradient
Boosting con features engineered.

3.5. Los papers modernos no usan la curva cruda
Para detección de exoplanetas, los mejores modelos usan:

Curva de luz phase-folded (plegada al período del candidato).

Features de tránsito (duración, profundidad, forma U vs V).

Features estelares (radio, masa, Teff).

Centroide del tránsito.

Sin estas features, incluso una CNN profunda tiene dificultades.

4. Sobre el Entrenamiento
4.1. El entrenamiento en CPU es viable para modelos pequeños
16K–47K parámetros.

30–50 épocas en 1–3 minutos (CPU Intel i7).

No se necesita GPU para prototipado.

Cuándo necesitas GPU: cuando el modelo pasa de ~1M parámetros, o cuando el
dataset tiene > 100K muestras.

4.2. Guarda el mejor modelo por F1, no por loss
El F1 es más informativo que el loss cuando hay desbalance. Guardar el mejor modelo
por F1 asegura que recuperas el punto de mejor generalización, no el de menor loss
de entrenamiento (que puede estar overfiteado).

python
if val_f1 > best_f1:
    best_f1 = val_f1
    torch.save(model.state_dict(), 'best.pt')
4.3. El threshold óptimo no es 0.5
Con clases desbalanceadas, el umbral 0.5 de argmax no es óptimo. Barrer umbrales
de 0.05 a 0.95 y elegir el que maximiza F1 en validación.

python
thresholds = np.arange(0.05, 0.95, 0.02)
f1s = [f1_score(y_true, (y_probs > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1s)]
Señal de alarma: si el threshold óptimo cae en 0.05 (el mínimo), el modelo no
separa bien las clases.

5. Sobre Windows + Jupyter
5.1. No uses emojis en print() de Python
Windows usa cp1252 por defecto, que no puede representar emojis. print("✅ OK")
lanza UnicodeEncodeError.

Solución: usar [OK], [ERROR], [WARN] en lugar de emojis. O forzar UTF-8:

python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
5.2. Selecciona el kernel correcto en VS Code
Si los imports fallan (NameError: pandas not defined), probablemente el kernel
de Jupyter no es el del .venv. Verifica:

python
import sys
print(sys.executable)  # Debe apuntar a .venv\Scripts\python.exe
Registrar un kernel:

bash
python -m ipykernel install --user --name=mi-venv --display-name="Python (mi-venv)"
5.3. El nombre del venv no importa
.venv es la convención oficial y VS Code lo detecta automáticamente. No lo
renombres después de crearlo (rompe las rutas absolutas internas).

6. Sobre la Ciencia
6.1. Los resultados negativos son valiosos
El 80% de los experimentos en investigación fallan. Documentarlos evita que otros
repitan el camino. Este proyecto "falló" en su objetivo pero dejó:

Infraestructura reutilizable.

Diagnóstico claro de por qué no funciona.

Sugerencias concretas para futuros intentos.

6.2. La visualización es tu mejor herramienta de diagnóstico
Antes de entrenar, grafica los datos. Después de entrenar, grafica las curvas
de aprendizaje. Durante la evaluación, grafica la matriz de confusión.

El 90% de los bugs en ML se detectan visualmente: curvas raras, distribuciones
inconsistentes, predicciones colapsadas.

6.3. Compara contra un baseline simple
Entrenar un MLP sin convoluciones ni SNN como baseline te dice si la arquitectura
compleja realmente aporta valor. Si el baseline gana, la complejidad no está
justificada.

6.4. El dataset perfecto no existe
Cada dataset tiene sus problemas:

Kaggle: solo 37 positivos, tránsitos pre-procesados.

Kepler real: descarga lenta, tránsitos someros, ruido instrumental.

Sin KOI: no significa "sin señal", significa "sin candidato detectable".

Aprender a trabajar con datos imperfectos es la mitad del trabajo en ML aplicado.

7. Checklist para Futuros Proyectos
□ Crear .gitignore desde el primer commit
□ Implementar checkpointing incremental en descargas largas
□ Visualizar los datos antes de entrenar
□ Usar normalización robusta (mediana + MAD)
□ Preservar señales débiles en sigma-clipping (unilateral)
□ Elegir un solo mecanismo de balanceo (sampler O class weights)
□ Guardar el mejor modelo por F1, no por loss
□ Calibrar el threshold de decisión en validación
□ Comparar contra un baseline simple
□ Documentar resultados negativos
□ No usar emojis en print() en Windows
□ Guardar .npy después de cada etapa de preprocessing
8. Reflexión Final
Este proyecto me enseñó que la parte más difícil del machine learning aplicado no
es el modelo, sino los datos. Un modelo simple con buenos datos supera a un modelo
sofisticado con datos malos. Las SNN son una herramienta elegante, pero no son una
bala de plata: tienen dominios donde brillan (temporal, eventos, control) y dominios
donde no (señales débiles estáticas).

La próxima vez que enfrente un problema de clasificación, empezaré por:

Graficar los datos.

Extraer features engineered.

Entrenar un baseline simple (XGBoost).

Solo si el baseline falla, considerar arquitecturas complejas.

"Todos los modelos están equivocados, pero algunos son útiles."
— George Box

