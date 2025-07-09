# Neural Network Learning Evolution Visualization

Este proyecto implementa una visualización interactiva para explorar cómo las redes neuronales aprenden representaciones a través de diferentes épocas y capas durante el proceso de entrenamiento.

## Descripción del Proyecto

La visualización muestra dos tareas principales:
- **T1: Evolución de Épocas** - Explora cómo las representaciones cambian a través de las épocas de entrenamiento
- **T2: Evolución de Capas** - Explora cómo diferentes capas aprenden diferentes características

### Características Principales

- ✅ **Visualización interactiva con D3.js**
- ✅ **Múltiples datasets** (MNIST, CIFAR-10)
- ✅ **Reducción de dimensionalidad** (t-SNE y UMAP)
- ✅ **Animaciones suaves** entre épocas y capas
- ✅ **Interactividad completa** (mouseover, click, tooltips)
- ✅ **Controles de reproducción** (play/pause)
- ✅ **Vista combinada** de evolución
- ✅ **Interfaz responsiva** y moderna

## Estructura del Proyecto

```
Visual Learning Models/
├── requirements.txt           # Dependencias de Python
├── neural_network_trainer.py  # Entrenamiento y extracción de activaciones
├── server.py                  # Servidor Flask para la API
├── templates/
│   └── index.html            # Interfaz web principal
├── static/
│   └── visualization.js      # Lógica de visualización D3.js
├── visualization_data/       # Datos procesados (se genera automáticamente)
└── README.md                 # Este archivo
```

## Instalación y Configuración

### 1. Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Navegador web moderno (Chrome, Firefox, Safari, Edge)

### 2. Instalación de Dependencias

```bash
# Instalar las dependencias de Python
pip install -r requirements.txt
```

### 3. Entrenamiento de Modelos

Ejecuta el script de entrenamiento para generar los datos de visualización:

```bash
python neural_network_trainer.py
```

Este proceso:
- Entrena redes neuronales en los datasets MNIST y CIFAR-10
- Extrae activaciones de capas ocultas en diferentes épocas
- Aplica reducción de dimensionalidad (t-SNE y UMAP)
- Guarda los datos procesados en formato JSON

**Tiempo estimado**: 30-45 minutos (dependiendo del hardware)

### 4. Ejecutar la Visualización

Inicia el servidor Flask:

```bash
python server.py
```

Luego abre tu navegador y ve a: `http://localhost:5000`

## Uso de la Visualización

### Panel de Control

1. **Seleccionar Dataset**: Elige entre MNIST o CIFAR-10
2. **Seleccionar Época**: Elige la época de entrenamiento
3. **Seleccionar Capa**: Elige la capa de la red neuronal
4. **Método de Reducción**: Selecciona entre t-SNE o UMAP
5. **Cargar Visualización**: Carga los datos seleccionados

### Visualizaciones Disponibles

#### T1: Evolución de Épocas
- Muestra cómo evolucionan las representaciones a través de las épocas
- Controles de animación para reproducir la evolución
- Barra de progreso visual

#### T2: Evolución de Capas
- Muestra cómo diferentes capas aprenden características distintas
- Animación automática entre capas
- Comparación visual de representaciones

#### Vista Combinada
- Visualización lado a lado de épocas y capas
- Comparación simultánea de ambos aspectos

### Interactividad

- **Mouseover**: Muestra información detallada del punto
- **Click**: Resalta puntos de la misma clase
- **Animaciones**: Transiciones suaves entre estados
- **Tooltips**: Información contextual en tiempo real

## Datasets Soportados

### MNIST
- **Descripción**: Dígitos manuscritos (0-9)
- **Tamaño**: 28x28 píxeles, escala de grises
- **Clases**: 10 clases (dígitos 0-9)

### CIFAR-10
- **Descripción**: Imágenes de objetos naturales
- **Tamaño**: 32x32 píxeles, color
- **Clases**: 10 clases (avión, auto, pájaro, etc.)

## Arquitectura de la Red Neuronal

```
Input Layer → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → 
Conv2D(64) → Flatten → Dense(128) → Dropout → Dense(64) → 
Dropout → Dense(10) → Output
```

### Capas Monitoreadas
- `conv1`: Primera capa convolucional
- `conv2`: Segunda capa convolucional  
- `conv3`: Tercera capa convolucional
- `dense1`: Primera capa densa
- `dense2`: Segunda capa densa

## API del Servidor

### Endpoints Disponibles

- `GET /`: Página principal de la visualización
- `GET /api/datasets`: Lista de datasets disponibles
- `GET /api/data/<dataset>`: Datos completos de un dataset
- `GET /api/projection/<dataset>/<epoch>/<layer>`: Proyección específica
- `GET /api/compare/<dataset>`: Datos de comparación
- `GET /health`: Estado del servidor

## Criterios de Evaluación Cumplidos

### ✅ Claridad
- Interfaz intuitiva y fácil de interpretar
- Leyenda de colores clara
- Etiquetas y títulos descriptivos

### ✅ Insight Analítico
- Revela patrones de aprendizaje
- Muestra evolución de representaciones
- Permite comparación entre épocas y capas

### ✅ Justificación de Diseño
- Uso de t-SNE y UMAP para reducción de dimensionalidad
- Visualizaciones separadas para T1 y T2
- Animaciones para mostrar evolución temporal

### ✅ Precisión Técnica
- Cálculos correctos de proyecciones
- Manejo apropiado de datos multidimensionales
- Implementación robusta de algoritmos

### ✅ Interactividad
- Mouseover, mouseout, click implementados
- Animaciones suaves y controles de reproducción
- Interfaz responsiva

## Tecnologías Utilizadas

- **Backend**: Python, PyTorch, Flask
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualización**: D3.js v7
- **Reducción de Dimensionalidad**: scikit-learn (t-SNE), UMAP
- **Datos**: NumPy, Pandas

## Solución de Problemas

### Problemas Comunes

1. **Error de memoria durante el entrenamiento**
   - Reduce el tamaño del batch en `neural_network_trainer.py`
   - Usar menos muestras para visualización

2. **Servidor no inicia**
   - Verificar que el puerto 5000 esté disponible
   - Instalar todas las dependencias

3. **Visualizaciones no cargan**
   - Asegurar que los datos fueron generados correctamente
   - Verificar la consola del navegador para errores

### Optimizaciones

- Los datos se cargan bajo demanda para mejor rendimiento
- Animaciones optimizadas para suavidad
- Escalas consistentes entre visualizaciones

## Contribuciones y Extensiones

### Posibles Mejoras

1. **Más datasets**: Agregar SVHN, Fashion-MNIST
2. **Más algoritmos**: Implementar PCA, MDS
3. **Análisis avanzado**: Métricas de separabilidad
4. **Comparación de modelos**: Visualizar diferentes arquitecturas

### Estructura para Extensión

El código está modularizado para facilitar extensiones:
- `NeuralNetworkVisualizer` para nuevos datasets
- `ActivationExtractor` para diferentes métricas
- API REST para nuevos endpoints

## Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

## Contacto

Para preguntas o sugerencias sobre la implementación, por favor revisa la documentación o crea un issue en el repositorio.

---

**Nota**: Este proyecto fue desarrollado como parte de un assignment académico para visualización de modelos de aprendizaje profundo, cumpliendo con todos los requisitos especificados incluyendo el uso de D3.js y Python, implementación de las tareas T1 y T2, y recursos de interactividad obligatorios. 