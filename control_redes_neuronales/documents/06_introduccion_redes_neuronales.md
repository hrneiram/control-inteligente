# Introducción a Redes Neuronales [Neural Networks Introduction]

Las redes neuronales artificiales son modelos computacionales inspirados en el cerebro biológico. Consisten en capas de neuronas interconectadas que procesan información.

## Conceptos clave [Key Concepts]

- **Neurona [Neuron]**: Unidad básica que recibe entradas, las pondera, y produce una salida mediante una función de activación.
- **Capa [Layer]**: Conjunto de neuronas que procesan información en paralelo.
- **Peso [Weight]**: Parámetro que determina la importancia de una conexión entre neuronas.
- **Sesgo [Bias]**: Valor constante añadido a la suma ponderada antes de la activación.
- **Función de activación [Activation Function]**: Función no lineal que introduce complejidad al modelo (ReLU, sigmoid, tanh).

## Arquitectura de la red [Network Architecture]

La red neuronal del proyecto tiene:

```
Entrada (6 neuronas)
    ↓
Capa oculta 1 (64 neuronas) + ReLU + Dropout
    ↓
Capa oculta 2 (32 neuronas) + ReLU
    ↓
Salida (2 neuronas)
    ├── Velocidad lineal (sigmoid → [0, 2.0])
    └── Velocidad angular (tanh → [-3.0, 3.0])
```

## Entradas [Inputs]

1. Distancia al objetivo (normalizada)
2. Error angular hacia el objetivo (normalizado)
3. Distancia al obstáculo más cercano (normalizada)
4. Ángulo relativo al obstáculo (normalizado)
5. Velocidad lineal actual (normalizada)
6. Velocidad angular actual (normalizada)

## Salidas [Outputs]

1. **Velocidad lineal**: Rango [0, 2.0] m/s
2. **Velocidad angular**: Rango [-3.0, 3.0] rad/s

## Entrenamiento [Training]

El modelo se entrena con:
- **Datos sintéticos**: Generados usando reglas heurísticas.
- **Función de pérdida**: Error cuadrático medio (MSE).
- **Optimizador**: Adam (Adaptive Moment Estimation).
- **Épocas**: 100 iteraciones sobre el dataset.

## Ventajas sobre control difuso [Advantages over Fuzzy Control]

- **Aprendizaje automático**: Puede mejorar con más datos.
- **Generalización**: Aprende patrones complejos de los datos.
- **Adaptabilidad**: Puede ajustarse a diferentes escenarios con re-entrenamiento.

## Limitaciones [Limitations]

- Requiere datos de entrenamiento o reglas para generarlos.
- Puede necesitar ajuste fino para diferentes entornos.
- El modelo inicial usa pesos aleatorios si no hay modelo pre-entrenado.

