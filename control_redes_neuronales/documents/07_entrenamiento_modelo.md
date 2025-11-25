# Entrenamiento del Modelo Neuronal [Neural Model Training]

Este documento explica cómo entrenar la red neuronal para el control de TurtleSim.

## Generación de datos [Data Generation]

El script `train_nn_model.py` genera datos sintéticos basados en reglas heurísticas:

### Reglas implementadas [Implemented Rules]

1. **Obstáculo muy cercano (< 1.0 unidades)**:
   - Velocidad lineal: 0.2 m/s (muy lenta)
   - Velocidad angular: 2.5 rad/s (girar lejos del obstáculo)

2. **Obstáculo cercano (1.0 - 2.5 unidades)**:
   - Velocidad lineal: 0.6 m/s (lenta)
   - Velocidad angular: 1.8 rad/s (evitar) o 1.2 rad/s (navegar)

3. **Sin obstáculos cercanos (> 2.5 unidades)**:
   - Velocidad lineal: Basada en distancia al objetivo
     - Cerca (< 1.0): 0.5 m/s
     - Media (1.0 - 3.0): 1.2 m/s
     - Lejos (> 3.0): 1.5 m/s
   - Velocidad angular: Basada en error angular
     - Grande (> 1.5 rad): 2.0 rad/s
     - Medio (1.0 - 1.5 rad): 1.5 rad/s
     - Pequeño (0.5 - 1.0 rad): 1.0 rad/s
     - Muy pequeño (< 0.5 rad): 0.5 rad/s

## Proceso de entrenamiento [Training Process]

1. **Generación**: Crear 5000 ejemplos de entrenamiento.
2. **Normalización**: Escalar entradas y salidas a rangos [0, 1] o [-1, 1].
3. **División en batches**: Procesar datos en grupos de 32 ejemplos.
4. **Forward pass**: Pasar datos por la red y obtener predicciones.
5. **Cálculo de pérdida**: Comparar predicciones con objetivos usando MSE.
6. **Backward pass**: Calcular gradientes y actualizar pesos.
7. **Iteración**: Repetir por 100 épocas.

## Ejecución [Execution]

```bash
cd ros2_ws/ros2_ws/src/turtle_nn_control/turtle_nn_control
python3 train_nn_model.py
```

El modelo entrenado se guarda como `turtle_nn_model.pth` en el mismo directorio.

## Parámetros ajustables [Adjustable Parameters]

- `num_samples`: Número de ejemplos de entrenamiento (default: 5000)
- `epochs`: Número de épocas (default: 100)
- `batch_size`: Tamaño del batch (default: 32)
- `learning_rate`: Tasa de aprendizaje (default: 0.001)

## Mejoras futuras [Future Improvements]

- **Datos reales**: Recopilar datos de ejecuciones exitosas.
- **Aprendizaje por refuerzo**: Entrenar con recompensas basadas en rendimiento.
- **Transfer learning**: Usar modelos pre-entrenados y ajustarlos.
- **Validación cruzada**: Evaluar el modelo en datos no vistos.

