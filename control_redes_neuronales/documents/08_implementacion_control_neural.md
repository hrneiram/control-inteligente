# Implementación del Control Neuronal [Neural Control Implementation]

Este documento detalla cómo se implementa el control basado en redes neuronales en `turtle_nn_controller.py`.

## Estructura general [General Structure]

```python
class TurtleNNController(Node):
    def control_loop(self):
        # 1. Calcular entradas (distancias, ángulos)
        # 2. Normalizar entradas
        # 3. Pasar por la red neuronal
        # 4. Desnormalizar salidas
        # 5. Publicar comandos de velocidad
```

## Obtención de entradas [Input Acquisition]

1. **Distancia al objetivo**: `sqrt((goal_x - x)² + (goal_y - y)²)`
2. **Error angular**: `atan2(dy, dx) - theta` (normalizado a [-π, π])
3. **Distancia al obstáculo**: Distancia al borde del obstáculo más cercano
4. **Ángulo al obstáculo**: Ángulo relativo desde la orientación actual
5. **Velocidades actuales**: Valores del ciclo anterior

## Normalización [Normalization]

Todas las entradas se normalizan para mejorar el entrenamiento:

```python
inputs = [
    dist_to_goal / 15.0,           # [0, 1]
    angle_error / π,               # [-1, 1]
    dist_to_obstacle / 15.0,       # [0, 1]
    angle_to_obstacle / π,         # [-1, 1]
    linear_vel / 2.0,              # [0, 1]
    angular_vel / 3.0              # [-1, 1]
]
```

## Inferencia neuronal [Neural Inference]

```python
with torch.no_grad():
    input_tensor = torch.FloatTensor(inputs).unsqueeze(0)
    output = self.model(input_tensor)
    linear_vel = output[0, 0].item()  # Ya desnormalizado por la red
    angular_vel = output[0, 1].item()
```

La red ya produce salidas en los rangos correctos gracias a las funciones de activación finales (sigmoid y tanh).

## Publicación de comandos [Command Publishing]

```python
cmd = Twist()
cmd.linear.x = linear_vel   # [0, 2.0] m/s
cmd.angular.z = angular_vel # [-3.0, 3.0] rad/s
self.velocity_publisher.publish(cmd)
```

## Manejo de objetivos [Goal Handling]

Cuando la tortuga alcanza un objetivo (distancia < 0.5):
1. Se genera un nuevo objetivo aleatorio.
2. Se verifica que esté lejos de obstáculos.
3. Se redibuja el objetivo en pantalla.

## Logging y depuración [Logging]

Cada 20 iteraciones se imprime:
- Estado (EVITANDO o LIBRE)
- Posición actual
- Distancia al objetivo
- Distancia al obstáculo más cercano
- Velocidades actuales

## Diferencias con control difuso [Differences from Fuzzy Control]

- **Lógica aprendida**: La red aprende de datos en lugar de reglas explícitas.
- **Generalización**: Puede manejar situaciones no contempladas en las reglas.
- **Adaptabilidad**: Puede mejorar con más entrenamiento.
- **Complejidad**: Requiere datos de entrenamiento o generación sintética.

