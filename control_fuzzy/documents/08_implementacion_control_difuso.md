# Implementación del Control Difuso en Código [Fuzzy Control Implementation]

Este documento detalla cómo se traducen los conceptos del control difuso a la implementación en `turtle_fuzzy_obstacles_visual.py`.

## Estructura general [General Structure]

```python
class FuzzyObstacleVisualController(Node):
    def control_loop(self):
        # 1. Calcular distancia y ángulos
        # 2. Evaluar obstáculos y urgencia
        # 3. Aplicar reglas difusas
        # 4. Publicar velocidades
```

La clase hereda de `Node` y aprovecha timers para ejecutar `control_loop` cada 100 ms.

## Obtención de entradas [Input Acquisition]

- `distance`: norma euclidiana entre la pose actual `(x_t, y_t)` y la meta `(goal_x, goal_y)`, calculada como `sqrt(dx**2 + dy**2)`.
- `angle_to_goal`: dirección hacia el objetivo obtenida con `math.atan2(dy, dx)` y normalizada mediante `atan2(sin(..), cos(..))` para limitarla a `[-π, π]`.
- `avoidance_urgency`: valor entre 0 y 1 proporcionado por `get_obstacle_avoidance_direction`, que resume la peligrosidad del obstáculo más cercano.

## Evaluación de urgencia [Urgency Evaluation]

```python
avoidance_angle, avoidance_urgency = self.get_obstacle_avoidance_direction(...)
```

La función:

- **`find_closest_obstacle`**: recorre la lista `self.obstacles` e invoca `Obstacle.distance_to(x_t, y_t)` que aplica la fórmula `max(0, dist_center - radius)` para medir la distancia al borde.
- **Selección de dirección**: si la distancia es muy pequeña se genera un `escape_angle` (opuesto al obstáculo). En caso contrario se elige el perpendicular que menos desvíe del objetivo (`perp_angle_1` o `perp_angle_2`).
- **Cálculo de urgencia**: se normaliza con respecto a un umbral (`far_band`) y se refuerza si la trayectoria hacia el objetivo pasa cerca del obstáculo (`angle_diff_to_goal`), reflejando el análisis manual.

## Reglas difusas codificadas [Encoded Fuzzy Rules]

La función `apply_fuzzy_control` implementa las reglas:

- **Velocidad lineal**:
  - `avoidance_urgency > 0.7` → `linear_vel = 0.3` (equivale a conjunto *Urgencia Alta* + *Velocidad Muy Baja* del cálculo manual).
  - `avoidance_urgency > 0.4` → `linear_vel = 0.6` (*Urgencia Media* → *Velocidad Baja*).
  - Cuando la urgencia es baja, la velocidad depende de la distancia (*Lejos* → `1.5`, *Media* → `1.2`, *Cerca* → `0.5`).
- **Velocidad angular**:
  - `avoidance_urgency > 0.7` → `angular_vel = 2.5 * sign(angle_error)` (*Urgencia Alta* → *Giro Fuerte*).
  - `angle_abs > 1.5` → `angular_vel = 2.2 * sign(angle_error)` (*Error Grande* → *Giro Medio/Fuerte*).
  - Escenarios con errores menores usan valores de `1.8`, `1.0` o `0.5`, siguiendo la tabla del cálculo manual.
- **Escalado dinámico**:
  - `linear_vel *= (1 - avoidance_urgency * 0.8)` atenúa el avance conforme el peligro aumenta.

Aunque las reglas se implementan con condicionales `if/elif`, reflejan la lógica difusa descrita: cada condición corresponde a un conjunto de pertenencia predominante.

## Publicación de comandos [Command Publishing]

```python
cmd = Twist()
cmd.linear.x = linear_vel
cmd.angular.z = angular_vel
self.velocity_publisher.publish(cmd)
```

`geometry_msgs/Twist` es el mensaje estándar para robots diferenciales.

## Log y depuración [Logging]

Cada 20 iteraciones se imprime información diagnóstica:

- Posición actual, distancia al objetivo, distancia al obstáculo más cercano.
- Estado (`EVITANDO` o `LIBRE`) en función de `avoidance_urgency`.

Esto ayuda a validar que las reglas se activen en el rango esperado.

## Relación con el cálculo manual [Link to Manual Design]

- Los estados *Cerca/Media/Lejos* de la distancia corresponden a los cortes `1.0` y `3.0` empleados en la función `apply_fuzzy_control`.
- Las categorías *Pequeño/Medio/Grande* del error angular se reflejan en los umbrales `0.5`, `1.0` y `1.5`.
- Las salidas numéricas (0.15, 0.3, 0.6, 1.5 para lineal; 0.5, 1.0, 1.8, 2.2, 2.5 para angular) son exactamente las definidas en la tabla de parámetros manuales.
- `get_obstacle_avoidance_direction` transforma la distancia al obstáculo en urgencia para cumplir con la regla primaria: *si la urgencia es alta, reduzca la velocidad lineal y aumente el giro*.

## Posibles extensiones [Future Extensions]

- Sustituir los condicionales por funciones de pertenencia explícitas y agregación con `min/max`.
- Aplicar defuzzificación basada en centroide para obtener valores más suaves.
- Incorporar sensores adicionales (por ejemplo, `LaserScan`) si se migra a un TurtleBot real.

En conjunto, el archivo traduce los conceptos del control difuso a decisiones concretas de velocidad, integrando ROS 2, `turtlesim` y la lógica de evitación de obstáculos.***

