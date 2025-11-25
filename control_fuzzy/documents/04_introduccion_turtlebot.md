# Introducción a TurtleBot / TurtleSim [TurtleBot Introduction]

Aunque el proyecto se ejecuta sobre `turtlesim`, los conceptos se inspiran en la familia de robots móviles TurtleBot. Este documento explica la relación entre ambos y cómo se emplea en `ros2_turtle_project`.

## ¿Qué es TurtleBot? [What is TurtleBot]

TurtleBot es una plataforma robótica educativa y de investigación basada en ROS. Proporciona sensores, actuadores y ejemplos de navegación. Para fines de prototipado rápido, `turtlesim` actúa como un sustituto virtual de un TurtleBot real.

## Rol de TurtleSim [TurtleSim Role]

- **Simulación 2D [2D Simulation]**: Ofrece un entorno plano donde una tortuga virtual se desplaza mediante comandos de velocidad.
- **Interfaces compatibles [Compatible Interfaces]**: Usa los mismos mensajes que un robot diferencial (`geometry_msgs/Twist`), lo que facilita migrar a hardware real.
- **Herramienta didáctica [Educational Tool]**: Permite probar algoritmos de control sin riesgos físicos.

## Uso en el proyecto [Usage in Project]

1. Iniciar el simulador:
   ```bash
   ros2 run turtlesim turtlesim_node
   ```
2. Ejecutar el controlador difuso:
   ```bash
   ros2 run turtle_fuzzy_control fuzzy_visual
   ```
3. El controlador envía velocidades a `/turtle1/cmd_vel`, exactamente igual a como lo haría con un TurtleBot real.
4. La pose de la tortuga se obtiene de `/turtle1/pose`, que contiene posición `(x, y)` y orientación `theta`.

## Transición a hardware [Hardware Transition]

Al usar interfaces estándar de ROS 2, el código del controlador puede migrarse a un TurtleBot físico reemplazando las suscripciones/publicaciones por los tópicos equivalentes de la plataforma real:

- `/cmd_vel` para velocidades.
- `/odom` o `/tf` para pose.
- Sensores de proximidad para estimar distancias a obstáculos.

En conclusión, `turtlesim` brinda un entorno sencillo para validar la lógica de navegación y evitar obstáculos antes de implementarla en un TurtleBot real o en simuladores más complejos.***

