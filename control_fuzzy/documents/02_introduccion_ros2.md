# Introducción a ROS 2 [ROS 2 Introduction]

ROS 2 (Robot Operating System 2) es un marco de trabajo para desarrollar aplicaciones robóticas distribuidas. Utiliza una arquitectura basada en nodos, tópicos, servicios y acciones. Este documento resume sus conceptos clave y el uso específico dentro de `ros2_turtle_project`.

## Conceptos fundamentales [Core Concepts]

- **Nodo [Node]**: Proceso que ejecuta una parte del sistema; en este proyecto destaca `turtle_fuzzy_obstacle_visual`.
- **Tópico [Topic]**: Canal de publicación/suscripción. Ejemplos:
  - `/turtle1/cmd_vel` (velocidades [`geometry_msgs/Twist`])
  - `/turtle1/pose` (pose de la tortuga [`turtlesim/Pose`])
- **Servicio [Service]**: Comunicación request/response usada para operaciones puntuales. Se emplean servicios como `/spawn`, `/kill` y `/teleport_absolute`.
- **Timer [Timer]** y **Callback [Callback]**: Mecanismos para ejecutar lógica periódica o reaccionar a eventos.

## Herramientas clave [CLI Tools]

- `ros2 run`: ejecuta nodos instalados mediante paquetes.
- `ros2 topic echo`: inspecciona mensajes que fluyen por un tópico.
- `ros2 node list`: verifica qué nodos están activos en la sesión.

## Flujo dentro del proyecto [Project Flow]

1. **Simulador**: `ros2 run turtlesim turtlesim_node` inaugura el nodo del simulador.
2. **Controlador difuso**: `ros2 run turtle_fuzzy_control fuzzy_visual` lanza el nodo que gestiona la navegación y el dibujo de obstáculos.
3. **Comunicación**:
   - El controlador se suscribe a `/turtle1/pose` para obtener retroalimentación.
   - Publica comandos en `/turtle1/cmd_vel`.
   - Usa servicios para crear tortugas temporales encargadas de dibujar círculos.
4. **Timers**:
   - Un `timer` inicial dibuja obstáculos y objetivo.
   - Otro `timer` (100 ms) ejecuta la lógica de control para actualizar velocidades.

## Integración con Docker [Docker Integration]

Dentro del contenedor, los comandos de ROS 2 funcionen igual que en una instalación nativa. Los scripts de arranque (`run_docker.sh`) ya incluyen la configuración de variables de entorno, por lo que el usuario solo debe ejecutar las instrucciones indicadas.

En síntesis, ROS 2 proporciona la infraestructura de comunicaciones necesaria para que el controlador difuso, el simulador `turtlesim` y los servicios auxiliares colaboren de manera modular.***

