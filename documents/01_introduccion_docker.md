# Introducción a Docker [Docker Introduction]

Este documento explica qué es Docker y cómo se aplica dentro del proyecto `ros2_turtle_project`.

## ¿Qué es Docker? [Docker Basics]

Docker es una plataforma de virtualización ligera basada en contenedores. Un contenedor encapsula aplicaciones junto con todas sus dependencias, lo que garantiza que se ejecuten de manera consistente sin importar el host. A diferencia de las máquinas virtuales tradicionales, los contenedores comparten el kernel del sistema operativo, lo que reduce el consumo de recursos y acelera el arranque.

## Beneficios para el proyecto [Project Benefits]

- **Reproducibilidad [Reproducibility]**: La imagen `ros2_turtlesim:latest` describe cada dependencia necesaria para ejecutar ROS 2 Humble y `turtlesim`. Esto evita conflictos de versiones.
- **Portabilidad [Portability]**: El contenedor puede ejecutarse en cualquier equipo con Docker, sin requerir instalaciones complejas de ROS.
- **Aislamiento [Isolation]**: Los procesos del controlador y del simulador quedan separados del sistema anfitrión, evitando que paquetes externos modifiquen el entorno.

## Flujo de trabajo [Workflow]

1. Construcción con `./build_docker.sh`, que invoca `docker build` usando el `Dockerfile`.
2. Ejecución mediante `./run_docker.sh`, que lanza el contenedor con soporte gráfico (`X11`) y monta dos volúmenes:
   - `/ros2_ws`: workspace de ROS 2 para las compilaciones con `colcon`.
   - `/workspace`: raíz del proyecto, donde reside `ros2_turtle_project`.
3. Dentro del contenedor, se inicializa el entorno con `source /opt/ros/humble/setup.bash` y `source /ros2_ws/install/setup.bash`.

## Uso en el proyecto [Usage in Project]

- Permite arrancar rápidamente el simulador `turtlesim` (`ros2 run turtlesim turtlesim_node`).
- Facilita ejecutar el controlador difuso empaquetado (`ros2 run turtle_fuzzy_control fuzzy_visual`) sin instalar ROS en el host.
- Sirve como base para compartir el proyecto con otros integrantes del equipo, garantizando que todos dispongan de la misma configuración.

En resumen, Docker aporta un entorno controlado, repetible y fácil de distribuir para experimentar con el control difuso y la evitación de obstáculos en ROS 2.***

