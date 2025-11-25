# Introducción a Docker [Docker Introduction]

Docker es una plataforma que permite empaquetar aplicaciones y sus dependencias en contenedores ligeros y portables. Para este proyecto, Docker nos permite ejecutar ROS 2 Humble y todas sus dependencias sin necesidad de instalarlo directamente en el sistema anfitrión.

## Conceptos clave [Key Concepts]

- **Contenedor [Container]**: Un entorno aislado que contiene una aplicación y todas sus dependencias.
- **Imagen [Image]**: Una plantilla read-only que define cómo construir un contenedor.
- **Dockerfile**: Un archivo de texto que contiene instrucciones para construir una imagen.
- **Volumen [Volume]**: Un mecanismo para compartir archivos entre el contenedor y el sistema anfitrión.

## Ventajas para este proyecto [Advantages]

- **Reproducibilidad**: El mismo entorno funciona en cualquier máquina con Docker.
- **Aislamiento**: No contamina el sistema anfitrión con dependencias de ROS 2.
- **Facilidad**: No requiere instalar ROS 2, PyTorch, y otras dependencias manualmente.

## Uso en este proyecto [Usage]

1. **Construir la imagen**: `./build_docker.sh` crea una imagen con ROS 2 Humble, TurtleSim y PyTorch.
2. **Ejecutar el contenedor**: `./run_docker.sh` inicia un contenedor con soporte gráfico para la simulación.
3. **Trabajar dentro**: El workspace se monta como volumen, permitiendo editar código desde el host.

## Comandos útiles [Useful Commands]

- `docker ps`: Lista contenedores en ejecución
- `docker exec -it ros2_turtle_nn /bin/bash`: Entrar a un contenedor en ejecución
- `docker images`: Lista imágenes disponibles

