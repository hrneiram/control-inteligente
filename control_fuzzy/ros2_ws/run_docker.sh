#!/bin/bash

# Script para ejecutar el contenedor Docker de ROS2 con soporte gráfico

echo "Iniciando contenedor ROS2 con TurtleSim..."

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_DIR="${SCRIPT_DIR}/ros2_ws"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Permitir conexiones X11 desde Docker
xhost +local:docker

# Ejecutar el contenedor
docker run -it --rm \
    --name ros2_turtle \
    --network host \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume "${WORKSPACE_DIR}:/ros2_ws" \
    --volume "${PROJECT_ROOT}:/workspace" \
    ros2_turtlesim:latest \
    /bin/bash

# Limpiar permisos X11 al salir
xhost -local:docker