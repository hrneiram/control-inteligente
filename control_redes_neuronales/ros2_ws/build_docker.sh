#!/bin/bash

# Script para construir la imagen Docker de ROS2 con soporte para redes neuronales

echo "🚀 Construyendo imagen Docker de ROS2 con TurtleSim y PyTorch..."

docker build -t ros2_turtlesim_nn:latest .

if [ $? -eq 0 ]; then
    echo "✅ Imagen construida exitosamente!"
    echo "📦 Nombre de la imagen: ros2_turtlesim_nn:latest"
else
    echo "❌ Error al construir la imagen"
    exit 1
fi

