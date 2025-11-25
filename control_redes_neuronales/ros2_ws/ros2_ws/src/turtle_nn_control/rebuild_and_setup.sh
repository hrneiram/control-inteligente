#!/bin/bash

# Script para recompilar el paquete y copiar el modelo entrenado

echo "🔨 Recompilando paquete turtle_nn_control..."

# Ir al directorio del workspace
cd /ros2_ws

# Recompilar el paquete
colcon build --packages-select turtle_nn_control

if [ $? -eq 0 ]; then
    echo "✅ Paquete recompilado exitosamente!"
    
    # Copiar el modelo entrenado al directorio de instalación si existe
    MODEL_SOURCE="/ros2_ws/src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth"
    MODEL_DEST="/ros2_ws/install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/turtle_nn_model.pth"
    
    if [ -f "$MODEL_SOURCE" ]; then
        echo "📦 Copiando modelo entrenado..."
        mkdir -p "$(dirname "$MODEL_DEST")"
        cp "$MODEL_SOURCE" "$MODEL_DEST"
        echo "✅ Modelo copiado a: $MODEL_DEST"
    else
        echo "⚠️  Modelo no encontrado en: $MODEL_SOURCE"
        echo "   Ejecuta primero: python3 train_nn_model.py"
    fi
    
    echo ""
    echo "🎯 Para usar el paquete, ejecuta:"
    echo "   source /ros2_ws/install/setup.bash"
else
    echo "❌ Error al recompilar el paquete"
    exit 1
fi

