#!/bin/bash

# Script rápido para entrenar, compilar y desplegar en un solo comando

set -e  # Salir si hay error

echo "🚀 Entrenamiento y Despliegue Rápido"
echo "===================================="
echo ""

# Ir al directorio del módulo
cd /ros2_ws/src/turtle_nn_control/turtle_nn_control

# Verificar que el script de entrenamiento existe
if [ ! -f "train_nn_model_improved.py" ]; then
    echo "⚠️  train_nn_model_improved.py no encontrado, usando train_nn_model.py"
    TRAIN_SCRIPT="train_nn_model.py"
else
    TRAIN_SCRIPT="train_nn_model_improved.py"
fi

# Parámetros por defecto (pueden ser sobrescritos con argumentos)
EPOCHS=${1:-150}
SAMPLES=${2:-10000}
OBSTACLE_FOCUS=${3:-0.4}

echo "📋 Parámetros:"
echo "   Épocas: $EPOCHS"
echo "   Muestras: $SAMPLES"
echo "   Foco en obstáculos: $OBSTACLE_FOCUS"
echo ""

# Entrenar
echo "🧠 Entrenando modelo..."
if [ "$TRAIN_SCRIPT" == "train_nn_model_improved.py" ]; then
    python3 train_nn_model_improved.py \
        --epochs $EPOCHS \
        --samples $SAMPLES \
        --obstacle_focus $OBSTACLE_FOCUS
else
    python3 train_nn_model.py
fi

if [ $? -ne 0 ]; then
    echo "❌ Error en el entrenamiento"
    exit 1
fi

echo ""
echo "🔨 Recompilando paquete..."

# Ir al workspace
cd /ros2_ws

# Recompilar
colcon build --packages-select turtle_nn_control

if [ $? -ne 0 ]; then
    echo "❌ Error en la compilación"
    exit 1
fi

echo ""
echo "📦 Copiando modelo..."

# Copiar modelo
MODEL_SOURCE="src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth"
MODEL_DEST="install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/turtle_nn_model.pth"

if [ ! -f "$MODEL_SOURCE" ]; then
    echo "❌ Modelo no encontrado en: $MODEL_SOURCE"
    exit 1
fi

mkdir -p "$(dirname "$MODEL_DEST")"
cp "$MODEL_SOURCE" "$MODEL_DEST"

echo ""
echo "✅ ¡Proceso completado exitosamente!"
echo ""
echo "🎯 Para ejecutar:"
echo "   source /ros2_ws/install/setup.bash"
echo "   ros2 run turtle_nn_control nn_controller"
echo ""
echo "💡 O ejecuta todo en un comando:"
echo "   source /ros2_ws/install/setup.bash && ros2 run turtle_nn_control nn_controller"

