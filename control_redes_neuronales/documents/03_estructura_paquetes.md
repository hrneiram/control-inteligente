# Estructura de Paquetes ROS 2 [ROS 2 Package Structure]

Este documento explica la estructura del paquete `turtle_nn_control` y cómo se organiza el código.

## Estructura del paquete [Package Structure]

```
turtle_nn_control/
├── package.xml          # Metadatos y dependencias del paquete
├── setup.py             # Configuración de instalación Python
├── setup.cfg            # Configuración adicional
├── resource/
│   └── turtle_nn_control  # Marcador de recurso
└── turtle_nn_control/   # Módulo Python principal
    ├── __init__.py
    ├── turtle_nn_controller.py  # Controlador principal con red neuronal
    └── train_nn_model.py        # Script de entrenamiento
```

## Componentes principales [Main Components]

### `turtle_nn_controller.py`

Contiene:
- `NeuralNetworkController`: Clase que define la arquitectura de la red neuronal.
- `TurtleNNController`: Nodo ROS 2 que implementa el control basado en redes neuronales.
- `Obstacle`: Clase auxiliar para representar obstáculos circulares.

### `train_nn_model.py`

Script independiente para:
- Generar datos de entrenamiento sintéticos.
- Entrenar la red neuronal.
- Guardar el modelo entrenado en `turtle_nn_model.pth`.

## Dependencias [Dependencies]

- `rclpy`: Cliente Python de ROS 2
- `geometry_msgs`: Mensajes geométricos (Twist, Pose)
- `turtlesim`: Simulador de tortuga
- `torch`: PyTorch para redes neuronales
- `numpy`: Operaciones numéricas

## Compilación [Building]

```bash
cd ros2_ws
colcon build --packages-select turtle_nn_control
source install/setup.bash
```

## Ejecución [Running]

```bash
ros2 run turtle_nn_control nn_controller
```

