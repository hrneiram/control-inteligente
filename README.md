# Proyectos de Control Inteligente para TurtleSim

Este repositorio contiene dos implementaciones de control inteligente para navegación autónoma con evitación de obstáculos en TurtleSim (ROS 2):

## 📁 Estructura del Repositorio

```
.
├── control_fuzzy/              # Control basado en Lógica Difusa
└── control_redes_neuronales/  # Control basado en Redes Neuronales
```

---

## 🎯 Proyectos

### 1. Control Difuso (`control_fuzzy/`)

Implementación de control difuso (fuzzy control) para navegación autónoma. Utiliza reglas heurísticas y conjuntos difusos para tomar decisiones de control.

**Características:**
- Control basado en reglas explícitas
- Lógica difusa con conjuntos de pertenencia
- Implementación directa y fácil de interpretar
- Excelente para entender los fundamentos del control inteligente

**📖 Documentación:** Ver `control_fuzzy/README.md`

---

### 2. Control con Redes Neuronales (`control_redes_neuronales/`)

Implementación de control neuronal usando PyTorch. Utiliza una red neuronal feedforward entrenada con datos sintéticos para navegación autónoma.

**Características:**
- Aprendizaje automático con PyTorch
- Red neuronal feedforward (6→64→32→2)
- Entrenamiento iterativo con scripts mejorados
- Capacidad de mejora continua mediante re-entrenamiento

**📖 Documentación:** Ver `control_redes_neuronales/README.md`

---

## 🚀 Inicio Rápido

### Control Difuso

```bash
cd control_fuzzy/ros2_ws
./build_docker.sh    # Primera vez
./run_docker.sh      # Iniciar contenedor
# Dentro del contenedor:
colcon build --packages-select turtle_fuzzy_control
source install/setup.bash
ros2 run turtlesim turtlesim_node
# En otra terminal:
ros2 run turtle_fuzzy_control fuzzy_visual
```

### Control Neuronal

```bash
cd control_redes_neuronales/ros2_ws
./build_docker.sh    # Primera vez
./run_docker.sh      # Iniciar contenedor
# Dentro del contenedor:
colcon build --packages-select turtle_nn_control
source install/setup.bash
# Entrenar modelo (opcional):
cd src/turtle_nn_control/turtle_nn_control
python3 train_nn_model_improved.py
cd /ros2_ws
# Copiar modelo y ejecutar:
./src/turtle_nn_control/quick_train_and_deploy.sh
ros2 run turtlesim turtlesim_node
# En otra terminal:
ros2 run turtle_nn_control nn_controller
```

---

## 🔄 Comparación de Enfoques

| Aspecto | Control Difuso | Redes Neuronales |
|---------|---------------|------------------|
| **Base** | Reglas heurísticas explícitas | Aprendizaje de datos |
| **Interpretabilidad** | Alta (reglas claras) | Media (caja negra) |
| **Ajuste** | Manual (modificar reglas) | Automático (entrenamiento) |
| **Datos** | No requiere entrenamiento | Requiere datos/entrenamiento |
| **Mejora continua** | Manual | Automática (re-entrenamiento) |
| **Complejidad** | Baja-Media | Media-Alta |

---

## 📚 Documentación

Cada proyecto incluye documentación completa:

- **Control Difuso**: 8 documentos en `control_fuzzy/documents/`
- **Redes Neuronales**: 9 documentos en `control_redes_neuronales/documents/` (incluye guía de entrenamiento iterativo)

---

## 🎓 Uso Académico

Ambos proyectos fueron desarrollados como parte del curso **Control Inteligente** en la Universidad Distrital Francisco José de Caldas, demostrando diferentes enfoques para el mismo problema de navegación autónoma.

---

## 📝 Notas

- Ambos proyectos utilizan **ROS 2 Humble** y **Docker** para reproducibilidad
- Los proyectos son independientes y pueden ejecutarse por separado
- Cada proyecto tiene su propio workspace de ROS 2
- La documentación está en español con referencias en inglés

---

## 🔗 Enlaces Rápidos

- [Control Difuso - README](./control_fuzzy/README.md)
- [Control Neuronal - README](./control_redes_neuronales/README.md)
- [Control Difuso - Documentación](./control_fuzzy/documents/)
- [Control Neuronal - Documentación](./control_redes_neuronales/documents/)

---

**Autor:** Hanssel Neira  
**Asignatura:** Control Inteligente  
**Profesor:** Jorge Federico Ramírez

