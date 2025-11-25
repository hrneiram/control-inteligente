# Proyecto ROS2 Turtle – Control Neuronal con Redes Neuronales

Este repositorio presenta un ejemplo completo de navegación con control basado en redes neuronales sobre ROS 2 Humble y el simulador `turtlesim`. El objetivo es que la tortuga alcance metas cambiantes mientras evita obstáculos circulares dibujados en pantalla, utilizando una red neuronal feedforward entrenada con PyTorch.

Todo el entorno puede replicarse mediante Docker, por lo que no es necesario instalar ROS 2 en el equipo anfitrión.

---

## Tecnologías utilizadas

- **Docker**: contenedores reproducibles para ROS 2 y `turtlesim`.
- **ROS 2 Humble**: middleware robótico, nodos y servicios.
- **Python 3**: implementación del controlador y scripts auxiliares.
- **PyTorch**: framework de deep learning para la red neuronal.
- **colcon**: herramienta de construcción de workspaces ROS 2.
- **NumPy**: utilidades matemáticas para cálculos de control.
- **turtlesim**: simulador 2D que emula un robot diferencial tipo TurtleBot.

---

## Arquitectura del proyecto

```
ros2_turtle_nn_project/
├── .gitignore
├── documents/                 # Documentación en español (Docker, ROS 2, redes neuronales, etc.)
├── README.md
└── ros2_ws/
    ├── Dockerfile
    ├── build_docker.sh
    ├── run_docker.sh
    └── ros2_ws/               # Workspace de ROS 2 utilizado dentro del contenedor
        ├── build/             # Generado por colcon (ignorados en git)
        ├── install/
        ├── log/
        └── src/
            └── turtle_nn_control/
                ├── package.xml
                ├── resource/
                ├── setup.cfg
                ├── setup.py
                └── turtle_nn_control/
                    ├── __init__.py
                    ├── turtle_nn_controller.py    # Controlador principal
                    └── train_nn_model.py          # Script de entrenamiento
```

- La carpeta `documents/` contiene ocho guías que explican la teoría y la práctica del proyecto (Docker, ROS 2, estructura de paquetes, redes neuronales, entrenamiento, etc.).
- El paquete activo es `turtle_nn_control`, que expone el nodo ejecutable `nn_controller`.

---

## Cómo ejecutar el proyecto

### 1. Lanzamiento con Docker (recomendado)

1. Desde la raíz del repositorio entra a la carpeta de utilidades:
   ```bash
   cd ros2_ws
   ```

2. Construye la imagen base (solo la primera vez o cuando cambie el `Dockerfile`):
   ```bash
   ./build_docker.sh
   ```

3. Inicia el contenedor (habilita automáticamente el soporte gráfico):
   ```bash
   ./run_docker.sh
   ```

4. Dentro del contenedor, prepara el entorno y compila el paquete:
   ```bash
   source /opt/ros/humble/setup.bash
   cd /ros2_ws
   colcon build --packages-select turtle_nn_control
   source install/setup.bash
   ```

5. (Opcional) Entrena el modelo neuronal antes de ejecutar:
   
   **Método rápido (recomendado):**
   ```bash
   cd /ros2_ws
   ./src/turtle_nn_control/quick_train_and_deploy.sh
   source install/setup.bash
   ```
   
   **Método manual:**
   ```bash
   cd src/turtle_nn_control/turtle_nn_control
   # Entrenamiento básico
   python3 train_nn_model.py
   # O entrenamiento mejorado con más control
   python3 train_nn_model_improved.py --epochs 150 --samples 10000 --obstacle_focus 0.4
   cd /ros2_ws
   colcon build --packages-select turtle_nn_control
   cp src/turtle_nn_control/turtle_nn_control/turtle_nn_model.pth \
      install/turtle_nn_control/lib/python3.10/site-packages/turtle_nn_control/
   source install/setup.bash
   ```
   
   **📖 Para una guía completa de entrenamiento iterativo, consulta:**
   `documents/09_guia_entrenamiento_iterativo.md`

6. Ejecuta el simulador:
   ```bash
   ros2 run turtlesim turtlesim_node
   ```

7. En otra terminal del host, adjúntate al mismo contenedor:
   ```bash
   docker exec -it ros2_turtle_nn /bin/bash
   source /opt/ros/humble/setup.bash
   source /ros2_ws/install/setup.bash
   ros2 run turtle_nn_control nn_controller
   ```
   La tortuga comenzará a navegar automáticamente usando el control neuronal.

8. Para terminar, cierra ambas terminales del contenedor; los permisos de X11 se restauran solos.

### 2. Ejecución nativa (sin Docker)

1. Instala ROS 2 Humble, `turtlesim`, PyTorch y herramientas básicas (`colcon`, `python3-colcon-common-extensions`).
2. Desde la raíz del proyecto, entra al workspace:
   ```bash
   cd ros2_ws/ros2_ws
   ```
3. Compila el paquete una vez:
   ```bash
   colcon build --packages-select turtle_nn_control
   ```
4. (Opcional) Entrena el modelo:
   ```bash
   cd src/turtle_nn_control/turtle_nn_control
   python3 train_nn_model.py
   cd ../../..
   ```
5. En cada terminal que vayas a utilizar:
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ```
6. Lanza el simulador y, en otra terminal, el controlador:
   ```bash
   ros2 run turtlesim turtlesim_node
   ros2 run turtle_nn_control nn_controller
   ```

---

## Flujo de control

1. El nodo `nn_controller` dibuja círculos rojos (obstáculos) y un objetivo verde utilizando servicios de `turtlesim`.
2. Se suscribe a `/turtle1/pose` para conocer la posición actual de la tortuga.
3. En cada iteración (100 ms):
   - Calcula la distancia al objetivo y al obstáculo más peligroso.
   - Normaliza las entradas (distancias, ángulos, velocidades actuales).
   - Pasa las entradas por la red neuronal feedforward.
   - Obtiene velocidades lineal y angular como salida.
4. Publica los comandos en `/turtle1/cmd_vel`.

Toda la lógica está documentada con mayor detalle en `documents/06_introduccion_redes_neuronales.md`, `documents/07_entrenamiento_modelo.md` y `documents/08_implementacion_control_neural.md`.

---

## Arquitectura de la red neuronal

La red neuronal tiene la siguiente estructura:

- **Entrada**: 6 neuronas (distancia al objetivo, error angular, distancia al obstáculo, ángulo al obstáculo, velocidades actuales)
- **Capa oculta 1**: 64 neuronas con ReLU y Dropout
- **Capa oculta 2**: 32 neuronas con ReLU
- **Salida**: 2 neuronas (velocidad lineal con sigmoid, velocidad angular con tanh)

---

## Entrenamiento del modelo

El modelo puede entrenarse usando datos sintéticos generados con reglas heurísticas.

### Entrenamiento Básico

```bash
cd ros2_ws/ros2_ws/src/turtle_nn_control/turtle_nn_control
python3 train_nn_model.py
```

### Entrenamiento Mejorado (Recomendado)

El script `train_nn_model_improved.py` ofrece mejor evitación de obstáculos y más opciones:

```bash
# Entrenamiento con parámetros por defecto mejorados
python3 train_nn_model_improved.py

# Entrenamiento personalizado
python3 train_nn_model_improved.py \
    --epochs 200 \
    --samples 15000 \
    --obstacle_focus 0.5 \
    --lr 0.0008
```

**Parámetros disponibles:**
- `--epochs`: Número de épocas (default: 150)
- `--samples`: Número de muestras de entrenamiento (default: 10000)
- `--obstacle_focus`: Proporción de ejemplos con obstáculos cercanos 0.0-1.0 (default: 0.4)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Tamaño del batch (default: 32)
- `--hidden1`, `--hidden2`: Tamaños de capas ocultas

### Script de Entrenamiento y Despliegue Rápido

```bash
cd /ros2_ws
./src/turtle_nn_control/quick_train_and_deploy.sh [epochs] [samples] [obstacle_focus]
```

Esto entrena, compila y despliega el modelo en un solo comando.

### Guía Completa de Entrenamiento Iterativo

Para mejorar sistemáticamente el modelo mediante iteraciones, consulta la guía extensa:

📖 **`documents/09_guia_entrenamiento_iterativo.md`**

Esta guía cubre:
- Flujo completo de trabajo
- Cómo compilar y desplegar
- Evaluación y análisis del comportamiento
- Iteración y mejora continua
- Ajuste de hiperparámetros
- Mejora de datos de entrenamiento
- Debugging y solución de problemas
- Checklist de mejora continua

El modelo entrenado se guarda como `turtle_nn_model.pth` y será cargado automáticamente por el controlador. Si no existe un modelo pre-entrenado, el controlador usará pesos aleatorios (puede funcionar pero con rendimiento limitado).

---

## Extensión y personalización

- **Nuevos obstáculos**: edita la lista `self.obstacles` en `turtle_nn_controller.py`.
- **Ajuste de la red**: modifica la arquitectura en `NeuralNetworkController` (tamaños de capas, funciones de activación).
- **Mejora del entrenamiento**: ajusta las reglas heurísticas en `train_nn_model.py` o recopila datos reales.
- **Aprendizaje por refuerzo**: implementa un sistema de recompensas para entrenar con RL.
- **Migración a hardware**: reemplaza los tópicos de `turtlesim` por los correspondientes a TurtleBot (`/cmd_vel`, `/odom`, sensores reales).

---

## Resolución de problemas

- **No aparece la ventana de `turtlesim`**: verifica que hayas ejecutado `xhost +local:docker` antes de abrir el contenedor y que la variable `DISPLAY` esté exportada.
- **Errores al lanzar el controlador**: asegúrate de que `turtlesim_node` se esté ejecutando; sin él, los servicios `/spawn` y `/kill` no estarán disponibles.
- **La tortuga no navega bien**: entrena el modelo primero con `train_nn_model.py` para obtener mejores resultados. Los pesos aleatorios pueden no funcionar correctamente.
- **Errores de PyTorch**: verifica que PyTorch esté instalado correctamente en el contenedor. El Dockerfile incluye la instalación automática.

---

## Comparación con control difuso

Este proyecto usa redes neuronales en lugar de control difuso. Ventajas:

- **Aprendizaje automático**: Puede mejorar con más datos de entrenamiento.
- **Generalización**: Aprende patrones complejos de los datos.
- **Adaptabilidad**: Puede ajustarse a diferentes escenarios con re-entrenamiento.

Desventajas:

- **Requiere entrenamiento**: Necesita datos o reglas para generar datos sintéticos.
- **Menos interpretable**: Las decisiones de la red son menos transparentes que las reglas difusas.

---

## Lecturas recomendadas

- Ian Goodfellow, Yoshua Bengio & Aaron Courville, *Deep Learning*, MIT Press, 2016.
- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Documentación oficial de ROS 2 sobre [Timers](https://docs.ros.org/en/humble/Tutorials/Intermediate/Timers/Timers.html) y [Service Clients](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Client-Library-Tutorials.html).

Con estos recursos puedes profundizar en la teoría de redes neuronales e incluso extender el proyecto con arquitecturas más sofisticadas (CNN, LSTM, etc.) o técnicas de aprendizaje por refuerzo.

