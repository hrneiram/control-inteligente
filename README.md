# Proyecto ROS2 Turtle – Evitación Difusa de Obstáculos

Este repositorio reúne un ejemplo completo de navegación con control difuso sobre ROS 2 Humble y el simulador clásico `turtlesim`. El objetivo es que la tortuga alcance metas cambiantes mientras evita obstáculos circulares dibujados en pantalla.

Todo el entorno puede replicarse mediante Docker, por lo que no es necesario instalar ROS 2 en el equipo anfitrión.

---

## Tecnologías utilizadas

- **Docker**: contenedores reproducibles para ROS 2 y `turtlesim`.
- **ROS 2 Humble**: middleware robótico, nodos y servicios.
- **Python 3**: implementación del controlador y scripts auxiliares.
- **colcon**: herramienta de construcción de workspaces ROS 2.
- **NumPy**: utilidades matemáticas para cálculos de control.
- **turtlesim**: simulador 2D que emula un robot diferencial tipo TurtleBot.

---

## Arquitectura del proyecto

```
ros2_turtle_project/
├── .gitignore
├── documents/                 # Documentación en español (Docker, ROS 2, control difuso, etc.)
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
            └── turtle_fuzzy_control/
                ├── package.xml
                ├── resource/
                ├── setup.cfg
                ├── setup.py
                └── turtle_fuzzy_control/
                    └── turtle_fuzzy_obstacles_visual.py
```

- Los archivos `obstacles*.py` originales se han dejado como referencia histórica y no forman parte del flujo principal.
- La carpeta `documents/` contiene ocho guías que explican la teoría y la práctica del proyecto (Docker, ROS 2, estructura de paquetes, control difuso, etc.).
- El paquete activo es `turtle_fuzzy_control`, que expone el nodo ejecutable `fuzzy_visual`.

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
4. Dentro del contenedor, prepara el entorno y ejecuta el simulador:
   ```bash
   source /opt/ros/humble/setup.bash
   source /ros2_ws/install/setup.bash
   ros2 run turtlesim turtlesim_node
   ```
5. En otra terminal del host, adjúntate al mismo contenedor:
   ```bash
   docker exec -it ros2_turtle /bin/bash
   source /opt/ros/humble/setup.bash
   source /ros2_ws/install/setup.bash
   ros2 run turtle_fuzzy_control fuzzy_visual
   ```
   La tortuga comenzará a navegar automáticamente usando el control difuso.
6. Para terminar, cierra ambas terminales del contenedor; los permisos de X11 se restauran solos.

### 2. Ejecución nativa (sin Docker)

1. Instala ROS 2 Humble, `turtlesim` y herramientas básicas (`colcon`, `python3-colcon-common-extensions`).
2. Desde la raíz del proyecto, entra al workspace:
   ```bash
   cd ros2_ws/ros2_ws
   ```
3. Compila el paquete una vez:
   ```bash
   colcon build
   ```
4. En cada terminal que vayas a utilizar:
   ```bash
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   ```
5. Lanza el simulador y, en otra terminal, el controlador:
   ```bash
   ros2 run turtlesim turtlesim_node
   ros2 run turtle_fuzzy_control fuzzy_visual
   ```

---

## Flujo de control

1. El nodo `fuzzy_visual` dibuja círculos rojos (obstáculos) y un objetivo verde utilizando servicios de `turtlesim`.
2. Se suscribe a `/turtle1/pose` para conocer la posición actual de la tortuga.
3. En cada iteración (100 ms):
   - Calcula la distancia al objetivo y al obstáculo más peligroso.
   - Evalúa la urgencia de evitación y el error angular.
   - Aplica reglas difusas para obtener velocidades lineal y angular.
4. Publica los comandos en `/turtle1/cmd_vel`.

Toda la lógica está documentada con mayor detalle en `documents/06_introduccion_control_difuso.md`, `documents/07_calculo_manual_parametros.md` y `documents/08_implementacion_control_difuso.md`.

---

## Extensión y personalización

- **Nuevos obstáculos**: edita la lista `self.obstacles` en `turtle_fuzzy_obstacles_visual.py`.
- **Ajuste de reglas**: modifica los umbrales de distancia/urgencia dentro de `apply_fuzzy_control`.
- **Registro de datos**: amplía el contador de logs o integra `rqt_plot` para inspeccionar las variables en tiempo real.
- **Migración a hardware**: reemplaza los tópicos de `turtlesim` por los correspondientes a TurtleBot (`/cmd_vel`, `/odom`, sensores reales).

---

## Resolución de problemas

- **No aparece la ventana de `turtlesim`**: verifica que hayas ejecutado `xhost +local:docker` antes de abrir el contenedor y que la variable `DISPLAY` esté exportada.
- **Errores al lanzar el controlador**: asegúrate de que `turtlesim_node` se esté ejecutando; sin él, los servicios `/spawn` y `/kill` no estarán disponibles.
- **La tortuga gira sin avanzar**: revisa la urgencia calculada; si está saturada puede deberse a obstáculos demasiado próximos. Ajusta radios o umbrales en el controlador.

---

## Lecturas recomendadas

- Lotfi A. Zadeh, “Fuzzy sets”, *Information and Control*, 1965.
- Kevin Passino & Stephen Yurkovich, *Fuzzy Control*, Addison-Wesley, 1998.
- Documentación oficial de ROS 2 sobre [Timers](https://docs.ros.org/en/humble/Tutorials/Intermediate/Timers/Timers.html) y [Service Clients](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Client-Library-Tutorials.html).

Con estos recursos puedes profundizar en la teoría e incluso extender el proyecto con motores de inferencia difusa más sofisticados.***


