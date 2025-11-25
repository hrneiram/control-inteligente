# Estructura de Paquetes [Package Structure]

Este documento describe cómo se organiza el código dentro de los paquetes ROS 2 presentes en `ros2_turtle_project` y cómo se utilizan en la ejecución.

## Visión general [Overview]

Dentro del workspace montado en `/ros2_ws` existen dos paquetes principales:

1. `turtle_fuzzy_control`
2. `turtle_fuzzy_obstacles` (marcado como legado pero conservado como referencia)

Cada paquete sigue la convención estándar de ROS 2: `package.xml`, `setup.py`, `setup.cfg`, carpeta `resource/` y el módulo Python en `nombre_del_paquete/`.

## Paquete activo: `turtle_fuzzy_control` [Active Package]

- **`setup.py`** define el entry point `fuzzy_visual`, que se publica con `ros2 run turtle_fuzzy_control fuzzy_visual`.
- **`turtle_fuzzy_control/turtle_fuzzy_obstacles_visual.py`** contiene la lógica de:
  - Detección de obstáculos (`Obstacle` class).
  - Dibujo de círculos rojos y del objetivo verde mediante servicios de `turtlesim`.
  - Cálculo de las velocidades lineales/angulares con base en reglas difusas.
- **Recursos [Resources]**: el archivo `resource/turtle_fuzzy_control` registra el nombre del paquete para el índice de ament.

## Paquete legado: `turtle_fuzzy_obstacles` [Legacy Package]

Se mantiene para documentación histórica. Las entradas de consola en `setup.py` están comentadas, por lo que no se instalan nodos adicionales. Sirve como repositorio de versiones previas y como guía para crear nuevos paquetes si fuera necesario.

## Flujo de instalación [Installation Flow]

Cuando se ejecuta `colcon build`:

1. Se generan los artefactos en `build/`.
2. Se instalan scripts y módulos en `install/<paquete>/`.
3. El script `setup.bash` generado permite que `ros2 run` encuentre los nodos distribuidos.

Gracias a esta estructura, los controladores pueden evolucionar sin interferir con otros proyectos. Añadir una nueva variante implica crear un nuevo script en el módulo Python y registrarlo como entry point en `setup.py`.***

