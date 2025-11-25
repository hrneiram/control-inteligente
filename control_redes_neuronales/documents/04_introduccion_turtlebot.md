# Introducción a TurtleSim [TurtleSim Introduction]

TurtleSim es un simulador 2D simple incluido en ROS 2 que emula un robot diferencial tipo TurtleBot. Es ideal para aprender y probar algoritmos de navegación.

## Conceptos [Concepts]

- **Robot diferencial**: Robot con dos ruedas independientes que se mueve variando las velocidades de cada rueda.
- **Pose**: Posición (x, y) y orientación (theta) del robot.
- **Velocidad lineal**: Velocidad de avance/retroceso (m/s).
- **Velocidad angular**: Velocidad de rotación (rad/s).

## Tópicos principales [Main Topics]

- `/turtle1/pose`: Publica la pose actual (x, y, theta, linear_velocity, angular_velocity).
- `/turtle1/cmd_vel`: Recibe comandos de velocidad (linear.x, angular.z).

## Servicios disponibles [Available Services]

- `/spawn`: Crea una nueva tortuga en una posición específica.
- `/kill`: Elimina una tortuga.
- `/teleport_absolute`: Teletransporta una tortuga a una posición.
- `/set_pen`: Configura el color y grosor del trazo.

## Coordenadas [Coordinates]

- **Origen**: Esquina inferior izquierda (0, 0).
- **Rango**: Aproximadamente [0, 11] en ambos ejes.
- **Ángulo**: En radianes, 0 apunta hacia la derecha, crece en sentido antihorario.

## Uso en el proyecto [Usage in Project]

El controlador neuronal:
1. Lee la pose actual desde `/turtle1/pose`.
2. Calcula distancias y ángulos al objetivo y obstáculos.
3. Usa la red neuronal para determinar velocidades.
4. Publica comandos en `/turtle1/cmd_vel`.

