# Introducción a ROS 2 [ROS 2 Introduction]

ROS 2 (Robot Operating System 2) es un framework de middleware para desarrollo de software robótico. Proporciona herramientas y bibliotecas para construir aplicaciones robóticas distribuidas.

## Conceptos fundamentales [Fundamental Concepts]

- **Nodo [Node]**: Un proceso que realiza una función específica (por ejemplo, publicar comandos de velocidad).
- **Tópico [Topic]**: Un canal de comunicación asíncrono donde los nodos publican y se suscriben a mensajes.
- **Mensaje [Message]**: Estructura de datos que se intercambia entre nodos (por ejemplo, `Twist` para velocidades).
- **Servicio [Service]**: Comunicación síncrona request-response entre nodos.
- **Workspace**: Directorio donde se organizan y compilan los paquetes ROS 2.

## Arquitectura del proyecto [Project Architecture]

```
turtle_nn_controller (Nodo)
    ├── Se suscribe a: /turtle1/pose (posición actual)
    ├── Publica en: /turtle1/cmd_vel (comandos de velocidad)
    └── Usa servicios: /spawn, /kill, /teleport_absolute, /set_pen
```

## Flujo de datos [Data Flow]

1. El nodo `turtle_nn_controller` se suscribe a `/turtle1/pose` para conocer la posición actual.
2. Calcula las entradas para la red neuronal (distancias, ángulos).
3. La red neuronal produce comandos de velocidad.
4. Publica los comandos en `/turtle1/cmd_vel`.
5. TurtleSim recibe los comandos y mueve la tortuga.

## Comandos útiles [Useful Commands]

- `ros2 run turtlesim turtlesim_node`: Inicia el simulador
- `ros2 run turtle_nn_control nn_controller`: Ejecuta el controlador neuronal
- `ros2 topic list`: Lista todos los tópicos activos
- `ros2 topic echo /turtle1/pose`: Muestra mensajes del tópico de posición

