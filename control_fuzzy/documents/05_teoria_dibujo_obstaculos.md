# Teoría del Dibujo de Obstáculos [Obstacle Drawing Theory]

El controlador visual genera obstáculos circulares y un objetivo en el simulador. Este documento explica la teoría y los pasos prácticos detrás de ese proceso.

## Concepto geométrico [Geometric Concept]

Los obstáculos se modelan como círculos definidos por su centro `(x, y)` y un radio `r`. Al visualizarse como circunferencias, resulta intuitivo estimar distancias mínimas y detectar colisiones, dado que la norma euclidiana es fácil de calcular.

## Servicios utilizados [Services Used]

`turtlesim` ofrece servicios ROS 2 para manipular tortugas:

- `/spawn` [Spawn Service]: crea una tortuga en la posición indicada.
- `/kill` [Kill Service]: elimina una tortuga existente.
- `/<tortuga>/set_pen` [Set Pen Service]: configura color y grosor del trazo.
- `/<tortuga>/cmd_vel` [Cmd Vel Topic]: recibe comandos de velocidad para mover la tortuga dibujante.

## Procedimiento [Procedure]

1. **Crear tortuga dibujante [Drawing Turtle]**: Se invoca `/spawn` posicionándola en el borde del círculo.
2. **Configurar lápiz [Pen Setup]**: Con `set_pen` se define color rojo (`255, 0, 0`) y grosor deseado.
3. **Mover en trayectoria circular [Circular Motion]**:
   - Se publica una velocidad lineal constante.
   - Se calcula la velocidad angular como `linear_speed / radius` para mantener la tortuga en un círculo.
   - El tiempo total de giro es `2πr / linear_speed`.
4. **Eliminar tortuga temporal [Cleanup]**: Se ejecuta `/kill` para no saturar el entorno con tortugas extra.

## Representación del objetivo [Goal Rendering]

El objetivo se dibuja de forma análoga, pero con un radio pequeño (`0.3`) y color verde (`0, 255, 0`) para diferenciarlo visualmente.

## Ventajas del método [Advantages]

- **Consistencia [Consistency]**: Todos los obstáculos comparten la misma rutina de dibujo, garantizando círculos uniformes.
- **Modularidad [Modularity]**: Al encapsular el comportamiento en la clase `Obstacle`, es sencillo ajustar la posición o el radio sin modificar el flujo principal.
- **Sin dependencia de gráficos externos [External Independence]**: Se aprovecha el propio motor de `turtlesim`, evitando librerías adicionales.

Así, la teoría geométrica básica de los círculos combinada con los servicios de ROS 2 permite visualizar claramente zonas peligrosas y objetivos dentro del simulador.***

