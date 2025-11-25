# Teoría del Dibujo de Obstáculos [Obstacle Drawing Theory]

Este documento explica cómo se dibujan los obstáculos y el objetivo en TurtleSim usando las capacidades gráficas del simulador.

## Método de dibujo [Drawing Method]

TurtleSim permite dibujar trazando el camino de las tortugas. Para crear formas visuales:

1. **Spawn temporal**: Se crea una tortuga temporal en una posición específica.
2. **Configuración del lápiz**: Se establece color (RGB) y grosor del trazo.
3. **Movimiento circular**: La tortuga se mueve en círculo para dibujar el obstáculo.
4. **Eliminación**: La tortuga temporal se elimina después de dibujar.

## Cálculo de círculos [Circle Calculation]

Para dibujar un círculo de radio `r`:
- **Velocidad lineal**: Constante (ej: 1.0 m/s)
- **Velocidad angular**: `ω = v / r` (rad/s)
- **Tiempo**: `t = 2πr / v` (segundos)

La tortuga gira mientras avanza, creando un círculo perfecto.

## Representación visual [Visual Representation]

- **Obstáculos**: Círculos rojos con grosor 3.
- **Objetivo**: Círculo verde pequeño con grosor 2.

## Implementación [Implementation]

La función `draw_circle()` en `turtle_nn_controller.py`:
1. Publica comandos de velocidad en el tópico de la tortuga temporal.
2. Mantiene las velocidades durante el tiempo necesario.
3. Detiene la tortuga al completar el círculo.

## Ventajas [Advantages]

- Visualización clara de obstáculos y objetivos.
- No requiere herramientas externas de visualización.
- Los obstáculos permanecen visibles durante toda la simulación.

