# Cálculo Manual de Parámetros Difusos [Manual Parameter Calculation]

Aunque el código implementa reglas heurísticas, es útil entender cómo se pueden seleccionar los parámetros de manera manual antes de codificarlos.

## Paso 1: Definir universos de discurso [Universe of Discourse]

- **Distancia al objetivo [Distance to Goal]**: valor escalar entre 0 y 10 unidades (el tablero de `turtlesim` mide 11×11). Representa la separación euclidiana entre la tortuga y el punto meta.
- **Error angular [Angular Error]**: diferencia entre el rumbo actual de la tortuga y el ángulo hacia el objetivo o hacia la dirección de evitación. Varía entre `-π` y `π`, donde 0 significa total alineación.
- **Urgencia de obstáculo [Obstacle Urgency]**: número adimensional entre 0 y 1 que resume el nivel de peligro de colisión. Cero indica que el obstáculo está lejos; uno significa que es necesario reaccionar de inmediato.
- **Velocidad lineal [Linear Velocity]**: velocidad hacia adelante (m/s) que impulsa a la tortuga a avanzar. En `turtlesim` se aplica sobre el eje x del robot.
- **Velocidad angular [Angular Velocity]**: tasa de giro (rad/s) que orienta el robot. Valores positivos giran en sentido antihorario.

## Paso 2: Establecer categorías lingüísticas [Linguistic Categories]

Para cada variable se eligen tres estados sencillos:

| Variable | Estados propuestos | Rango aproximado |
|----------|-------------------|------------------|
| Distancia | *Cerca*, *Media*, *Lejos* | [0–1], [1–3], [3–10] |
| Error angular | *Pequeño*, *Medio*, *Grande* | [0–0.4], [0.4–1.2], [>1.2] |
| Urgencia | *Baja*, *Media*, *Alta* | [0–0.3], [0.3–0.6], [>0.6] |

## Paso 3: Asignar funciones de pertenencia [Membership Functions]

El objetivo es traducir números (por ejemplo, “distancia = 2.4”) a grados de pertenencia en etiquetas como *Cerca* o *Media*. Para mantener el diseño simple y flexible se utilizan funciones triangulares. Cada triángulo tiene:

- Un punto donde la pertenencia es 1 (pico o valor representativo).
- Dos puntos donde la pertenencia es 0 (límite inferior y superior del triángulo).

### Ejemplo 1: Distancia [Example – Distance]

Suponiendo tres etiquetas (*Cerca*, *Media*, *Lejos*), podemos definir:

- *Cerca*: triángulo con base entre 0 y 2.0 y pico en 1.0.
- *Media*: triángulo con base entre 1.0 y 5.0 y pico en 3.0.
- *Lejos*: triángulo con base entre 3.0 y 10.0 y pico en 7.0.

Si la distancia vale 1.5:

- Pertenencia en *Cerca*: `(2.0 - 1.5) / (2.0 - 1.0) = 0.5`.
- Pertenencia en *Media*: `(1.5 - 1.0) / (3.0 - 1.0) = 0.25`.
- Pertenencia en *Lejos*: 0 (todavía no entra en ese triángulo).

Esto significa que la distancia “se siente” mitad *Cerca* y una cuarta parte *Media*, lo cual coincide con la intuición de que 1.5 está más cerca que lejos.

Visualmente, cada triángulo puede imaginarse así:

```
Pertenencia
1.0 |        /\
    |       /  \
0.0 +------+----+----> Distancia
      0    1    2
     Cerca
```

### Ejemplo 2: Error angular [Example – Angular Error]

Se trabaja con el valor absoluto del error:

- *Pequeño*: triángulo entre 0 y 0.6, pico en 0.0.
- *Medio*: triángulo entre 0.3 y 1.5, pico en 0.9.
- *Grande*: triángulo entre 1.0 y 3.0, pico en 2.0.

Si `|error| = 1.1`:

- *Pequeño*: 0 (fuera del triángulo).
- *Medio*: `(1.5 - 1.1) / (1.5 - 0.9) ≈ 0.67`.
- *Grande*: `(1.1 - 1.0) / (2.0 - 1.0) = 0.1`.

La mayor pertenencia es *Media*, así que el controlador aplicaría reglas orientadas a giros moderados.

```
Pertenencia
1.0 |      /\        /\
    |     /  \      /  \
0.0 +----+----+----+----+--> |Error|
      0  0.4 0.9  1.5  2.0
    Pequeño  Medio  Grande
```

### Ejemplo 3: Urgencia [Example – Urgency]

La urgencia depende de la distancia al obstáculo:

- *Baja*: triángulo entre 0 y 0.4, pico en 0.0.
- *Media*: triángulo entre 0.2 y 0.8, pico en 0.5.
- *Alta*: triángulo entre 0.6 y 1.0, pico en 0.9.

Si la urgencia calculada es 0.7:

- *Baja*: 0 (demasiado grande).
- *Media*: `(0.8 - 0.7) / (0.8 - 0.5) ≈ 0.33`.
- *Alta*: `(0.7 - 0.6) / (0.9 - 0.6) ≈ 0.33`.

El valor se encuentra cerca de la frontera entre media y alta, por lo que dos reglas podrían activarse simultáneamente (por ejemplo, “reduzca velocidad” y “gire con fuerza”).

```
Pertenencia
1.0 |    /\      /\
    |   /  \    /  \
0.0 +---+--+----+--+--> Urgencia
     0 0.2 0.5 0.8 1.0
    Baja   Media  Alta
```

### Razón para usar triángulos [Why Triangles]

- Son fáciles de calcular y ajustar manualmente.
- Generan transiciones suaves entre estados (evitan cambios bruscos).
- Permiten activar varias reglas a la vez, lo que contribuye a un comportamiento más natural.

Una vez definidas estas funciones, cualquier entrada numérica se transforma en multiplicadores que indican la intensidad con que se dispara cada regla difusa.***

## Paso 4: Construir base de reglas [Rule Base]

Ejemplos:

1. **Regla R1**: Si *Urgencia* es Alta → Velocidad lineal = Muy baja.
2. **Regla R2**: Si *Distancia* es Lejos y *Urgencia* es Baja → Velocidad lineal = Alta.
3. **Regla R3**: Si *Error angular* es Grande → Velocidad angular = Giro fuerte.
4. **Regla R4**: Si *Distancia* es Cerca y *Error* es Pequeño → Velocidad lineal = Media.

Estas reglas se priorizan según el escenario de navegación: primero seguridad, luego progreso hacia el objetivo.

## Paso 5: Definir salidas numéricas [Output Values]

- **Velocidad lineal [Linear Speed]**:
  - Muy baja: 0.15 (solo para escapar o detenerse cerca de obstáculos).
  - Baja: 0.3 (avanzar lentamente en zonas peligrosas).
  - Media: 0.6 (cuando la urgencia es moderada y aún se requiere progreso).
  - Alta: 1.5 (cuando no hay obstáculos relevantes y el objetivo está lejos).
- **Velocidad angular [Angular Speed]**:
  - Giro fuerte: 2.5 rad/s (reacción brusca para evitar colisiones).
  - Giro medio: 1.6 rad/s (correctiva pero sin perder demasiada velocidad lineal).
  - Giro suave: 0.9 rad/s (ajustes pequeños).
  - Sin giro: 0.0 rad/s (alineación perfecta).

## Paso 6: Ajustes iterativos [Iterative Tuning]

1. Correr simulaciones y observar si la tortuga se acerca demasiado a obstáculos.
2. Incrementar la salida angular o reducir la velocidad lineal cuando la urgencia crezca.
3. Ajustar los límites de pertenencia según la respuesta deseada (por ejemplo, adelantar el umbral de *Urgencia Alta* si la tortuga se “encierra”).

Este procedimiento manual ofrece una base sólida antes de automatizar el diseño con herramientas de optimización o aprendizaje.***

