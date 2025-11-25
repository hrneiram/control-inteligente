# Introducción al Control Difuso [Fuzzy Control Introduction]

El control difuso (Fuzzy Control) es una técnica que usa lógica difusa para manejar sistemas con incertidumbre o comportamiento no lineal. En lugar de valores binarios, opera con grados de pertenencia entre 0 y 1.

## Conceptos clave [Key Concepts]

- **Conjuntos difusos [Fuzzy Sets]**: Ideas borrosas como *Cerca* o *Lejos*. En vez de decir “verdadero/falso”, permiten grados intermedios. Ejemplo: estar a 1.5 metros puede ser 0.6 *Cerca* y 0.4 *Medio*.
- **Funciones de pertenencia [Membership Functions]**: Dibujos matemáticos (triángulos, trapecios…) que transforman un número en qué tanto pertenece a una etiqueta. Son como reglas de traducción.
- **Reglas difusas [Fuzzy Rules]**: Sentencias estilo “SI estoy cerca del obstáculo ENTONCES reduzco la velocidad”. Capturan la intuición del operador humano.
- **Inferencia [Inference]**: Mezcla las reglas que se activan al mismo tiempo. Si varias reglas piden acciones distintas, se combinan ponderadas por su grado de pertenencia.
- **Defuzzificación [Defuzzification]**: Último paso donde se convierte el resultado borroso en un número concreto (por ejemplo, 0.3 m/s). Es lo que finalmente se envía al motor.

## Glosario amigable [Friendly Glossary]

| Término | Explicación sencilla |
|--------|-----------------------|
| Universo de discurso [Universe of Discourse] | Rango de valores posibles para una variable (por ejemplo, distancia entre 0 y 10 metros). |
| Etiqueta lingüística [Linguistic Label] | Palabra que describe un estado (Cerca, Medio, Lejos). |
| Pertenencia [Membership Degree] | Número entre 0 y 1 que indica qué tan bien encaja un valor en una etiqueta. |
| Superposición [Overlap] | Zona donde dos etiquetas se solapan; permite transiciones suaves. |
| Salida crisp [Crisp Output] | Resultado numérico después de defuzzificar, listo para usar en el actuador. |

## ¿Por qué usar control difuso? [Why Fuzzy Control]

- Maneja situaciones imprecisas sin requerir un modelo matemático exacto.
- Permite codificar la experiencia de un operador humano en forma de reglas.
- Proporciona transiciones suaves entre comportamientos (evitando saturaciones).

## Aplicación en el proyecto [Project Application]

1. **Variables de entrada [Input Variables]**:
   - Distancia al objetivo.
   - Error angular respecto al rumbo deseado.
   - Urgencia de evitación de obstáculos.
2. **Reglas [Rules]**:
   - Si la urgencia es alta, reducir drásticamente la velocidad lineal.
   - Si el error angular es grande, aplicar un giro fuerte.
   - Si la distancia es grande y la urgencia es baja, acelerar.
3. **Salida [Output]**:
   - Velocidad lineal (`linear_vel`).
   - Velocidad angular (`angular_vel`).

## Terminología [Terminology]

- *Fuzzy Controller*: controlador difuso.
- *Inference Engine*: motor de inferencia.
- *Rule Base*: base de reglas.
- *Membership Function*: función de pertenencia.
- *Centroid*: método de defuzzificación basado en el centroide.

Si encuentras un término técnico en el documento de cálculo manual, vuelve a este glosario: cada palabra utilizada allí se apoya en estas definiciones.

El uso de estas nociones permite adaptar el movimiento del robot a entornos con obstáculos de forma intuitiva y flexible.***

