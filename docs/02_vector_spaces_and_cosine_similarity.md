# Lab Note 02: Geometría del Espacio Vectorial y Similitud del Coseno

## 1. El Problema de Negocio / Backend
En sistemas de búsqueda tradicionales (como SQL `LIKE` o Full-Text Search), dependemos de la coincidencia exacta de caracteres. Si el usuario busca "base de datos" y nuestro documento dice "Postgres", no hay correspondencia (*match*). Necesitamos comparar por significado (semántica), no por sintaxis.

## 2. La Intuición Teórica
Cuando pasamos texto por un modelo de embedding, obtenemos un vector (una lista de números de N dimensiones). En este espacio:
- La **dirección** del vector representa el **significado semántico**.
- La **longitud (magnitud)** del vector suele representar la **insistencia o repetición** de palabras.

No podemos usar la Distancia Euclídea ($L2$) porque si un documento es muy largo, su vector será enorme debido a la acumulación de términos y se alejará geométricamente de un documento corto, aunque hablen exactamente de lo mismo. Por eso usamos la **Similitud del Coseno**, que ignora la longitud del texto y solo mide el ángulo entre vectores.

### 📍 Ejemplo Concreto: Componentes y Dirección Vectorial
Para entender cómo un array de floats se convierte en una "dirección semántica", imaginemos un espacio simplificado de **2 dimensiones** (2D), donde cada eje representa un concepto puro:
* **Eje X:** Nivel de relación con "Desarrollo de Software / Tecnología".
* **Eje Y:** Nivel de relación con "Gastronomía / Cocina".

Cuando pasamos tres frases de ejemplo por nuestro sistema, este les asigna coordenadas (valores entre 0.0 y 1.0) formando los siguientes vectores:

1. **Frase A:** *"Cómo desplgar un contenedor de Docker"* $$\vec{v}_A = [0.90, 0.05]$$
   *(Apunta casi por completo hacia el eje X tecnológico, apenas se desvía hacia arriba).*

2. **Frase B:** *"Receta tradicional para hornear pan casero"* $$\vec{v}_B = [0.00, 0.95]$$
   *(Apunta verticalmente hacia el eje Y de cocina, cero relación con software).*

3. **Frase C:** *"Script en Python para automatizar un temporizador de cocina"* $$\vec{v}_C = [0.75, 0.60]$$
   *(Tiene componentes de ambos mundos: es código fuente, pero sirve para la cocina. Apunta en una dirección diagonal).*



La **dirección** es la relación o proporción que existe entre los números del array. Si trazas una línea desde el origen $(0,0)$ hasta el punto del vector, verás que cada uno forma un ángulo respecto al eje X:
* El vector de Docker ($\vec{v}_A$) tiene un ángulo muy cerrado (cercano a 0°).
* El vector del script híbrido ($\vec{v}_C$) tiene un ángulo intermedio (unos 38°).

Al aplicar la Similitud del Coseno entre la Frase A (Docker) y la Frase C (Script de cocina), la fórmula calculará el coseno del ángulo que se forma *entre ellas dos*. Como comparten el "vecindario" de la programación, el ángulo es estrecho y el coseno será alto (ej: `0.78`), indicándole a nuestro backend que existe una relación semántica que la búsqueda tradicional por palabras clave jamás habría detectado.

## 3. El Filtro Matemático
La fórmula del coseno normaliza los vectores dividiendo su producto escalar por el producto de sus normas (magnitudes):

$$\text{Similitud}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **Resultado = 1.0**: Los vectores apuntan exactamente en la misma dirección (Mismo significado).
- **Resultado = 0.0**: Los vectores son ortogonales / perpendiculares (Sin relación semántica).
- **Resultado = -1.0**: Los vectores apuntan en direcciones opuestas (Significados opuestos; raro en texto ya que los valores suelen ser positivos).

## 4. Implementación Backend (Gotchas & Optimización)
- **Evitar bucles `for` nativos:** Iterar sobre arrays de 1536 dimensiones en Python puro destruye la latencia del endpoint. Se utiliza `np.dot` y `np.linalg.norm` porque NumPy corre sobre subrutinas vectorizadas en C (BLAS/LAPACK) que aprovechan la arquitectura de la CPU (instrucciones SIMD).
- **Defensa ante Edge Cases (División por Cero):** Si un texto de entrada está vacío o el modelo genera por error un vector nulo (todo ceros), su norma será 0. Esto provocaría un fallo crítico de tipo `ZeroDivisionError` en producción. La función en `src/core/metrics.py` intercepta esto activamente devolviendo una similitud de `0.0`.