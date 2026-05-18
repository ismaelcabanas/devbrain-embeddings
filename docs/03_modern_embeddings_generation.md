# Lab Note 03: Generación de Embeddings con Transformers

## 1. El Problema de Negocio / Backend
Integrar modelos de IA en producción requiere decidir entre dos arquitecturas de infraestructura:
- **APIs de pago por uso (SaaS):** OpenAI, Cohere. Ventaja: Cero mantenimiento de infraestructura, dimensiones masivas (1536+). Desventaja: Latencia de red, coste variable por token, dependes de un tercero y problemas de privacidad de datos.
- **Modelos Locales (Self-Hosted):** SentenceTransformers (Hugging Face). Ventaja: Gratis, datos 100% privados, latencia de red cero, control total. Desventaja: Consumo de CPU/GPU propio y dimensiones típicamente menores (384 - 768).

## 2. La Intuición Teórica: El Pipeline del Transformer
¿Cómo genera un Transformer un embedding?
Cuando usas una librería como sentence-transformers o una API como la de OpenAI, el texto no se convierte en un vector por arte de magia. Pasa por un pipeline de tres etapas críticas:
1. **Tokenización:** El texto se segmenta en unidades mínimas (tokens) con identificadores numéricos. El modelo no entiende palabras completas ni letras sueltas. Rompe el texto en tokens (subpalabras). Por ejemplo, "desplegando" podría convertirse en ["despleg", "ando"]. Cada token tiene asignado un ID numérico inmutable.
2. **Paso por el Transformer**(Capas de Atención): Los vectores iniciales de los tokens se procesan en paralelo. Mediante operaciones matriciales, el significado de cada token se "contamina" positivamente con el significado de los tokens vecinos (contextualización). Estos IDs de tokens entran al modelo. A través de múltiples capas de redes neuronales y mecanismos de autoatención (**Self-Attention**), los tokens hablan entre sí. El modelo calcula cómo influye el contexto de la frase en cada token individual. Si la frase es "banco de dinero", el token "banco" se inclina hacia conceptos financieros gracias a la atención recibida de "dinero".
3. **Pooling (Mean Pooling):** Convierte la matriz de salida de tokens ($Tokens \times Dimensiones$) en un único vector de frase ($1 \times Dimensiones$) calculando el promedio geométrico. Al salir de la última capa del Transformer, tenemos un vector por cada token. Pero nosotros queremos un único vector para toda la frase. El Pooling es la estrategia matemática para colapsar esos vectores de tokens en uno solo. La más común es el Mean Pooling (calcular la media aritmética de todos los vectores de los tokens).

## 3. Decisiones de Diseño Backend (Gotchas de Producción)
- **Manejo de Strings Vacíos:** Pasar un string con espacios en blanco a un transformer puede disparar comportamientos inesperados o errores en las capas de pooling. El wrapper intercepta esto devolviendo un vector de ceros (`np.zeros`) seguro.
- **Eficiencia en Lote (Batching):** Si tienes que indexar 1000 documentos, **nunca** llames a `get_embedding()` en un bucle `for` individual. Utiliza `get_embeddings()`. La librería procesa el lote aprovechando el paralelismo de la CPU/GPU, reduciendo drásticamente el tiempo de cómputo.
- **Tipo de Dato (`float32` vs `float64`):** Forzamos el casteo a `np.float32`. Los embeddings no necesitan la precisión extrema de un `float64` de 64 bits. Usar de 32 bits reduce el consumo de memoria RAM y el tamaño de almacenamiento en disco a la mitad.