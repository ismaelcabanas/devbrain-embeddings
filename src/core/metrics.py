import numpy as np

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores utilizando NumPy.
    
    Args:
        vector_a (np.ndarray): Primer vector (array de floats).
        vector_b (np.ndarray): Segundo vector (array de floats).
        
    Returns:
        float: Valor entre -1.0 y 1.0 (típicamente 0.0 a 1.0 en texto),
               donde 1.0 significa que los vectores son semánticamente idénticos.
    
    Nota: Para una explicación detallada de la matemática detrás de esta métrica
    y por qué se prefiere sobre la distancia Euclídea en procesamiento de texto,
    consulte: /docs/02_vector_spaces_and_cosine_similarity.md
    """
    # 1. Calcular el producto escalar (numerador de la fórmula)
    dot_product = np.dot(vector_a, vector_b)
    
    # 2. Calcular la norma L2 (magnitud) de cada vector (denominador)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # 3. Buenas prácticas de backend: prevenir la división por cero
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    # 4. Calcular y retornar el coseno del ángulo
    return float(dot_product / (norm_a * norm_b))