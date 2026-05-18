# src/labs/lab02/test_cosine_similarity.py
import numpy as np
from src.core.metrics import cosine_similarity

# Simulemos vectores de 3 dimensiones: [Tecnología, Cocina, Deportes]
# Un texto de Docker tendrá mucha "Tecnología"
vector_docker = np.array([0.9, 0.0, 0.1])

# Un texto de Kubernetes tendrá una dirección muy similar
vector_k8s = np.array([0.8, 0.0, 0.2])

# Un texto sobre hacer una pizza apuntará hacia "Cocina"
vector_pizza = np.array([0.0, 0.95, 0.0])

# Calcular similitudes
sim_tech = cosine_similarity(vector_docker, vector_k8s)
sim_cook = cosine_similarity(vector_docker, vector_pizza)

print(f"Similitud entre Docker y Kubernetes: {sim_tech:.4f}") # Debería ser muy alta (~0.99)
print(f"Similitud entre Docker y una Pizza: {sim_cook:.4f}")  # Debería ser cercana a 0.0