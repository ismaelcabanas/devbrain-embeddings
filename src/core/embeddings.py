import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEmbeddingEngine:
    """
    Wrapper para la gestión y generación de embeddings utilizando modelos 
    locales de Hugging Face a través de SentenceTransformers.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el modelo. La primera vez lo descargará localmente
        y lo guardará en la caché del sistema.
        """
        print(f"[EmbeddingEngine] Cargando modelo local: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingEngine] Modelo listo. Dimensiones del vector: {self.dimension}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera el embedding para una única cadena de texto.
        """
        if not text.strip():
            # Defensa backend: string vacío genera vector nulo
            return np.zeros(self.dimension, dtype=np.float32)
            
        # El modelo devuelve un array de numpy por defecto
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """
        Genera embeddings en lote (batch processing). 
        Optimizado internamente por la librería para procesamiento paralelo.
        """
        if not texts:
            return []
            
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.astype(np.float32) for emb in embeddings]