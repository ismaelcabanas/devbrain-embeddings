# src/labs/lab03/run_cli.py
from src.core.embeddings import LocalEmbeddingEngine
from src.core.metrics import cosine_similarity

def main():
    # 1. Inicializar el motor de IA (Descargará el modelo la primera vez)
    engine = LocalEmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    # 2. Nuestra pequeña base de conocimiento (Mock de base de datos)
    documents = [
        "Para levantar la base de datos de desarrollo usa: docker-compose up -d postgres",
        "La receta de la tarta de Santiago lleva almendras, azúcar, huevos y canela.",
        "Asegúrate de configurar la variable JWT_SECRET en tu archivo .env para proteger la API.",
        "El mantenimiento preventivo del coche incluye revisar el aceite y los frenos cada 10.000 km."
    ]
    
    # Generar embeddings de nuestros documentos
    print("\n[CLI] Indexando documentos en memoria...")
    doc_embeddings = engine.get_embeddings(documents)
    
    # 3. Simular una consulta de usuario por consola
    query = "levantar bd con docker"
    print(f"\n[CLI] Nueva consulta de usuario: '{query}'")
    
    # Generar embedding de la consulta
    query_embedding = engine.get_embedding(query)
    
    # 4. Búsqueda semántica mediante escaneo lineal (Fuerza Bruta)
    print("\n[CLI] Calculando similitudes de coseno...")
    for doc, doc_emb in zip(documents, doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_emb)
        print(f"-> Similitud: {similarity:.4f} | Documento: {doc}")

if __name__ == "__main__":
    main()