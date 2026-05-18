# src/labs/lab03/run_cli.py
from src.core.embeddings import LocalEmbeddingEngine
from src.core.metrics import cosine_similarity

def main():
    engine = LocalEmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    documents = [
        "Para levantar la base de datos de desarrollo usa: docker-compose up -d postgres",
        "La receta de la tarta de Santiago lleva almendras, azúcar, huevos y canela."
    ]
    
    print("\n[Lab 03] Indexando documentos en memoria...")
    doc_embeddings = engine.get_embeddings(documents)
    
    # --- NUEVA SECCIÓN: INSPECCIÓN DE VECTORES DE LOS DOCUMENTOS ---
    print("\n" + "="*60)
    print("INSPECCIÓN DE EMBEDDINGS EN MEMORIA (DATO CRUDO)")
    print("="*60)
    for doc, doc_emb in zip(documents, doc_embeddings):
        print(f"\n📄 Documento: '{doc[:30]}...'")
        print(f"   -> Tipo de objeto: {type(doc_emb)}")
        print(f"   -> Tipo de dato interno (dtype): {doc_emb.dtype}")
        print(f"   -> Dimensiones (Shape): {doc_emb.shape}")
        # Mostramos solo los primeros 5 números del vector como muestra
        sample_slice = doc_emb[:5]
        print(f"   -> Muestra del vector (primeros 5 componentes): {sample_slice}")
    print("="*60 + "\n")
    # ---------------------------------------------------------------
    
    query = "levantar bd con docker"
    print(f"[Lab 03] Nueva consulta de usuario: '{query}'")
    query_embedding = engine.get_embedding(query)
    
    print("\n[Lab 03] Calculando similitudes de coseno...")
    for doc, doc_emb in zip(documents, doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_emb)
        print(f"-> Similitud: {similarity:.4f} | Documento: {doc}")

if __name__ == "__main__":
    main()