import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Importar el motor de recuperación de información
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from information_retrieval import InformationRetrievalEngine
from evaluate_ir_metrics import evaluate, load_files, validate

BENCHMARK_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "resultados"
QUERIES_PATH = BENCHMARK_DIR / "queries.tsv"
QRELS_PATH = BENCHMARK_DIR / "qrels.tsv"

def evaluate_retrieval(
    content="description",
    feature="both",
    representation="tfidf",
    top_k=10,
    results_dir=RESULTS_DIR,
):
    print("=" * 80)
    print("9. BENCHMARK AND EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Cargando consultas desde: {QUERIES_PATH}")
    print(f"Cargando juicios de relevancia (qrels) desde: {QRELS_PATH}")
    
    queries_df = pd.read_csv(QUERIES_PATH, sep='\t', encoding='utf-8')
    engine = InformationRetrievalEngine()
    
    # 1. Generar rankings para las 15 consultas
    print(f"\n[1/3] Generando predicciones Top-{top_k} para las {len(queries_df)} consultas...")
    print(f"      Configuración del Sistema: Contenido={content.upper()} | Feature={feature.upper()} | Modelo={representation.upper()}")
    
    rankings_records = []
    for _, qrow in queries_df.iterrows():
        qid = str(qrow['query_id'])
        qtext = str(qrow['query'])
        
        # Ejecutar búsqueda
        res = engine.search(
            query=qtext,
            content=content,
            feature=feature,
            representation=representation,
            top_k=top_k
        )
        
        # Extraer Top 10 predichos y asignar juicios
        retrieved_docs = res['results']
        for r in retrieved_docs:
            rank = r['rank']
            did = str(r['id'])
            score = r['score']
            rankings_records.append({
                "query_id": qid,
                "document_id": did,
                "rank": rank,
                "score": round(score, 4)
            })

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = results_dir / "rankings.tsv"
    eval_results_path = results_dir / "evaluation_results.tsv"

    # 2. Guardar rankings.tsv
    print("\n[2/3] Guardando predicciones de ranking en: ", rankings_path)
    rankings_df = pd.DataFrame(rankings_records)
    rankings_df.to_csv(rankings_path, sep='\t', index=False, encoding='utf-8')
    print(f"      Archivo guardado exitosamente con {len(rankings_df)} filas (15 queries x 10 ranks).")

    # 3. Generar reporte cuantitativo con el evaluador oficial del proyecto
    print("\n[3/3] Calculando métricas agregadas y generando reporte final...")
    rankings_df = pd.DataFrame(rankings_records)
    official_rankings, official_qrels = load_files(rankings_path, QRELS_PATH)
    validate(official_rankings, official_qrels)
    eval_df, mean = evaluate(official_rankings, official_qrels)
    eval_df.to_csv(eval_results_path, sep='\t', index=False, encoding='utf-8', float_format='%.6f')
    display_df = eval_df.merge(
        queries_df[["query_id", "difficulty_level", "query"]],
        on="query_id",
        how="left",
    )
    
    # Calcular promedios globales y por nivel de dificultad
    mean_p10 = mean['P@10']
    mean_r10 = mean['R@10']
    mean_f1 = mean['F1@10']
    mrr = mean['RR']
    mean_ndcg = mean['nDCG@10']
    print(f"      Tabla de evaluación guardada en: {eval_results_path}")
    
    # Despliegue de Resultados en Pantalla
    print("\n" + "=" * 95)
    print("EVALUATION RESULTS PER QUERY (BENCHMARK 15 QUERIES)")
    print("=" * 95)
    header = f"{'QUERY_ID':<8} | {'DIFF':<4} | {'P@10':<7} | {'R@10':<7} | {'F1@10':<7} | {'RR':<7} | {'nDCG@10':<8} | {'QUERY TEXT'}"
    print(header)
    print("-" * 95)
    for _, r in display_df.iterrows():
        print(f"{r['query_id']:<8} | {r['difficulty_level']:<4} | {r['P@10']:<7.4f} | {r['R@10']:<7.4f} | {r['F1@10']:<7.4f} | {r['RR']:<7.4f} | {r['nDCG@10']:<8.4f} | {r['query'][:40]}")
    
    print("=" * 95)
    print("GLOBAL BENCHMARK AVERAGES (SUMMARY METRICS)")
    print("=" * 95)
    print(f"Mean Precision@10 (P@10):  {mean_p10:.4f}")
    print(f"Mean Recall@10 (R@10):     {mean_r10:.4f}")
    print(f"Mean F1@10:                {mean_f1:.4f}")
    print(f"Mean Reciprocal Rank (MRR):{mrr:.4f}")
    print(f"Mean nDCG@10:              {mean_ndcg:.4f}")
    print("=" * 95)

    # Desglose por Dificultad
    print("\n" + "=" * 65)
    print("DESGLOSE DE RENDIMIENTO POR NIVEL DE DIFICULTAD")
    print("=" * 65)
    diff_summary = display_df.groupby('difficulty_level')[['P@10', 'R@10', 'F1@10', 'RR', 'nDCG@10']].mean().reset_index()
    diff_labels = {1: "Level 1: Easy (Strong lexical overlap)", 
                   2: "Level 2: Medium (Lexical variation)", 
                   3: "Level 3: Hard (Semantic / Limited overlap)"}
    diff_summary['difficulty_name'] = diff_summary['difficulty_level'].map(diff_labels)
    for _, drow in diff_summary.iterrows():
        print(f"--> {drow['difficulty_name']}")
        print(f"    P@10: {drow['P@10']:.4f} | R@10: {drow['R@10']:.4f} | F1: {drow['F1@10']:.4f} | MRR: {drow['RR']:.4f} | nDCG@10: {drow['nDCG@10']:.4f}")
    print("=" * 65)

    return eval_df

if __name__ == "__main__":
    evaluate_retrieval()
