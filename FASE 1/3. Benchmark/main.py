import os
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR / "codigo"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from information_retrieval import InformationRetrievalEngine, display_search_results
from evaluate_benchmark import evaluate_retrieval

def run_search(query, content="title", feature="unigram", representation="tfidf", top_k=10):
    print("\n" + "=" * 70)
    print("RECUPERACIÓN DE INFORMACIÓN (TOP-10)")
    print("=" * 70)
    engine = InformationRetrievalEngine()
    res = engine.search(
        query=query,
        content=content,
        feature=feature,
        representation=representation,
        top_k=top_k
    )
    display_search_results(res)

def run_benchmark(content="description", feature="both", representation="tfidf", top_k=10):
    print("\n" + "=" * 70)
    print("EVALUACIÓN DE BENCHMARK Y REPORTE DE MÉTRICAS (15 CONSULTAS)")
    print("=" * 70)
    features = ["unigram", "bigram", "both"] if feature == "both" else [feature]
    representations = ["frequency", "binary", "tfidf"] if representation == "all" else [representation]
    comparison_rows = []
    for selected_feature in features:
        for selected_representation in representations:
            results_dir = BASE_DIR / "resultados" / selected_feature / selected_representation
            print(
                f"\n>>> Ejecutando configuración: "
                f"{selected_feature.upper()} + {selected_representation.upper()}"
            )
            eval_df = evaluate_retrieval(
                content=content,
                feature=selected_feature,
                representation=selected_representation,
                top_k=top_k,
                results_dir=results_dir,
            )
            means = eval_df[["P@10", "R@10", "F1@10", "RR", "nDCG@10"]].mean()
            comparison_rows.append({
                "content": content,
                "feature": selected_feature,
                "representation": selected_representation,
                **means.to_dict(),
            })
    comparison_path = BASE_DIR / "resultados" / "comparison.tsv"
    pd.DataFrame(comparison_rows).sort_values(
        ["nDCG@10", "P@10"], ascending=False
    ).to_csv(comparison_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nTabla comparativa guardada en: {comparison_path}")

def run_interactive():
    print("\n" + "=" * 70)
    print("SISTEMA DE RECUPERACIÓN DE NOTICIAS (PROYECTO ETAPA 1 - NLP)")
    print("=" * 70)
    while True:
        print("\nSeleccione la tarea que desea ejecutar:")
        print("  1. Realizar una consulta en texto libre y ver Top-10")
        print("  2. Ejecutar el Benchmark oficial (15 consultas) y calcular métricas")
        print("  3. Salir")
        
        choice = input("\nIngrese una opción [1-3]: ").strip()
        if choice == "1":
            q = input("\nEscriba su consulta en texto libre: ").strip()
            if not q:
                continue
            cnt = input("Campo a comparar [1: title (default) / 2: description]: ").strip()
            cnt = "description" if cnt == "2" else "title"
            
            feat = input("Feature a extraer [1: unigram (default) / 2: bigram / 3: both]: ").strip()
            if feat == "2":
                feat = "bigram"
            elif feat == "3":
                feat = "both"
            else:
                feat = "unigram"
                
            rep = input("Representación [1: tfidf (default) / 2: frequency / 3: binary]: ").strip()
            if rep == "2":
                rep = "frequency"
            elif rep == "3":
                rep = "binary"
            else:
                rep = "tfidf"
                
            run_search(query=q, content=cnt, feature=feat, representation=rep, top_k=10)
        elif choice == "2":
            run_benchmark()
        elif choice == "3" or choice.lower() in ["exit", "salir", "q"]:
            print("\nFinalizando programa. ¡Hasta luego!\n")
            break
        else:
            print("[!] Opción inválida. Ingrese 1, 2 o 3.")

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Recuperación de Noticias - NLP Proyecto Etapa 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py
  python main.py search --query "artificial intelligence regulation" --content title --feature unigram --representation tfidf
  python main.py benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subcomando: search
    search_parser = subparsers.add_parser("search", help="Ejecuta una consulta libre y devuelve el Top-10")
    search_parser.add_argument("--query", type=str, required=True, help="Texto de la consulta")
    search_parser.add_argument("--content", type=str, default="title", choices=["title", "description"], help="Contenido comparativo (default: title)")
    search_parser.add_argument("--feature", type=str, default="unigram", choices=["unigram", "bigram", "both"], help="N-gramas a extraer (default: unigram)")
    search_parser.add_argument(
        "--representation",
        type=str,
        default="tfidf",
        choices=["frequency", "binary", "tfidf"],
        help="Representación vectorial (default: tfidf)",
    )
    search_parser.add_argument("--top_k", type=int, default=10, help="Número de documentos a recuperar (default: 10)")
    
    # Subcomando: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Ejecuta el benchmark de 15 consultas y calcula métricas")
    bench_parser.add_argument("--content", type=str, default="description", choices=["title", "description"], help="Contenido comparativo (default: description)")
    bench_parser.add_argument("--feature", type=str, default="both", choices=["unigram", "bigram", "both"], help="Feature; both ejecuta unigramas y bigramas por separado")
    bench_parser.add_argument(
        "--representation",
        type=str,
        default="tfidf",
        choices=["frequency", "binary", "tfidf", "all"],
        help="Representación; all compara frequency, binary y tfidf",
    )
    bench_parser.add_argument("--top_k", type=int, default=10, help="Top K para evaluación (default: 10)")
    
    # Subcomando: interactive
    subparsers.add_parser("interactive", help="Inicia la interfaz de consola interactiva")

    args = parser.parse_args()
    
    if args.command == "search":
        run_search(
            query=args.query,
            content=args.content,
            feature=args.feature,
            representation=args.representation,
            top_k=args.top_k
        )
    elif args.command == "benchmark":
        run_benchmark(
            content=args.content,
            feature=args.feature,
            representation=args.representation,
            top_k=args.top_k
        )
    else:
        # Por defecto al ejecutar `python main.py` sin argumentos, iniciar el menú interactivo directamente
        run_interactive()

if __name__ == "__main__":
    main()
