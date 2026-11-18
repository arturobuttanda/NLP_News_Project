"""Create a PDF report from actual saved evaluation results (no fabricated metrics)."""
from pathlib import Path
from html import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

BASE = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE.parent
RESULTS = BASE / "resultados"


def main():
    evidence = RESULTS / "evidence"
    baseline = evidence / "description_unigram_tfidf"
    queries = pd.read_csv(BASE / "data" / "queries.tsv", sep="\t")
    metrics = pd.read_csv(baseline / "evaluation_results.tsv", sep="\t")
    details = pd.read_csv(baseline / "query_results.tsv", sep="\t")
    normalized = pd.read_csv(baseline / "normalized_queries.tsv", sep="\t")
    qrels = pd.read_csv(BASE / "data" / "qrels.tsv", sep="\t")
    normalization = PROJECT_DIR / "2. Normalizacion"
    raw = pd.read_csv(normalization / "data" / "raw" / "news_raw_corpus.tsv", sep="\t").fillna("")
    processed = pd.read_csv(normalization / "resultados" / "processed" / "news_normalized_corpus.tsv", sep="\t").fillna("").set_index("id")
    comparison = pd.read_csv(evidence / "comparison.tsv", sep="\t")
    styles = getSampleStyleSheet()
    story = []
    def para(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 7))
    def table(frame):
        data = [list(frame.columns)] + frame.astype(str).values.tolist()
        t = Table(data, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("FONTSIZE", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("GRID", (0,0), (-1,-1), .25, colors.grey)]))
        story.append(t)
        story.append(Spacer(1, 12))
    para("Recuperacion de noticias - Fase 1", "Title")
    para("Integrantes: PENDIENTES DE COMPLETAR")
    para("Benchmark propuesto: juicios elaborados con asistencia de IA a partir de titulares y descripciones. Requiere revision del equipo antes de entregar. Las metricas se calcularon ejecutando el evaluador del profesor, sin modificarlo.")
    para("Corpus y metodo", "Heading1")
    para(f"{len(raw)} noticias; {raw.source.nunique()} fuentes; {raw.category.nunique()} categorias. Corpus suministrado por el equipo; no se repitio la recoleccion. Fechas de publicacion: {raw.date.min()} a {raw.date.max()}. Se conservaron las noticias en ingles.")
    for field in ["category", "source"]:
        table(raw[field].value_counts().rename_axis(field).reset_index(name="noticias"))
    table(raw.date.str[:10].value_counts().sort_index().rename_axis("fecha").reset_index(name="noticias"))
    para("spaCy en_core_web_sm: limpieza de espacios y caracteres, etiquetado gramatical, filtrado POS y lematizacion. Consultas y corpus usan las mismas reglas. Scikit-learn: frecuencia, binaria y TF-IDF; unigramas, bigramas y combinacion; campos titulo y descripcion. Similitud coseno; desempate estable por orden del corpus. Se conservan diez resultados incluso con score cero para el formato oficial.")
    para("Diseno del benchmark", "Heading1")
    para("15 consultas: cinco faciles, cinco medias y cinco dificiles. Cinco temas (oncologia, exploracion espacial, inflacion, Ucrania y clima), cada uno con tres formulaciones. Cada consulta tiene 12 documentos: relevancia 2 para cobertura directa y 1 para contexto o relacion parcial. Los conjuntos se definieron antes de ejecutar rankings. Los tres niveles comparten el conjunto tematico: esto permite comparar reformulaciones, pero las 15 consultas no son independientes.")
    para("Los juicios no son exhaustivos: otros articulos pertinentes no incluidos se consideran cero por convencion del evaluador. Recall mide cobertura del conjunto juzgado, no de toda la informacion pertinente del corpus. No se debe interpretar el resultado como evaluacion humana definitiva ni como validacion de hechos publicados.")
    for row in queries.itertuples():
        para(f"{row.query_id} | nivel {row.difficulty_level} | {row.query}")
    story.append(PageBreak())
    para("Resultados de referencia: descripcion, unigramas, TF-IDF", "Heading1")
    table(metrics.round(4))
    para("RR es el reciproco del primer relevante en el Top-10; el promedio de RR es MRR. El evaluador usa ganancias 2^relevancia - 1 para nDCG. Con 12 relevantes por consulta, el maximo Recall@10 es 10/12.")
    merged = metrics[metrics.query_id != "MEAN"].merge(queries, on="query_id")
    para("Promedios por dificultad", "Heading2")
    table(merged.groupby("difficulty_level")[["P@10", "R@10", "F1@10", "RR", "nDCG@10"]].mean().round(4).reset_index())
    para("Comparacion de las 18 configuraciones", "Heading1")
    table(comparison.drop(columns="query_id").sort_values("nDCG@10", ascending=False).round(4))
    para("La referencia se eligio por ser una configuracion convencional, antes de comparar. La mejor configuracion de esta tabla es descriptiva de este benchmark; no hay un conjunto de prueba independiente.")
    ordered = merged.sort_values(["nDCG@10", "P@10", "query_id"], ascending=[False, False, True])
    for heading, subset in [("Cinco consultas con mejor desempeno relativo", ordered.head(5)), ("Cinco consultas con peor desempeno relativo", ordered.tail(5))]:
        story.append(PageBreak())
        para(heading, "Heading1")
        for row in subset.itertuples():
            qid = row.query_id
            m = metrics[metrics.query_id == qid].iloc[0]
            para(f"{qid}: {row.query}", "Heading2")
            norm = normalized[normalized.query_id == qid].iloc[0].normalized_query
            para(f"Normalizada: {norm}. P@10={m['P@10']:.4f}; R@10={m['R@10']:.4f}; nDCG@10={m['nDCG@10']:.4f}.")
            hits = details[details.query_id == qid]
            rels = qrels[qrels.query_id == qid]
            missing = rels[~rels.document_id.isin(hits.id)]
            first = hits.iloc[0]
            para(f"Primer resultado: {first.id}, coseno={first.score:.6f}, {first.raw_title}.")
            query_terms = set(str(norm).split())
            first_overlap = sorted(query_terms & set(processed.loc[first.id, "description"].split()))
            relevant_ids = set(rels.document_id)
            para(f"Aciertos juzgados en Top-10: {len(set(hits.id) & relevant_ids)} de 10. Terminos de consulta presentes en la descripcion del primer resultado: {', '.join(first_overlap) or '(ninguno)'}. Relevancia asignada al primer resultado: {int(rels[rels.document_id == first.id].relevance.iloc[0]) if first.id in relevant_ids else 0}.")
            if len(missing):
                docid = missing.iloc[0].document_id
                para(f"Relevante omitido del Top-10: {docid}, {raw[raw.id == docid].iloc[0].title}.")
                overlap = sorted(query_terms & set(processed.loc[docid, "description"].split()))
                para(f"Terminos compartidos con ese documento omitido: {', '.join(overlap) or '(ninguno)'}. La coincidencia de palabras no garantiza entrar al Top-10: TF-IDF tambien pondera rareza y longitud.")
            para("Interpretacion: " + {1: "El vocabulario explicito del tema facilita coincidencias lexicas. Los relevantes omitidos ilustran que un tema amplio puede expresarse con nombres o terminos diferentes y que un conjunto incompleto de juicios tambien limita la precision medida.", 2: "La reformulacion introduce variacion lexica. El modelo solo compara terminos lematizados compartidos; no aprende sinonimos. Los terminos generales pueden competir con articulos ajenos al foco tematico.", 3: "La necesidad se expresa indirectamente y pierde las palabras distintivas del tema. TF-IDF no representa equivalencias semanticas; palabras generales favorecen coincidencias incidentales. Se propone conservar este caso para comparar embeddings en una etapa posterior."}[row.difficulty_level])
    para("Archivos y pendientes", "Heading1")
    para("rankings.tsv contiene query_id, document_id, rank y score. query_results.tsv incluye los textos y URLs; normalized_queries.tsv conserva la consulta procesada; evaluation_results.tsv contiene las metricas por consulta y MEAN. evaluator_output.txt es la salida original del evaluador. comparison.tsv compara configuraciones. manifest.json fija hashes de entradas; requirements-lock.txt registra versiones.")
    para("Antes de entregar: completar integrantes; revisar judgments_review.tsv y los niveles de dificultad; si cambian los juicios, volver a ejecutar benchmark e informe. No se ha subido ningun archivo a la plataforma escolar porque no se proporciono acceso ni destino.")
    SimpleDocTemplate(str(evidence / "informe_resultados.pdf"), title="News Retrieval - Stage 1", rightMargin=36, leftMargin=36).build(story)
    print(evidence / "informe_resultados.pdf")


if __name__ == "__main__":
    main()
