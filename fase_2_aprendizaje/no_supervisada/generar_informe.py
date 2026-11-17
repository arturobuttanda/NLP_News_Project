"""Genera figuras, informe Markdown y PDF desde las salidas reales del experimento."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image

OUT = HERE/'resultados'


def fmt(value):
    return f'{float(value):.4f}'


def generate():
    exps = pd.read_csv(OUT/'experiments.tsv',sep='\t')
    metrics = pd.read_csv(OUT/'metricas_finales.tsv',sep='\t')
    sizes = pd.read_csv(OUT/'cluster_sizes.tsv',sep='\t')
    words = pd.read_csv(OUT/'top_words.tsv',sep='\t')
    reps = pd.read_csv(OUT/'representative_documents.tsv',sep='\t')
    dev_clusters = pd.read_csv(OUT/'dev_cluster_interpretation.tsv',sep='\t')
    selection = json.loads((OUT/'seleccion_dev.json').read_text())
    stability = json.loads((OUT/'estabilidad_dev.json').read_text())
    config = json.loads((OUT/'configuracion_final.json').read_text())
    notes = json.loads((HERE/'interpretacion_manual.json').read_text())
    cfg = selection['config']
    figures = OUT/'figuras'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax = plt.subplots(figsize=(9,4))
    sizes.set_index('cluster_id')[['train','dev','test']].plot.bar(stacked=True,ax=ax,color=['#1c5679','#44a49c','#f4b65b'])
    ax.set(xlabel='Cluster',ylabel='Número de noticias',title='Tamaños de los grupos del modelo final')
    fig.tight_layout(); fig.savefig(figures/'tamanos_clusters.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for eligible,label,color in [(False,'No admisible','#bfc6cc'),(True,'Admisible','#287b9a')]:
        data=exps[exps.eligible.eq(eligible)]
        ax.scatter(data.index+1,data.silhouette,label=label,color=color,s=22)
    win=exps[exps.selected].iloc[0]
    ax.scatter([win.name+1],[win.silhouette],marker='*',s=220,color='#e79a2d',label='Seleccionado',zorder=5)
    ax.set(xlabel='Experimento',ylabel='Silhouette en dev',title='Selección con silhouette y restricciones de tamaño')
    ax.legend();fig.tight_layout();fig.savefig(figures/'experimentos_dev.png',dpi=160);plt.close(fig)
    pd.DataFrame([{'factor':factor,'value':str(value),'mean_silhouette':g.silhouette.mean(),'admissible':int(g.eligible.sum()),'total':len(g)} for factor in ['normalization','representation','ngram','k'] for value,g in exps[exps.stage.eq('factorial')].groupby(factor)]).to_csv(OUT/'resumen_factores.tsv',sep='\t',index=False)
    # Parejas del factorial que cambian un solo factor, conservando los demás.
    factors=['normalization','representation','ngram','k']
    controlled=[]
    factorial=exps[exps.stage.eq('factorial')]
    for factor in factors:
        others=[f for f in factors if f!=factor]
        for _, group in factorial.groupby(others):
            rows=list(group.to_dict('records'))
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    controlled.append(dict(factor=factor,experiment_a=a['experiment_id'],experiment_b=b['experiment_id'],value_a=a[factor],value_b=b[factor],silhouette_a=a['silhouette'],silhouette_b=b['silhouette'],delta=b['silhouette']-a['silhouette'],eligible_a=a['eligible'],eligible_b=b['eligible']))
    pairs=pd.DataFrame(controlled)
    pairs.to_csv(OUT/'comparaciones_controladas.tsv',sep='\t',index=False)
    lines=[]
    def add(text=''): lines.append(text)
    add('# Aprendizaje no supervisado de noticias con K-Means')
    add('Proyecto Etapa II - Tarea B. Resultados ejecutados el 17 de noviembre de 2026. Elaboración conjunta de todos los integrantes del equipo.')
    add('Autores del trabajo: Ricardo Hernandez Espinoza; Diana Karen Espinoza Diaz; Byron Robles Lozano; Hector Arturo Saenz Buttanda; Diamanda Abigail Zurita Salinas.')
    add('## Objetivo y alcance')
    add('Agrupar noticias por similitud textual e identificar temas o eventos recurrentes. Se implementa únicamente K-Means. Este informe cubre la tarea no supervisada; la clasificación supervisada y su integración en el informe general pertenecen al otro módulo.')
    add('## Corpus y control de fuga de información')
    add('Se reutilizaron las 4,645 noticias y los identificadores del proyecto. Entrada: título + descripción; se verificó coincidencia exacta de textos y categorías con news_raw_corpus_4645.tsv. Se conservaron las particiones existentes: 3,251 train, 697 dev y 697 test (70/15/15, estratificadas originalmente con random_state=0). No se recolectaron noticias nuevas ni se modificó el benchmark de etapa I.')
    add('La columna category no se suministra a vectorizadores, SVD, K-Means ni al selector. Las categorías proceden del feed RSS de origen; solo se usan al final para ARI/NMI y tablas de contingencia externas. Los parámetros aprendidos se ajustan únicamente en train durante la selección. Una vez congelada la configuración se reajusta el pipeline completo en train + dev, incluidos vocabulario, IDF y SVD. Test se usa después para la evidencia final, sin escoger configuraciones a partir de sus resultados.')
    add('## Configuraciones y metodología')
    add('Se reutiliza NormalizadorNoticias del pipeline del equipo. Mínima: texto original con minúsculas y tokenización del vectorizador. Sin stopwords: tokenización y lista inglesa de scikit-learn. Lemas: limpieza de texto/boilerplate y lematización inglesa spaCy en_core_web_sm 3.8.0 con filtrado POS; no es solamente lematización ni equivale exactamente a la lista de stopwords. Estas transformaciones fijas se calculan antes de vectorizar sin aprender estadísticas del corpus.')
    add('Se compararon Binary CountVectorizer, frecuencia CountVectorizer y TF-IDF; rangos (1,1), (1,2) y (1,3). Los dos últimos incluyen unigramas y bigramas, y unigramas, bigramas y trigramas respectivamente. No se exploran las siete combinaciones posibles: el PDF pide probar rangos y combinaciones, no exige el producto completo. Vocabulario: min_df=2, max_df=0.95, max_features=30,000, aprendidos en train. Estos límites reducen términos raros y el costo de cómputo, pero pueden perder eventos poco frecuentes.')
    add('El factorial inicial cruza 3 normalizaciones x 3 representaciones x 3 rangos x k en {4,8,12}: 81 experimentos, SVD=100, n_init=10, init=k-means++, max_iter=300. Después se comparan SVD sin reducción/50/150, k=6/10 y cambios individuales de n_init (1,20), init (random) y max_iter (100,500), manteniendo los demás parámetros del mejor candidato de la etapa. Es una búsqueda escalonada controlada, no exhaustiva. Semilla principal fija: 0.')
    add('Después de SVD (cuando se usa) se aplica normalización L2. K-Means optimiza distancia euclidiana en ese espacio; silhouette se mide en el mismo espacio y con distancia euclidiana sobre los 697 documentos dev completos, sin muestreo. Sin SVD se usa el espacio léxico disperso normalizado L2. Se registran vectores cero: estos se conservan, su asignación puede ser poco informativa y se informa su cantidad. Si solo se ocupa un grupo, silhouette queda sin definir y ese experimento se descarta.')
    add(f'Se ejecutaron {len(exps)} experimentos de selección, más tres ajustes de estabilidad. Regla registrada antes de los experimentos: todos los k grupos deben aparecer en dev, cada uno con al menos 1% de dev y ninguno con más de 65%; entre candidatos admisibles gana el mayor silhouette sin redondear (desempate por primer ID). Son umbrales de diseño, no una garantía de coherencia temática. No se fuerza k=10 ni se usa el número de categorías para fijarlo. La interpretación cualitativa posterior documenta límites del resultado y no reabre la selección con test.')
    add('## Resultados de selección en dev')
    add(f"Ganador: {selection['experiment_id']}; normalización={cfg['normalization']}; representación={cfg['representation']}; n-gramas={cfg['ngram']}; SVD={cfg['svd']}; k={cfg['k']}; n_init={cfg['n_init']}; init={cfg['init']}; max_iter={cfg['max_iter']}; silhouette dev={fmt(selection['silhouette_dev'])}.")
    add('Mejores candidatos admisibles (modelo ajustado solo en train):')
    for r in exps[exps.eligible].sort_values('silhouette',ascending=False).head(6).itertuples():
        add(f'- {r.experiment_id}: {r.normalization}, {r.representation}, ngram={r.ngram}, SVD={r.svd}, k={r.k}; silhouette={fmt(r.silhouette)}; tamaños dev={r.sizes}.')
    highest=exps.sort_values('silhouette',ascending=False).iloc[0]
    add(f"Mayor silhouette sin restricciones: {highest.experiment_id} ({fmt(highest.silhouette)}), admisible={highest.eligible}; tamaños dev={highest.sizes}. En esta corrida el máximo global también es admisible. Las restricciones permiten descartar otras particiones con grupos extremos; no cambiaron el ganador respecto al máximo global.")
    add('![Experimentos en dev](figuras/experimentos_dev.png)')
    add('## Efecto de cambiar un factor')
    add('Las siguientes parejas del factorial mantienen los demás factores constantes. Se muestra la mayor diferencia absoluta por factor para ilustrar sensibilidad, no una mejora universal; todas las parejas están en comparaciones_controladas.tsv. Las medias del archivo resumen_factores.tsv son descriptivas y no sustituyen estas comparaciones.')
    for factor,g in pairs.groupby('factor'):
        r=g.loc[g.delta.abs().idxmax()]
        add(f"- {factor}: {r.experiment_a} ({r.value_a}, {fmt(r.silhouette_a)}) frente a {r.experiment_b} ({r.value_b}, {fmt(r.silhouette_b)}); cambio B-A={float(r.delta):+.4f}; admisibles A/B={r.eligible_a}/{r.eligible_b}.")
    base=exps[exps.stage.eq('factorial') & exps.eligible].sort_values('silhouette',ascending=False).iloc[0]
    add(f"Base para comparar SVD y refinamiento de k: {base.experiment_id}, silhouette={fmt(base.silhouette)}.")
    for r in exps[~exps.stage.eq('factorial')].itertuples():
        add(f'- {r.experiment_id} ({r.stage}): SVD={r.svd}, k={r.k}, n_init={r.n_init}, init={r.init}, max_iter={r.max_iter}; silhouette={fmt(r.silhouette)}; admisible={r.eligible}.')
    add('Cambios que no modifican las asignaciones o el silhouette no justifican afirmar mejoras; aumentar max_iter puede no tener efecto si K-Means ya convergió antes del límite. SVD y L2 modifican la geometría; comparar silhouettes entre espacios distintos debe acompañarse de lectura de palabras y documentos.')
    add('## Estabilidad e interpretabilidad en dev')
    add('Se mantuvieron fijos vectorización, SVD y parámetros seleccionados, cambiando solo la semilla de K-Means a 0, 1 y 2. ARI se calcula entre particiones de los mismos documentos dev; no utiliza categorías reales.')
    for row in stability['pairwise_ari']:
        add(f"- Semillas {row['seed_a']} y {row['seed_b']}: ARI={fmt(row['ari'])}.")
    for row in stability['trials']:
        add(f"- Semilla {row['seed']}: silhouette={fmt(row['silhouette'])}; tamaños={row['sizes']}.")
    add(notes['stability_comment'])
    add(notes['dev_comparison'])
    for row in notes['dev_examples']:
        add(f"- Experimento {row['experiment_id']}, grupo {row['cluster_id']}: {row['comment']}")
    add('Los términos y dos noticias dev por grupo para cada experimento se conservan en dev_cluster_interpretation.tsv, incluidas las particiones rechazadas, para inspeccionar temas genéricos y eventos repetidos.')
    add('## Evaluación del modelo final')
    add('Los IDs de los grupos son arbitrarios y pueden permutarse tras reajustar. Los siguientes resultados corresponden al modelo final entrenado con train + dev; su métrica dev ya no es una estimación independiente. La selección se basó en el resultado anterior, ajustado solo con train.')
    for r in metrics.itertuples():
        add(f'- {r.split}: silhouette={fmt(r.silhouette)}, ARI externo={fmt(r.ari)}, NMI externo={fmt(r.nmi)}, vectores cero={r.zero_vectors}, tamaños={r.sizes}.')
    add('ARI/NMI no son accuracy de clasificación ni se optimizaron: las diez categorías de origen son una referencia temática imperfecta. La tabla de contingencia por partición está en contingencia_train/dev/test.tsv.')
    add('![Tamaños por cluster](figuras/tamanos_clusters.png)')
    add('## Interpretación de los clusters finales')
    add('Se muestran diez términos del centroide en el espacio léxico original. Con SVD, se reconstruye el centroide mediante inverse_transform; son pesos aproximados, no probabilidades de tema. Las dos noticias más cercanas al centroide dentro del grupo se seleccionan entre train + dev. Se añade una noticia lejana como caso de frontera. El PDF muestra extractos; los textos completos quedan en representative_documents.tsv. Los nombres temáticos son una lectura de términos y documentos, no las categorías del feed.')
    for c in sizes.cluster_id:
        size=sizes[sizes.cluster_id.eq(c)].iloc[0]
        note=notes['clusters'][str(c)]
        add(f"### Cluster {c}: {note['title']}")
        add(f'Tamaño: train={size["train"]}, dev={size["dev"]}, test={size["test"]}, total={size.total}.')
        add('Diez términos: '+', '.join(words[words.cluster_id.eq(c)].term)+'.')
        add(note['analysis'])
        for r in reps[reps.cluster_id.eq(c)].itertuples():
            add(f'- {r.role} {r.rank}: {r.document_id} ({r.split}), distancia={fmt(r.distance)}. Extracto: {r.text[:900]} URL: {r.url}')
    add('## Cobertura de eventos recurrentes')
    docs = pd.read_csv(OUT/'labeled_documents.tsv',sep='\t')
    assigned = pd.read_csv(OUT/'clustering_results.tsv',sep='\t').set_index('document_id')
    event_rows=[]
    for case in notes['event_cases']:
        a,b=case['document_a'],case['document_b']
        ca,cb=int(assigned.loc[a,'cluster_id']),int(assigned.loc[b,'cluster_id'])
        assert docs.set_index('document_id').loc[[a,b],'split'].ne('test').all()
        event_rows.append(dict(document_a=a,document_b=b,cluster_a=ca,cluster_b=cb,same_cluster=ca==cb,interpretation=case['interpretation']))
        add(f'- {a} (grupo {ca}) y {b} (grupo {cb}): '+case['interpretation'])
    pd.DataFrame(event_rows).to_csv(OUT/'eventos_recurrentes.tsv',sep='\t',index=False)
    add('## Limitaciones y revisión de casos problemáticos')
    for item in notes['limitations']:
        add('- '+item)
    add('- Los términos incluyen n-gramas superpuestos; no siempre son diez conceptos independientes. Los textos son títulos y descripciones RSS, no artículos completos. La similitud léxica puede agrupar nombres propios o estilos de una fuente en lugar de temas.')
    add('- Los umbrales de tamaño se aplican en dev durante selección; el reajuste final puede cambiar tamaños y coherencia. Las matrices finales se guardan con IDs y vocabulario para rastrear esas diferencias. Los grupos y eventos se interpretan sobre evidencia real, sin exigir que coincidan con las diez categorías supervisadas.')
    add('- La búsqueda escalonada y el límite de 30,000 términos reducen costo, pero no demuestran optimalidad global. La estabilidad de tres semillas explora sensibilidad de inicialización; no equivale a validación temporal ni evaluación humana exhaustiva.')
    add('## Reproducibilidad y entregables')
    add('Segunda revisión: auditoría independiente registrada en auditoria_cumplimiento.json. experiments_completo.tsv consolida los 91 candidatos de selección y los tres ajustes de estabilidad (94), distinguidos por candidate_for_selection. Los parámetros completos del modelo están en parametros_pipeline_final.json. La guía de estudio GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf/Markdown relaciona cada requisito con archivos y código exacto.')
    add('Comandos desde la raíz del proyecto:')
    add('1. Crear entorno: python3 -m venv .venv')
    add('2. Instalar: .venv/bin/python -m pip install -r fase_2_aprendizaje/no_supervisada/requirements-lock.txt')
    add('3. Ejecutar en una copia limpia de esta carpeta: .venv/bin/python -u fase_2_aprendizaje/no_supervisada/experimentos.py')
    add('4. Validar: .venv/bin/python fase_2_aprendizaje/no_supervisada/validar_resultados.py')
    add('5. Consultar: .venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --cluster 0')
    add('6. Regenerar informe desde resultados existentes: .venv/bin/python fase_2_aprendizaje/no_supervisada/generar_informe.py')
    add('El programa rechaza sobrescribir una corrida final existente. Modelo: modelos/kmeans_pipeline.joblib. Versiones exactas: requirements-lock.txt y configuracion_final.json. Archivo de ejecución: ejecucion.log. Fuente y plan: plan_experimental.json, incluido SHA-256 del corpus etiquetado. Los archivos binarios locales están excluidos por el .gitignore general; deben incluirse al compartir la carpeta como entrega.')
    add('Verificación de resultados: '+(OUT/'validacion.txt').read_text().strip())
    add('## Correspondencia con el PDF y pendientes de integración')
    add('- Secciones 4-5: corpus y particiones originales; tres normalizaciones, tres representaciones y varios n-gramas; ajustes aislados en train y pipeline final persistido.')
    add('- Sección 7: K-Means, k/n_init/init/max_iter, SVD, silhouette con espacio y distancia, tamaños, diez términos, dos noticias, grupos mezclados y estabilidad dev; ARI/NMI y contingencia solo externos.')
    add('- Secciones 9-11: bitácora experiments.tsv, configuración elegida en dev, interfaz inspeccionar.py, asignaciones clustering_results.tsv e informe PDF con figuras.')
    add('- Secciones 12-13: esta carpeta completa la evidencia de la rama K-Means. Para entregar la etapa II completa, el equipo debe integrar este PDF y sus tablas con clasificación supervisada, revisar los pendientes de reproducibilidad de ese módulo y consolidar el informe general. No se requiere implementar LDA cuando se elige K-Means.')
    add('Referencia de requisitos: Proyecto_01_Noticias_Etapa_II.pdf entregado por la usuaria; secciones 4, 5, 7, 9, 10, 11, 12 y 13. Las interpretaciones cualitativas se basan en las noticias exportadas en esta corrida.')
    md='\n\n'.join(lines)+'\n'
    (OUT/'INFORME_NO_SUPERVISADO.md').write_text(md,encoding='utf-8')
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=9
    styles['BodyText'].leading=13
    styles['BodyText'].spaceAfter=7
    story=[]
    for line in lines:
        if line.startswith('!['):
            filename=line.split('](')[1].rstrip(')')
            story.append(Image(str(OUT/filename),width=17*cm,height=7.55*cm))
        elif line.startswith('# '): story.append(Paragraph(escape(line[2:]),styles['Title']))
        elif line.startswith('## '): story.append(Paragraph(escape(line[3:]),styles['Heading1']))
        elif line.startswith('### '): story.append(Paragraph(escape(line[4:]),styles['Heading2']))
        else: story.append(Paragraph(escape(line),styles['BodyText']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(1.5*cm,1*cm,'Proyecto NLP - K-Means | Evidencia de ejecución')
        canvas.drawRightString(19.5*cm,1*cm,str(doc.page))
    SimpleDocTemplate(str(OUT/'INFORME_NO_SUPERVISADO.pdf'),pagesize=(21*cm,29.7*cm),rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.5*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Informe Markdown/PDF y figuras generados.')


if __name__=='__main__':
    generate()
