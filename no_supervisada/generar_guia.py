"""Guía trazable: fragmentos copiados automáticamente de los archivos reales."""
import ast
import json
import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, Spacer, PageBreak
from material_apoyo_estudio import GLOSSARY, FILES, OLD_NEW, QUESTIONS

HERE=Path(__file__).resolve().parent
OUT=HERE/'resultados'


def lines(rel):
    return (HERE/rel).read_text().splitlines()


def span(rel,start,end=None):
    rows=lines(rel)
    first=next(i for i,s in enumerate(rows) if start in s)
    last=next(i for i in range(first,len(rows)) if end in rows[i]) if end else first
    return rel,first+1,last+1,'\n'.join(rows[first:last+1])


def function(rel,name):
    rows=lines(rel)
    tree=ast.parse('\n'.join(rows))
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return rel,node.lineno,node.end_lineno,'\n'.join(rows[node.lineno-1:node.end_lineno])


def consolidate_log():
    """Reúne los 91 candidatos y los tres ajustes de estabilidad sin reabrir selección."""
    import pandas as pd
    exp=pd.read_csv(OUT/'experiments.tsv',sep='\t')
    exp['candidate_for_selection']=True
    stability=json.loads((OUT/'estabilidad_dev.json').read_text())
    cfg=json.loads((OUT/'seleccion_dev.json').read_text())['config']
    cfg={**cfg,'ngram':str(tuple(cfg['ngram']))}
    additional=[]
    for row in stability['trials']:
        q={key:value for key,value in row.items() if key!='seed'}
        additional.append(dict(experiment_id=f"STAB{row['seed']:03d}",task='clustering',method='KMeans',stage='stability_dev',**cfg,**q,random_state=row['seed'],selected=False,candidate_for_selection=False,eligible=q['occupied_clusters']==cfg['k'] and q['min_share']>=.01 and q['max_share']<=.65))
    full=pd.concat([exp,pd.DataFrame(additional)],ignore_index=True)
    assert len(full)==94 and full.experiment_id.is_unique
    full.to_csv(OUT/'experiments_completo.tsv',sep='\t',index=False)


def main():
    consolidate_log()
    audit=json.loads((OUT/'auditoria_cumplimiento.json').read_text())
    sections=[]
    def item(title,pdf,explain,outputs,*blocks):
        sections.append(dict(title=title,pdf=pdf,explain=explain,outputs=outputs,blocks=blocks))
    item('1. Reutilizar el corpus, IDs y título + descripción','4 y 11',
         'Se leen las noticias originales y se comprueba que cada texto sea exactamente título + espacio + descripción. Las etiquetas se conservan como metadatos externos. Se comprobó además que los 2157 registros de etapa I permanecen idénticos en el corpus ampliado. No se hace scraping nuevo.',
         ['datos/news_raw_corpus_2157.tsv','datos/news_raw_corpus_4645.tsv','datos/labeled_documents.tsv','resultados/labeled_documents.tsv'],
         function('experimentos.py','load_documents'),span('auditar_cumplimiento.py','    original=pd.read_csv',"    check('IDs y artículos"))
    item('2. Particiones train/dev/test reproducibles','4',
         'El entrenamiento reutiliza los splits existentes: 3251/697/697. La auditoría reproduce de forma independiente el doble train_test_split, con shuffle=True, semilla 0 y estratificación, y comprueba que coincida noticia por noticia. La estratificación original usa categorías para repartir, no para entrenar K-Means.',
         ['resultados/distribucion_corpus.tsv','resultados/auditoria_cumplimiento.json'],
         span('experimentos.py',"    train = docs[docs.split == 'train']","    dev = docs[docs.split == 'dev']"),
         span('auditar_cumplimiento.py','    train,rest=train_test_split',"    check('Todas las clases"))
    item('3. Excluir las categorías del entrenamiento y aislar dev','4, 5 y 7',
         'Las matrices se aprenden con textos train; dev solo se transforma. fit(Xtr) recibe características, sin category ni un vector y. Las categorías se usan únicamente al calcular métricas externas y contingencia después de congelar la configuración.',
         ['resultados/experiments.tsv','resultados/seleccion_dev.json'],
         span('experimentos.py',"            tr, dv = normalized[cfg['normalization']]",'            Xdv = fp.transform(dv)'),
         span('experimentos.py','        fp, Xtr, Xdv = caches[featurekey]',"        q = quality(Xdv, labels, cfg['k'])"))
    item('4. Tres normalizaciones, incluida una mínima','5 y 7',
         'minima deja el texto a la tokenización y minúsculas del vectorizador. sin_stopwords tokeniza y filtra la lista inglesa. lemas limpia boilerplate, aplica spaCy en inglés y filtra POS. Esta última no equivale a aplicar exactamente la misma lista de stopwords. El archivo de soporte es una copia del normalizador del equipo.',
         ['soporte/normalizacion.py','resultados/experiments.tsv','resultados/corpus_normalizado_final.tsv'],
         span('experimentos.py',"NORMS =",'GRAMS ='),function('soporte/normalizacion.py','transform'),
         function('soporte/normalizacion.py','lematizar'))
    item('5. Binaria, frecuencia, TF-IDF y rangos de n-gramas','5 y 7',
         'Se ejecutan tres ponderaciones. CountVectorizer(binary=True) marca presencia; binary=False cuenta ocurrencias. TfidfVectorizer pondera por TF-IDF. (1,1) usa unigramas; (1,2) uni+bi; (1,3) uni+bi+tri. El PDF no obliga a probar trigramas solos ni las siete combinaciones del pipeline original. min_df=2, max_df=.95 y max_features=30000 son fijos para comparar; se aprende vocabulario solo en train durante selección.',
         ['resultados/experiments.tsv','resultados/vocabulario.tsv','resultados/parametros_pipeline_final.json'],
         function('experimentos.py','feature_pipeline'),span('experimentos.py',"REPS =",'GRAMS ='))
    item('6. SVD y normalización de longitud L2','5 y 7',
         'Se comparan sin SVD y 50, 100, 150 dimensiones. L2 escala cada vector a longitud unitaria; no es la normalización lingüística del texto. K-Means y silhouette operan en el mismo espacio euclidiano: léxico+L2 o SVD+L2.',
         ['resultados/experiments.tsv','resultados/metricas_finales.tsv','modelos/kmeans_pipeline.joblib'],
         span('experimentos.py',"    if cfg['svd']:",'    return Pipeline(steps)'),
         span('experimentos.py','    for svd in [None,50,150]:',"        trial({**base,'svd':svd},'svd')"))
    item('7. Usar únicamente K-Means y variar hiperparámetros','7',
         'El único algoritmo de agrupamiento es KMeans. Se prueban k=4/6/8/10/12, n_init=1/10/20, init aleatoria y k-means++, max_iter=100/300/500. La búsqueda principal fija random_state=0. No se impone k=10 por haber diez categorías. Se elige una sola rama: LDA no corresponde a esta entrega.',
         ['resultados/experiments.tsv','resultados/plan_experimental.json'],
         span('experimentos.py','SEED ='),function('experimentos.py','kmeans'),
         span('experimentos.py','    for mode,rep,ngram,k in itertools.product','    winner = best()'))
    item('8. Seleccionar en dev y explicar el criterio','7 y 9',
         'Se exige ocupar los k grupos en dev, al menos 1% de noticias por grupo y máximo 65% en uno solo. Entre candidatos admisibles gana el mayor silhouette sin redondear; el primer ID resuelve empates. Son umbrales de diseño, no garantía de temas puros. La interpretación no se utiliza para ajustar usando test.',
         ['resultados/seleccion_dev.json','resultados/experiments.tsv'],
         span('experimentos.py','MIN_SHARE =','MAX_SHARE ='),
         span('experimentos.py','        eligible ='),function('experimentos.py','best'),
         span('experimentos.py','    winner = best()',"    write_json(OUT/'seleccion_dev.json'"))
    item('9. Silhouette con espacio y distancia explícitos','7',
         'Se usa distancia euclidiana y todos los documentos de dev, sin muestreo. Un resultado más alto describe separación geométrica, no accuracy ni coherencia semántica garantizada. Si solo se ocupa un grupo, la métrica no se calcula y el candidato queda fuera.',
         ['resultados/experiments.tsv','resultados/metricas_finales.tsv'],
         function('experimentos.py','quality'))
    item('10. Comparar estabilidad en dev','7',
         'Se mantienen configuración y SVD, y se ajusta K-Means con semillas 0,1,2. ARI entre pares compara particiones de los mismos documentos, sin categorías gold. No se elige una semilla nueva por su mejor puntuación. Los tres ajustes de estabilidad están en JSON y se incluyen en la bitácora consolidada de 94 ajustes.',
         ['resultados/estabilidad_dev.json','resultados/experiments_completo.tsv'],
         span('experimentos.py','    assignments = {}',"    write_json(OUT/'estabilidad_dev.json'"))
    item('11. Congelar y reajustar el pipeline completo','4, 5, 9; flujo solicitado',
         'Tras decidir en dev, se reajustan normalizador, vectorizador, SVD, L2 y K-Means con 3948 textos train+dev. Test no participa en fit. La sección 6 exige explícitamente este refit para clasificación; en K-Means se hace además conforme al flujo pedido en esta conversación.',
         ['modelos/kmeans_pipeline.joblib','resultados/configuracion_final.json','resultados/parametros_pipeline_final.json'],
         span('experimentos.py',"    train_dev = docs[docs.split.isin", "    write_json(OUT/'configuracion_final.json'"))
    item('12. Asignar documentos reservados con predict','7 y 11',
         'Se aplican únicamente transformaciones del modelo ya ajustado y predict. Se conservan IDs y split al guardar los clusters. La auditoría recalcula la misma evaluación guardada para comprobar consistencia: no vuelve a entrenar K-Means ni elige parámetros con test.',
         ['resultados/clustering_results.tsv','resultados/metricas_finales.tsv'],
         span('experimentos.py',"    normalized_all = final['normalizacion']", "    results.to_csv(OUT/'clustering_results.tsv'"))
    item('13. Reportar tamaño de cada grupo por partición','7 y 11',
         'La tabla separa train, dev y test y añade total. Suma 4645 noticias. Así no se confunde el tamaño solo de train con el del corpus completo.',
         ['resultados/cluster_sizes.tsv','resultados/figuras/tamanos_clusters.png'],
         span('experimentos.py','    counts = pd.crosstab',"    counts.to_csv(OUT/'cluster_sizes.tsv'"))
    item('14. Diez palabras representativas en el espacio original','7 y 11',
         'Con SVD, inverse_transform reconstruye el centroide en las columnas del vocabulario original. Se ordenan los pesos y se toman diez términos; con n-gramas un término puede tener varias palabras. No son probabilidades de LDA ni nombres de componentes SVD. Hay 120 filas: diez por doce grupos.',
         ['resultados/top_words.tsv','resultados/vocabulario.tsv'],
         function('experimentos.py','lexical_summary'),
         span('experimentos.py',"        for rank, word in enumerate(words[c],1):", "            top_rows.append("))
    item('15. Dos noticias representativas y un caso de frontera','7 y 11',
         'Se toman las dos noticias train+dev más cercanas a su propio centroide; se exportan texto completo, URL, distancia, ID y split. Una tercera noticia, la más lejana del grupo, sirve para analizar fronteras. No se usa test para escoger estas noticias de interpretación.',
         ['resultados/representative_documents.tsv'],
         span('experimentos.py',"        ids = np.flatnonzero((labels==c)","                rep_rows.append("),
         span('experimentos.py',"    pd.DataFrame(rep_rows).to_csv"))
    item('16. Interpretación, temas, grupos mezclados/pequeños/grandes','7 y 12',
         'La interpretación textual está separada de las asignaciones numéricas: cada grupo tiene un título propuesto y un análisis basado en sus términos y noticias. También se examinan experimentos rechazados: KM031 tiene un grupo de 467/697 dev; KM082 uno de cuatro. Los grupos finales por formato o vocabulario temporal se reconocen como limitaciones, no como temas puros.',
         ['interpretacion_manual.json','resultados/INFORME_NO_SUPERVISADO.md','resultados/INFORME_NO_SUPERVISADO.pdf'],
         span('generar_informe.py',"    for c in sizes.cluster_id:","            add(f'- {r.role}"))
    item('17. Comprobar agrupamiento de eventos recurrentes','7',
         'Se comparan pares de IDs sobre el mismo evento y se verifica si su cluster coincide. Ejemplo favorable: N2256 y N2295, sobre los artículos contra Hegseth, quedan en el grupo 2. El video N2250 del mismo suceso queda en el 5: un fallo relacionado con formato. Los ejemplos usan train/dev.',
         ['resultados/eventos_recurrentes.tsv','interpretacion_manual.json'],
         span('generar_informe.py',"    for case in notes['event_cases']:","    pd.DataFrame(event_rows).to_csv"))
    item('18. ARI/NMI y contingencia solo externos','4 y 7',
         'Se comparan los grupos finales con las categorías de origen únicamente para interpretación. Estas métricas no intervienen en best(). ARI externo con category es distinto del ARI entre semillas usado en estabilidad.',
         ['resultados/metricas_finales.tsv','resultados/contingencia_train.tsv','resultados/contingencia_dev.tsv','resultados/contingencia_test.tsv'],
         span('experimentos.py','    metrics = []',"    pd.DataFrame(metrics).to_csv"))
    item('19. Bitácora completa y comparaciones de un factor','9',
         'experiments.tsv conserva 91 candidatos de selección; experiments_completo.tsv reúne esos 91 más los tres ajustes de estabilidad (94) sin convertir las semillas adicionales en candidatos. comparaciones_controladas.tsv contiene pares del factorial que mantienen todos los otros factores. El informe explica mejoras, empeoramientos y resultados iguales.',
         ['resultados/experiments.tsv','resultados/experiments_completo.tsv','resultados/comparaciones_controladas.tsv','resultados/resumen_factores.tsv'],
         span('experimentos.py',"        row = dict(experiment_id=exp_id", "        pd.DataFrame(interpretations).to_csv"),
         span('generar_informe.py',"    factors=['normalization'", "    pairs.to_csv("),function('generar_guia.py','consolidate_log'))
    item('20. Interfaz para consultar métricas, grupos y nuevas noticias','10',
         'Sin argumentos muestra el resumen y tamaños. --cluster N muestra términos y noticias. --texto aplica el modelo guardado a un título+descripción nuevo en inglés. Estos comandos no entrenan.',
         ['inspeccionar.py','README.md'],
         span('inspeccionar.py',"parser = argparse.ArgumentParser", "    print('Cluster:'"),
         span('inspeccionar.py','elif args.cluster is not None:',"    print(pd.read_csv(HERE/'resultados/cluster_sizes.tsv'"))
    item('21. Versiones, semillas, datos y artefactos reproducibles','3, 9 y 11',
         'Se guardan versiones exactas, modelo spaCy, huella del corpus, parámetros finales, vocabulario y matrices léxicas por partición. El orden de filas de cada NPZ corresponde al TSV filtrado por split. El ZIP incorpora los binarios ignorados por Git.',
         ['requirements-lock.txt','resultados/configuracion_final.json','resultados/parametros_pipeline_final.json','resultados/manifest.json','modelos/X_words_train.npz','modelos/X_words_dev.npz','modelos/X_words_test.npz','ejecucion.log'],
         span('experimentos.py',"    write_json(OUT/'configuracion_final.json'"),
         span('experimentos.py',"    for split in ['train','dev','test']:", "        q = quality(X[mask],labels[mask],cfg['k'])"),
         span('experimentos.py',"        sparse.save_npz("))
    item('22. Informe PDF con figuras, nombres y limitaciones','11 y 12',
         'El informe analítico documenta resultados y limitaciones. La guía explica cada paso para estudiarlo. Todo el trabajo se reconoce como elaboración conjunta de los integrantes del equipo, conforme a la aclaración de la usuaria.',
         ['resultados/INFORME_NO_SUPERVISADO.pdf','resultados/figuras/experimentos_dev.png','resultados/figuras/tamanos_clusters.png'],
         span('generar_informe.py',"    md='", "    (OUT/'INFORME_NO_SUPERVISADO.md').write_text"),
         span('generar_informe.py',"    SimpleDocTemplate("))
    item('23. Auditoría ejecutada y protección de la corrida final','4, 9, 11 y 13',
         f'validar_resultados.py comprueba integridad general. auditar_cumplimiento.py agregó {audit["total_ok"]} comprobaciones independientes de datos, particiones, parámetros reales, matrices completas, asignaciones, métricas, vocabulario y representantes. No se reentrenó K-Means ni se cambió el ganador. La protección impide sobrescribir automáticamente la corrida final.',
         ['validar_resultados.py','auditar_cumplimiento.py','resultados/validacion.txt','resultados/auditoria_cumplimiento.json'],
         span('experimentos.py',"    if (OUT / 'configuracion_final.json').exists():",'        raise SystemExit('),
         span('auditar_cumplimiento.py',"        check('Modelo y asignaciones", "        check('Silhouette guardado"))
    study=json.loads((HERE/'contenido_estudio.json').read_text())
    md=[]; story=[]
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=10;styles['BodyText'].leading=14
    styles['BodyText'].spaceAfter=5
    code_style=ParagraphStyle('CodeExact',fontName='Courier',fontSize=7.6,leading=9.5,backColor=colors.HexColor('#f1f4f6'),spaceAfter=8)
    def prose(text,style='BodyText'):
        md.append(text+'\n')
        story.append(Paragraph(escape(text.lstrip('# ')),styles[style]));story.append(Spacer(1,4))
    def code_block(code,lang='text',first=None):
        md.append(f'```{lang}\n{code}\n```\n')
        formatted=[]
        for i,row in enumerate(code.splitlines()):
            prefix=f'{first+i:>3} | ' if first is not None else ''
            chunks=textwrap.wrap(row,width=100-len(prefix),replace_whitespace=False,drop_whitespace=False) if len(row)+len(prefix)>100 else [row]
            formatted.extend((prefix if j==0 else ' '*len(prefix))+part for j,part in enumerate(chunks))
        story.append(Preformatted('\n'.join(formatted),code_style));story.append(Spacer(1,6))
    def source_context(rel,a,b):
        if not rel.endswith('.py'): return 'archivo de datos'
        tree=ast.parse((HERE/rel).read_text())
        parent={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        containing=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.lineno<=a and n.end_lineno>=b]
        if not containing:return 'nivel principal del módulo'
        node=min(containing,key=lambda n:n.end_lineno-n.lineno)
        names=[node.name];current=parent.get(node)
        while current is not None:
            if isinstance(current,(ast.FunctionDef,ast.ClassDef)):names.append(current.name)
            current=parent.get(current)
        return '.'.join(reversed(names))
    def explain_line(row):
        t=row.strip()
        rules=[
            ("assert docs.document_id.is_unique",'Comprueba que no haya IDs duplicados y que existan exactamente las particiones train, dev y test.'),
            ("assert docs.text.str.strip()",'Exige textos no vacíos después de quitar espacios y categorías no vacías como metadatos.'),
            ("assert len(merged)",'Comprueba que el cruce conservó todas las noticias de entrada.'),
            ("assert (merged.text",'Exige que cada texto coincida exactamente con título más un espacio más descripción.'),
            ("assert (merged.category",'Exige que la categoría conservada coincida con la del registro original; esta comprobación no la entrega a K-Means.'),
            ("near = ids[",'Ordena de menor a mayor distancia al centroide y toma las primeras dos posiciones: las noticias más cercanas de ese grupo.'),
            ("far = ids[",'Toma la última posición del orden creciente de distancias: la noticia más lejana del grupo, usada como frontera.'),
            ("for role, indices",'Procesa primero las posiciones de las dos noticias representativas y después la posición del caso de frontera.'),
            ("for rank,i in enumerate",'Recorre las posiciones seleccionadas y les asigna una numeración desde 1 para guardarlas ordenadas.'),
            ("for c in range",'Repite el cálculo para cada identificador de cluster: en el modelo final son 0 a 11.'),
            ("center.argsort()[::-1][:10]",'Ordena los pesos del centro de mayor a menor, conserva diez posiciones y recupera sus términos del vocabulario.'),
            ("top_word_indices",'Obtiene los índices de las características con más peso, no números de documentos.'),
            ("good = [r",'Filtra los experimentos y conserva solamente los que tienen eligible verdadero.'),
            ("cfg_of(best())",'Toma únicamente la configuración del mejor candidato actual para construir las siguientes comparaciones controladas.'),
            ("caches[featurekey]",'Reutiliza las mismas matrices y transformaciones para esa configuración de características, sin incorporar textos nuevos.'),
            ("fit_transform(tr)",'Aprende la representación exclusivamente con textos train y devuelve su matriz.'),
            ("transform(dv)",'Representa dev con los objetos ya ajustados; aquí no se aprende vocabulario nuevo.'),
            ("fit(Xtr)",'Ajusta K-Means con la matriz de entrenamiento, sin pasar categorías como respuesta.'),
            ("predict(Xdv)",'Asigna los documentos dev a los centros aprendidos con train.'),
            ("final.fit(train_dev.text)",'Reajusta el pipeline completo con los textos de train y dev juntos; test queda fuera.'),
            ("silhouette_score",'Calcula separación geométrica sobre los vectores y grupos indicados; la distancia se fija en euclidean.'),
            ("adjusted_rand_score",'Compara dos particiones. Revisa los argumentos para distinguir estabilidad entre semillas de análisis externo con categorías.'),
            ("normalized_mutual_info_score",'Calcula información compartida entre categorías de referencia y clusters; es análisis externo.'),
            ("inverse_transform",'Reconstruye el centroide desde SVD hacia las columnas del vocabulario original.'),
            ("get_feature_names_out",'Recupera los nombres de los términos en el mismo orden que las columnas de la matriz.'),
            ("np.bincount",'Cuenta noticias por identificador de cluster; minlength incluye también grupos vacíos.'),
            ("np.count_nonzero",'Cuenta cuántos elementos cumplen la condición numérica, como grupos ocupados o vectores sin conteos.'),
            ("np.flatnonzero",'Obtiene posiciones de filas que cumplen la máscara; no son los IDs textuales N0001, N0002, etcétera.'),
            ("argsort",'Ordena índices por el valor numérico: se usa para pesos de términos o distancias; revisa si el corte toma mayores o menores.'),
            ("pd.crosstab",'Cruza dos variables y cuenta cuántos documentos hay en cada combinación.'),
            ("pd.read_csv",'Lee una tabla del disco; sep=\\t indica que sus columnas están separadas por tabuladores.'),
            ("validate='one_to_one'",'Cruza documentos por ID y exige que cada ID aparezca una sola vez en ambos lados.'),
            (".to_csv",'Exporta la tabla al archivo indicado. Esta es la línea que produce la evidencia TSV.'),
            ("write_json",'Guarda en JSON la configuración, decisión o información estructurada que se le entrega.'),
            ("joblib.dump",'Guarda el objeto ajustado en disco para poder cargarlo después sin volver a entrenar.'),
            ("joblib.load",'Carga el modelo persistido; por sí mismo no ajusta ni cambia los centros.'),
            ("save_npz",'Guarda la matriz dispersa de esa partición, conservando el orden de filas y columnas.'),
            ("TfidfVectorizer",'Construye TF-IDF si la representación lo pide; de lo contrario se utiliza el contador de términos.'),
            ("CountVectorizer",'Construye la representación por conteo; binary=True convierte las apariciones en presencia/ausencia.'),
            ("TruncatedSVD",'Define la reducción de dimensiones y su semilla; los parámetros se aprenden cuando se hace fit.'),
            ("Normalizer()",'Añade normalización numérica L2, diferente de la limpieza lingüística.'),
            ("train_test_split",'Realiza el reparto reproducible que la auditoría compara con los splits existentes.'),
            ("eligible =",'Comprueba que la partición cumpla la regla declarada de ocupación, tamaños y métrica definida.'),
            ("return max(good",'Elige el mayor silhouette entre los candidatos admisibles, usando sus valores completos.'),
            ("row = dict(",'Reúne en una fila los parámetros y resultados del experimento; los campos explican cómo se obtuvo su puntuación.'),
            ("rep_rows.append",'Añade ID, texto, URL, distancia y partición de la noticia al listado de representantes o frontera.'),
            ("top_rows.append",'Añade un término representativo con su cluster y posición dentro del top diez.'),
            ("selected",'Registra o consulta la decisión de selección; no es la categoría original de la noticia.'),
            ("ENGLISH_STOP_WORDS",'Elimina tokens pertenecientes a la lista inglesa; no elimina todo tipo de boilerplate.'),
            ("token.lemma_",'Obtiene el lema propuesto por spaCy para ese token.'),
            ("token.pos_",'Consulta su clase gramatical para decidir si se descarta en esta variante.'),
            ("pipe(textos",'Procesa varios textos con spaCy por lotes, en vez de tratar cada texto como una llamada independiente.'),
            ("split",'Filtra, conserva o comprueba la partición de los documentos; no la convierte en característica del texto.'),
            ("ngram_range",'Configura los tamaños de secuencias de tokens que se convertirán en características.'),
            ("n_clusters",'Configura cuántos grupos solicita K-Means, junto con los otros hiperparámetros de esta llamada.'),
            ("random_state",'Fija la semilla usada por la etapa para reproducir la parte aleatoria.'),
            ("distance",'Registra o utiliza la distancia en el espacio de características indicado.'),
            ("text_normalized",'Conserva o utiliza el texto después de la normalización lingüística, antes de los pesos vectoriales.'),
            ("document_id",'Conserva la identidad de la noticia para enlazar textos, matrices y resultados.'),
            ("SimpleDocTemplate",'Construye el PDF a partir de texto, figuras y estilos, sin volver a entrenar el modelo.'),
        ]
        if not t:return None
        if t.startswith('#'):return 'Comentario que documenta la intención; Python no lo ejecuta.'
        if t.startswith('def '):return 'Define la función que se invoca desde otro punto del programa; definirla todavía no ejecuta su contenido.'
        if t.startswith('assert '):return 'Exige que la condición sea verdadera. Si no lo es, la ejecución se detiene con AssertionError.'
        if t.startswith('raise '):return 'Detiene el proceso y comunica la condición que impide continuar.'
        for needle,meaning in rules:
            if needle in t:return meaning
        if t.startswith('for '):return 'Repite el bloque para cada valor de la colección indicada, por ejemplo una configuración, semilla o grupo.'
        if t.startswith(('if ','elif ')):return 'Evalúa la condición y ejecuta esta rama únicamente cuando corresponde.'
        if t.startswith('else:'):return 'Ejecuta la alternativa cuando la condición anterior no se cumple.'
        if t.startswith('return '):return 'Devuelve el resultado de la función a quien la llamó; no implica que se guarde automáticamente en disco.'
        if 'append(' in t:return 'Agrega un elemento a la colección que se exportará o presentará después.'
        if 'print(' in t:return 'Muestra información al ejecutar la consola; no cambia el ajuste del modelo.'
        if t.startswith('add('):return 'Añade este texto o resultado a la versión escrita del informe.'
        if '=' in t:return 'Asigna el valor o resultado de la derecha a la variable de la izquierda para usarlo en los pasos siguientes.'
        return 'Continúa la expresión de la línea anterior; hay que leerla dentro de la llamada o estructura completa mostrada arriba.'

    prose('# Guía de estudio: noticias y K-Means, paso a paso','Title')
    prose('Elaboración conjunta de todos los integrantes del equipo: Ricardo Hernandez Espinoza; Diana Karen Espinoza Diaz; Byron Robles Lozano; Hector Arturo Saenz Buttanda; Diamanda Abigail Zurita Salinas.')
    prose('Esta guía es para entender y exponer el trabajo. Primero explica las ideas, después qué hace cada archivo y finalmente dónde está cada requisito del profesor, con función, líneas, código exacto y lectura del fragmento. El informe de resultados sigue en resultados/INFORME_NO_SUPERVISADO.pdf.')
    prose('Carpeta base: fase_2_aprendizaje/no_supervisada/. Las rutas de esta guía son relativas a esa carpeta. Los enlaces Markdown abren el código local. En el PDF se muestran números de línea y se dividen visualmente las líneas largas; para copiar código exacto usa Markdown o el archivo fuente.')
    prose('## Cómo estudiar esta guía','Heading1')
    for text in [
        'Primera lectura: sigue el recorrido y el glosario, sin intentar memorizar nombres de archivos.',
        'Segunda lectura: revisa las fichas de archivos para distinguir entradas, programas, modelo y resultados.',
        'Tercera lectura: estudia los apartados 1–23; en cada uno conecta requisito, idea, código y evidencia.',
        'Antes de exponer: practica las preguntas del final y usa inspeccionar.py para mostrar un grupo real.',
        'Para buscar algo concreto: usa Ctrl+F o Cmd+F con palabras como datos, validación, stopwords, silhouette, representantes o el nombre del TSV.'
    ]:prose('- '+text)
    prose('## El recorrido completo, contado sin código','Heading1')
    for text in [
        '1. Partimos de las mismas noticias del equipo. Revisamos IDs, texto y particiones, para no entrenar con un archivo desalineado o incompleto.',
        '2. Cada candidato toma una forma de normalizar palabras, una representación y un rango de n-gramas. Esas decisiones producen una matriz de números.',
        '3. Ajustamos las transformaciones con train; reducimos con SVD si corresponde, normalizamos L2 y entrenamos K-Means.',
        '4. Representamos dev con esas mismas transformaciones y asignamos sus noticias con predict. Medimos separación y tamaños para comparar candidatos.',
        '5. Elegimos KM083 mediante la regla declarada, revisamos sensibilidad a semillas y congelamos su configuración.',
        '6. Reajustamos el pipeline completo en train+dev y guardamos el modelo final. Después asignamos test sin cambiar los parámetros.',
        '7. Exportamos grupos, tamaños, palabras, noticias cercanas y lejanas, métricas y ejemplos de eventos. El equipo interpreta temas y limitaciones.'
    ]:prose(text)
    code_block('Datos originales + texto + split\n    -> normalización lingüística\n    -> vectorizador (pesos + n-gramas)\n    -> SVD opcional -> L2 -> K-Means\n    -> comparación en dev -> selección\n    -> reajuste con train+dev -> predict en test\n    -> tablas + modelo + interpretación + informe')
    prose('## Cuando dices revalidación: cuatro cosas diferentes','Heading1')
    prose('La palabra revalidar puede resultar ambigua. No hay que inventar un paso del PDF con ese nombre: las secciones 4, 5, 7 y 9 piden controles y evaluación concretos. Estas son las cuatro operaciones que sí puedes señalar en el proyecto:')
    validation_cases=[
        ('Validar la integridad de los datos','experimentos.py','load_documents','Comprueba IDs, texto no vacío y coincidencia exacta con título + descripción. Es la revisión del archivo de entrada.'),
        ('Validar candidatos con dev','experimentos.py','trial','Ajusta en train, representa dev, predice sus grupos y calcula silhouette y tamaños. Sirve para comparar configuraciones.'),
        ('Reajustar el modelo final','experimentos.py','run','final.fit(train_dev.text) aprende de nuevo con train+dev tras elegir. No es una comprobación del TSV.'),
        ('Auditar los resultados existentes','auditar_cumplimiento.py','run','Contrasta matrices, métricas, representantes y particiones ya guardadas. No vuelve a seleccionar parámetros ni reentrena K-Means.')
    ]
    for title,rel,name,meaning in validation_cases:
        block=function(rel,name)
        if title.startswith('Reajustar'):block=span(rel,'    final.fit(train_dev.text)')
        prose(title+': '+meaning)
        prose(f'Ubicación precisa: {rel}, función {source_context(rel,block[1],block[2])}, líneas {block[1]}–{block[2]}. Los fragmentos completos se explican en los apartados correspondientes.')
    prose('## Diccionario para entender el código','Heading1')
    for term,meaning in GLOSSARY:prose(term+': '+meaning)
    prose('## Tu captura: archivos anteriores y archivos actuales','Heading1')
    prose('La captura corresponde a la carpeta anterior fase_2_aprendizaje/agrupamiento/. Conservamos esa ejecución como referencia. La entrega nueva está en no_supervisada/. No mezcles sus métricas ni sus nombres de modelo: el trabajo anterior tenía 50 pruebas; el nuevo tiene 91 candidatos, tres ajustes de estabilidad y doce grupos finales.')
    for old,newname,meaning in OLD_NEW:
        prose('Antes: '+old+' | Ahora: '+newname+'. '+meaning)
    prose('## Mapa de carpetas','Heading1')
    code_block('no_supervisada/\n  datos/       Noticias originales, ampliación y texto con split\n  soporte/     Normalizador del pipeline del equipo\n  referencias/ PDF original del profesor\n  modelos/     Pipeline ajustado y matrices binarias finales\n  resultados/  Tablas, métricas, auditoría e informe\n    figuras/   Gráficas incorporadas al informe\n  *.py         Programas de aprendizaje, consulta y documentación\n  *.json       Interpretación y material de estudio\n  *.md / *.pdf Guía, instrucciones y estado\n  *.zip        Paquete con toda la entrega')
    prose('## Qué hace cada archivo y cómo se usa','Heading1')
    prose('Un archivo de resultados no contiene código ejecutable: en sus fichas explicamos qué significa y cómo se abre. El código que lo genera está identificado en los apartados por requisito. Una fila de TSV es un registro, no necesariamente una línea física si su texto incluye saltos. Para el código sí damos líneas físicas exactas.')
    inventory=sorted(set(FILES)|{str(f.relative_to(HERE)) for f in HERE.rglob('*') if f.is_file() and not any(x in f.parts for x in ['.cache','__pycache__']) and f.name!='entrega_no_supervisada.zip'})
    inventory.append('entrega_no_supervisada.zip') if 'entrega_no_supervisada.zip' not in inventory else None
    for rel in inventory:
        kind,purpose,use=FILES.get(rel,('Archivo auxiliar','Recurso de la carpeta de estudio o de la ejecución. Consultar su nombre y extensión antes de usarlo.','No interviene por sí mismo en la selección del modelo.'))
        prose('### '+rel,'Heading2')
        prose('Tipo: '+kind+'. Qué hace: '+purpose)
        prose('Cómo usarlo: '+use)
        if rel.endswith('.py') and (HERE/rel).exists():
            tree=ast.parse((HERE/rel).read_text())
            functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
            if functions:
                prose('Funciones para localizar: '+'; '.join(f'{source_context(rel,n.lineno,n.end_lineno)} (líneas {n.lineno}–{n.end_lineno})' for n in sorted(functions,key=lambda n:n.lineno))+'.')
    prose('## Requisitos del profesor explicados y localizados','Heading1')
    prose('Cada referencia al PDF es una explicación fiel en español, no una cita literal. Primero se explica la intención, después la implementación y finalmente el código y la evidencia. El número del apartado de esta guía no es el número de sección del PDF: se indica por separado.')
    for section in sections:
        number=section['title'].split('.')[0]
        teaching=study[number]
        prose('## '+section['title'],'Heading1')
        prose('Qué pide el profesor: consultar sección(es) '+section['pdf']+' del PDF. Resumen del requisito: '+section['explain'])
        prose('### Para entenderlo desde cero','Heading2')
        for paragraph in teaching['explicacion']:prose(paragraph)
        prose('Ejemplo: '+teaching['ejemplo'])
        prose('### Dónde está y qué archivo demuestra el resultado','Heading2')
        for rel in section['outputs']:
            kind,purpose,use=FILES.get(rel,('Evidencia','Archivo del proceso descrito.','Consultar la ficha del catálogo.'))
            prose('- '+rel+': '+purpose)
        prose('### Cómo leer la implementación','Heading2')
        for step in teaching['lectura_codigo']:prose('- '+step)
        for rel,a,b,code in section['blocks']:
            absolute=HERE/rel
            context=source_context(rel,a,b)
            md.append(f'Archivo y enlace exacto: [{rel}]({absolute}:{a}). Función o contexto: `{context}`. Líneas {a}–{b}.\n')
            story.append(Paragraph(escape(f'Archivo: {rel} | Función/contexto: {context} | Líneas {a}–{b}'),styles['BodyText']))
            code_block(code,'python' if rel.endswith('.py') else 'text',first=a)
            prose('Lectura de las líneas de este fragmento:')
            for line_number,row in enumerate(code.splitlines(),a):
                meaning=explain_line(row)
                if meaning:prose(f'- Línea {line_number}: '+meaning)
        prose('Para explicarlo con tus palabras: '+teaching['para_exponer'])
    prose('## Cómo leer los archivos con resultados: ejemplos reales','Heading1')
    import pandas as pd
    results=pd.read_csv(OUT/'clustering_results.tsv',sep='\t')
    example=results[results.document_id.eq('N2256')].iloc[0]
    prose(f'Ejemplo de asignación real: document_id={example.document_id}; split={example.split}; cluster_id={example.cluster_id}. document_id identifica la noticia, split indica su partición y cluster_id el grupo aprendido. method/configuration describe la configuración; no es la etiqueta gold.')
    sizes=pd.read_csv(OUT/'cluster_sizes.tsv',sep='\t')
    size=sizes[sizes.cluster_id.eq(0)].iloc[0]
    prose(f'Ejemplo de tamaño real: cluster 0 tiene {size["train"]} train, {size["dev"]} dev y {size["test"]} test: {size.total} en total. Es el mismo modelo final que generó las asignaciones.')
    terms=pd.read_csv(OUT/'top_words.tsv',sep='\t')
    prose('Ejemplo de términos del cluster 1: '+', '.join(terms[terms.cluster_id.eq(1)].sort_values('rank').term)+'. Son características de texto: algunos términos tienen más de una palabra.')
    prose('Ejemplo de una fila de experiments.tsv: KM083 tiene normalization=sin_stopwords, representation=binaria, ngram=(1,2), svd=50, k=12, n_init=10, init=k-means++, max_iter=300, random_state=0, eligible=True y selected=True. No significa que esos sean los únicos valores probados.')
    prose('Diferencia de métricas que debes recordar: seleccion_dev.json contiene dev de selección 0.165800; metricas_finales.tsv contiene train 0.155980, dev 0.154702 y test 0.177708 del modelo ya reajustado. El dev reajustado no se usa para justificar el ganador.')
    prose('## Comandos de estudio y demostración','Heading1')
    prose('Ejecutar desde la raíz NLP_News_Project, con el entorno ya preparado:')
    code_block('.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py\n.venv/bin/python fase_2_aprendizaje/no_supervisada/inspeccionar.py --cluster 2\n.venv/bin/python fase_2_aprendizaje/no_supervisada/validar_resultados.py\n.venv/bin/python fase_2_aprendizaje/no_supervisada/auditar_cumplimiento.py','bash')
    prose('Primer comando: resumen. Segundo: términos y noticias de un grupo. Tercero: validación breve. Cuarto: auditoría profunda. No necesitas ejecutar experimentos.py para estudiar; eso corresponde a una corrida nueva en una copia limpia.')
    prose('## Preguntas para practicar antes de exponer','Heading1')
    for question,answer in QUESTIONS:
        prose('Pregunta: '+question,'Heading2');prose('Respuesta: '+answer)
    prose('## Un guion breve para contar el trabajo del equipo','Heading1')
    prose('Nuestro equipo reutilizó las noticias y las particiones de la etapa anterior. Comparamos tres normalizaciones, tres representaciones y tres rangos de n-gramas con K-Means, sin usar categorías para entrenar. Seleccionamos con dev mediante silhouette y controles de tamaño; ganó una representación binaria sin stopwords, con unigramas y bigramas, SVD de 50 dimensiones y doce grupos. Después reajustamos el pipeline completo con train+dev y evaluamos test sin volver a elegir parámetros. Exportamos tamaños, diez términos y dos noticias por grupo, estudiamos estabilidad y eventos y explicamos tanto los temas reconocibles como los grupos contaminados por formato editorial.')
    prose('## Qué corresponde a esta rama y qué va en el informe general','Heading1')
    prose('Esta guía estudia K-Means, la tarea B elegida por el equipo. LDA y perplexity pertenecen a otra rama opcional. Macro-F1, accuracy y los modelos Naive Bayes, regresión logística y SVC corresponden a supervisada. La entrega completa de etapa II integra ambas tareas en el informe general. La autoría de esta parte es conjunta de todos los integrantes. Los grupos mixtos y el silhouette modesto son resultados analizados, no garantías de que existan doce temas puros.')
    final_md='\n'.join(md)
    for name in ['GUIA_DE_ESTUDIO_NO_SUPERVISADO.md','GUIA_REQUISITOS_Y_CODIGO.md']:
        (HERE/name).write_text(final_md,encoding='utf-8')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.drawString(1.4*cm,.9*cm,'Guía de estudio del equipo - K-Means');canvas.drawRightString(19.6*cm,.9*cm,str(doc.page))
    class StudyDocument(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name=='Heading1':
                key=f'study-{self.seq.nextf("outline")}';self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0,closed=False)
    document=StudyDocument(str(HERE/'GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf'),pagesize=(21*cm,29.7*cm),leftMargin=1.4*cm,rightMargin=1.4*cm,topMargin=1.4*cm,bottomMargin=1.5*cm)
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    (HERE/'GUIA_REQUISITOS_Y_CODIGO.pdf').write_bytes((HERE/'GUIA_DE_ESTUDIO_NO_SUPERVISADO.pdf').read_bytes())
    for section in sections:
        for rel,a,b,code in section['blocks']:
            assert code=='\n'.join(lines(rel)[a-1:b])
    print('Guía creada:',len(sections),'requisitos,',sum(len(s['blocks']) for s in sections),'fragmentos exactos verificados.')


if __name__=='__main__':
    main()
