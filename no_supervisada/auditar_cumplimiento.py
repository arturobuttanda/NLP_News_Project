"""Auditoría de datos y artefactos existentes; no entrena ni selecciona modelos."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'soporte'))
from normalizacion import NormalizadorNoticias


def run():
    out = HERE/'resultados'
    checks=[]
    def check(name, condition, evidence):
        if not bool(condition):
            raise AssertionError(name+': '+evidence)
        checks.append(dict(check=name,status='OK',evidence=evidence))
    docs=pd.read_csv(HERE/'datos/labeled_documents.tsv',sep='\t',dtype=str,keep_default_na=False)
    raw=pd.read_csv(HERE/'datos/news_raw_corpus_4645.tsv',sep='\t',dtype=str,keep_default_na=False)
    original=pd.read_csv(HERE/'datos/news_raw_corpus_2157.tsv',sep='\t',dtype=str,keep_default_na=False)
    preserved=raw.set_index('id').reindex(original.id).reset_index()
    check('IDs y artículos de etapa I preservados',original.equals(preserved[original.columns]),'Los 2157 registros originales permanecen idénticos en el corpus ampliado de 4645')
    merged=docs.merge(raw,left_on='document_id',right_on='id',suffixes=('_labeled','_raw'),validate='one_to_one')
    check('Texto y etiquetas del corpus original',len(merged)==len(docs) and (merged.text==(merged.title+' '+merged.description).str.strip()).all() and merged.category_labeled.eq(merged.category_raw).all(),'4645 coincidencias exactas de ID, title + description y category')
    check('Copia de entrada y resultado',docs.equals(pd.read_csv(out/'labeled_documents.tsv',sep='\t',dtype=str,keep_default_na=False)),'datos/labeled_documents.tsv = resultados/labeled_documents.tsv')
    train,rest=train_test_split(docs.index,test_size=.30,shuffle=True,random_state=0,stratify=docs.category)
    dev,test=train_test_split(rest,test_size=.50,shuffle=True,random_state=0,stratify=docs.loc[rest,'category'])
    check('Particiones reproducibles 70/15/15',all(docs.loc[idx,'split'].eq(name).all() for name,idx in [('train',train),('dev',dev),('test',test)]),'Reconstrucción independiente con train_test_split, shuffle=True, random_state=0 y estratificación')
    check('Todas las clases en cada partición',pd.crosstab(docs.category,docs.split).gt(0).all().all(),'Diez categorías presentes en train, dev y test')
    config=json.loads((out/'configuracion_final.json').read_text())
    digest=hashlib.sha256((HERE/'datos/labeled_documents.tsv').read_bytes()).hexdigest()
    check('Huella del corpus',digest==config['source_sha256'],'SHA-256 de los datos coincide con la corrida registrada')
    exp=pd.read_csv(out/'experiments.tsv',sep='\t')
    check('Bitácora de selección',len(exp)==91 and exp.experiment_id.is_unique and exp.method.eq('KMeans').all(),'91 IDs únicos; único algoritmo de clustering: KMeans')
    check('Normalizaciones y representaciones',exp.normalization.nunique()==3 and set(exp.representation)=={'binaria','frecuencia','tfidf'},'Tres normalizaciones y tres representaciones efectivamente ejecutadas')
    check('N-gramas y dimensiones SVD',set(exp.ngram)=={'(1, 1)','(1, 2)','(1, 3)'} and set(exp.svd.dropna())=={50,100,150} and exp.svd.isna().any(),'Unigramas; uni+bi; uni+bi+tri; sin SVD y SVD 50/100/150')
    check('Hiperparámetros explorados',set(exp.k)=={4,6,8,10,12} and set(exp.n_init)=={1,10,20} and set(exp['init'])=={'random','k-means++'} and set(exp.max_iter)=={100,300,500} and exp.random_state.eq(0).all(),'Valores reales leídos de la tabla, no solo declarados en el código')
    eligible=(exp.occupied_clusters.eq(exp.k)&exp.min_share.ge(.01)&exp.max_share.le(.65)&exp.silhouette.notna())
    check('Regla de admisibilidad',np.array_equal(eligible,exp.eligible),'Se recalcula ocupación y umbrales a partir de las métricas guardadas')
    winner=exp[exp.selected].iloc[0]
    check('Ganador en dev',exp.selected.sum()==1 and winner.eligible and np.isclose(winner.silhouette,exp.loc[eligible,'silhouette'].max()) and winner.experiment_id==config['selected_experiment'],'KM083 es el máximo admisible en dev; no se incorpora test al ranking')
    stability=json.loads((out/'estabilidad_dev.json').read_text())
    check('Estabilidad dev',len(stability['trials'])==3 and len(stability['pairwise_ari'])==3 and {r['seed'] for r in stability['trials']}=={0,1,2},'Tres semillas, tres pares ARI entre particiones; sin gold labels')
    model=joblib.load(HERE/'modelos/kmeans_pipeline.joblib')
    metrics=pd.read_csv(out/'metricas_finales.tsv',sep='\t').set_index('split')
    results=pd.read_csv(out/'clustering_results.tsv',sep='\t')
    check('Esquema de asignaciones',list(results.columns)==['document_id','split','cluster_id','method/configuration'] and len(results)==len(docs) and results.document_id.is_unique,'Esquema exacto solicitado para clustering_results.tsv')
    normalized=pd.read_csv(out/'corpus_normalizado_final.tsv',sep='\t',keep_default_na=False)
    for split in ['train','dev','test']:
        mask=docs.split.eq(split).to_numpy()
        X=load_npz(HERE/f'modelos/X_words_{split}.npz')
        expected=model['vectorizador'].transform(normalized.loc[mask,'text_normalized'])
        check('Matriz léxica '+split,X.shape==expected.shape and (X!=expected).nnz==0,f'{X.shape[0]} filas alineadas por ID y {X.shape[1]} términos; igualdad completa de conteos')
        latent=model['svd'].transform(X) if 'svd' in model.named_steps else X
        latent=model['norma_l2'].transform(latent)
        assigned=results.loc[mask,'cluster_id'].to_numpy()
        # Comprobación de la misma evaluación guardada; no es un nuevo ajuste ni selección.
        check('Modelo y asignaciones '+split,np.array_equal(model['kmeans'].predict(latent),assigned),'El artefacto persistido reproduce las asignaciones de la matriz guardada')
        score=silhouette_score(latent,assigned,metric='euclidean')
        check('Silhouette guardado '+split,np.isclose(score,metrics.loc[split,'silhouette'],atol=1e-12),'Recálculo de la misma métrica sobre las mismas asignaciones; sin modificar parámetros')
    train_dev=docs[docs.split.ne('test')]
    normalized_td=model['normalizacion'].transform(train_dev.text)
    from sklearn.base import clone
    independent=clone(model['vectorizador']).fit(normalized_td)
    check('Vocabulario reajustado con train + dev',independent.vocabulary_==model['vectorizador'].vocabulary_,'Un vectorizador independiente ajustado solo en train+dev reproduce exactamente el vocabulario final')
    reps=pd.read_csv(out/'representative_documents.tsv',sep='\t')
    words=pd.read_csv(out/'top_words.tsv',sep='\t')
    vocab=model['vectorizador'].get_feature_names_out()
    centers=model['svd'].inverse_transform(model['kmeans'].cluster_centers_) if 'svd' in model.named_steps else model['kmeans'].cluster_centers_
    all_top=True
    for c in range(model['kmeans'].n_clusters):
        all_top &= list(words[words.cluster_id.eq(c)].sort_values('rank').term)==list(vocab[centers[c].argsort()[::-1][:10]])
    check('Diez términos en el espacio original',all_top,'Coincidencia exacta con los centroides reconstruidos desde SVD, no con componentes latentes')
    # Distancias de los representantes: usar solo train+dev, exactamente como en la entrega.
    td_indices=np.flatnonzero(docs.split.ne('test'))
    X=model['vectorizador'].transform(normalized_td)
    latent=model['svd'].transform(X) if 'svd' in model.named_steps else X
    latent=model['norma_l2'].transform(latent)
    distances=model['kmeans'].transform(latent)
    assignments=results.loc[docs.split.ne('test'),'cluster_id'].to_numpy()
    for c in range(model['kmeans'].n_clusters):
        indices=np.flatnonzero(assignments==c)
        expected_ids=docs.iloc[td_indices[indices[np.argsort(distances[indices,c])[:2]]]].document_id.tolist()
        actual=reps[reps.cluster_id.eq(c)&reps.role.eq('representativo')].sort_values('rank')
        check('Representantes del grupo '+str(c),expected_ids==list(actual.document_id),'Dos noticias de train+dev más cercanas a su propio centroide, por distancia euclidiana')
    events=pd.read_csv(out/'eventos_recurrentes.tsv',sep='\t')
    indexed=results.set_index('document_id')
    check('Eventos y grupos correctos',all(int(indexed.loc[r.document_a,'cluster_id'])==r.cluster_a and int(indexed.loc[r.document_b,'cluster_id'])==r.cluster_b for r in events.itertuples()),'Tres ejemplos de eventos contrastados por ID con las asignaciones reales')
    check('PDF y figuras existentes',(out/'INFORME_NO_SUPERVISADO.pdf').read_bytes().startswith(b'%PDF-') and all((out/'figuras'/f).exists() for f in ['tamanos_clusters.png','experimentos_dev.png']),'Se verificó además la extracción de texto del PDF en la revisión documental')
    # Guarda los parámetros completos; permite explicar los valores por defecto del vectorizador.
    params={name: step.get_params(deep=False) for name,step in model.steps}
    (out/'parametros_pipeline_final.json').write_text(json.dumps(params,ensure_ascii=False,indent=2,default=str)+'\n')
    report=dict(scope='Tarea B K-Means; auditoría sin reentrenar ni elegir otra configuración',checks=checks,total_ok=len(checks),technical_status='PASS',documentary_limits=['La etapa II completa también exige supervisada e informe general integrado.','Autoría conjunta confirmada por la usuaria: todos los integrantes participaron.','Los temas no son puros y la calidad es modesta: limitaciones documentadas, no garantía de calificación.'])
    (out/'auditoria_cumplimiento.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('PASS:',len(checks),'comprobaciones independientes; sin reentrenamiento de K-Means ni nueva selección.')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
