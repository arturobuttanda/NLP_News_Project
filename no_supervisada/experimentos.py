"""Experimentos de K-Means: selección en dev y ajuste final sin usar test.

Ejecutar desde la raíz con .venv/bin/python -u fase_2_aprendizaje/no_supervisada/experimentos.py
"""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'soporte'))
from normalizacion import NormalizadorNoticias

OUT = HERE / 'resultados'
MOD = HERE / 'modelos'
SEED = 0
NORMS = ['minima', 'sin_stopwords', 'lemas']
REPS = ['binaria', 'frecuencia', 'tfidf']
GRAMS = [(1, 1), (1, 2), (1, 3)]
# Regla predefinida de admisibilidad, sin consultar categorías ni test.
MIN_SHARE = .01
MAX_SHARE = .65


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def load_documents():
    path = HERE / 'datos/labeled_documents.tsv'
    docs = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    assert docs.document_id.is_unique and set(docs.split) == {'train', 'dev', 'test'}
    assert docs.text.str.strip().ne('').all() and docs.category.ne('').all()
    # Comprueba que texto y etiquetas corresponden al corpus original, sin cambiar los splits.
    raw = pd.read_csv(HERE / 'datos/news_raw_corpus_4645.tsv', sep='\t', dtype=str, keep_default_na=False)
    merged = docs.merge(raw, left_on='document_id', right_on='id', suffixes=('_labeled', '_raw'), validate='one_to_one')
    assert len(merged) == len(docs)
    assert (merged.text == (merged.title + ' ' + merged.description).str.strip()).all()
    assert (merged.category_labeled == merged.category_raw).all()
    return docs, path


def feature_pipeline(cfg):
    opts = dict(ngram_range=tuple(cfg['ngram']), min_df=2, max_df=.95, max_features=30000)
    vect = TfidfVectorizer(**opts) if cfg['representation'] == 'tfidf' else CountVectorizer(binary=cfg['representation'] == 'binaria', **opts)
    steps = [('vectorizador', vect)]
    if cfg['svd']:
        steps.append(('svd', TruncatedSVD(n_components=cfg['svd'], random_state=SEED)))
    steps.append(('norma_l2', Normalizer()))
    return Pipeline(steps)


def kmeans(cfg, seed=SEED):
    return KMeans(n_clusters=cfg['k'], n_init=cfg['n_init'], init=cfg['init'], max_iter=cfg['max_iter'], random_state=seed)


def quality(X, labels, k):
    counts = np.bincount(labels, minlength=k)
    n_labels = np.count_nonzero(counts)
    sil = float(silhouette_score(X, labels, metric='euclidean')) if 1 < n_labels < len(labels) else None
    return dict(silhouette=sil, occupied_clusters=int(n_labels), min_cluster=int(counts.min()), max_cluster=int(counts.max()), min_share=float(counts.min()/len(labels)), max_share=float(counts.max()/len(labels)), sizes=counts.tolist())


def lexical_summary(vect, svd, centers):
    centers = svd.inverse_transform(centers) if svd is not None else centers
    vocab = vect.get_feature_names_out()
    return [[str(vocab[i]) for i in center.argsort()[::-1][:10]] for center in centers]


def experiment_interpretation(exp_id, cfg, fp, km, Xdev, dev, labels):
    words = lexical_summary(fp['vectorizador'], fp.named_steps.get('svd'), km.cluster_centers_)
    dist = km.transform(Xdev)
    rows = []
    for c in range(cfg['k']):
        ids = np.flatnonzero(labels == c)
        nearest = ids[np.argsort(dist[ids, c])[:2]]
        rows.append(dict(experiment_id=exp_id, cluster_id=c, dev_size=len(ids), top_words=', '.join(words[c]), representative_ids=' | '.join(dev.iloc[nearest].document_id), representative_texts=' || '.join(dev.iloc[nearest].text)))
    return rows


def run():
    OUT.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    if (OUT / 'configuracion_final.json').exists():
        raise SystemExit('Ya existe una corrida final. Usa una copia de la carpeta para un nuevo estudio; no se reabre test automáticamente.')
    docs, source = load_documents()
    docs.to_csv(OUT / 'labeled_documents.tsv', sep='\t', index=False)
    train = docs[docs.split == 'train'].reset_index(drop=True)
    dev = docs[docs.split == 'dev'].reset_index(drop=True)
    plan = dict(seed=SEED, normalizations=NORMS, representations=REPS, ngrams=GRAMS, k=[4,8,12], stage1_svd=100, vectorizer=dict(min_df=2,max_df=.95,max_features=30000), distance='euclidean after SVD (if used) and L2 normalization', selection=f'All k dev clusters occupied; min share >= {MIN_SHARE}; max share <= {MAX_SHARE}; then maximum unrounded dev silhouette, stable ID tie break. If no admissible trial, halt before test.', stages=['81 factorial trials','SVD none/50/150 and k 6/10 around stage-1 winner','n_init/init/max_iter one-factor comparisons','seed stability 0/1/2, fixed selected parameters'], source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    write_json(OUT / 'plan_experimental.json', plan)
    docs.groupby(['category','split']).size().unstack(fill_value=0).to_csv(OUT/'distribucion_corpus.tsv',sep='\t')
    normalized = {}
    for mode in NORMS:
        print('Normalizando train/dev:', mode, flush=True)
        normalizer = NormalizadorNoticias(modo=mode)
        normalized[mode] = (normalizer.transform(train.text), normalizer.transform(dev.text))
    records, interpretations, caches = [], [], {}
    seen = set()

    def trial(cfg, stage):
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        t0 = time.perf_counter()
        featurekey = (cfg['normalization'],cfg['representation'],tuple(cfg['ngram']),cfg['svd'])
        if featurekey not in caches:
            fp = feature_pipeline(cfg)
            tr, dv = normalized[cfg['normalization']]
            Xtr = fp.fit_transform(tr)
            Xdv = fp.transform(dv)
            # Mantener sólo unos pocos espacios en memoria.
            if len(caches) >= 4:
                caches.pop(next(iter(caches)))
            caches[featurekey] = (fp,Xtr,Xdv)
        fp, Xtr, Xdv = caches[featurekey]
        km = kmeans(cfg).fit(Xtr)
        labels = km.predict(Xdv)
        q = quality(Xdv, labels, cfg['k'])
        eligible = q['silhouette'] is not None and q['occupied_clusters'] == cfg['k'] and q['min_share'] >= MIN_SHARE and q['max_share'] <= MAX_SHARE
        exp_id = f'KM{len(records)+1:03d}'
        row = dict(experiment_id=exp_id,task='clustering',method='KMeans',stage=stage,**cfg,**q,eligible=eligible,selected=False,random_state=SEED,n_iter=int(km.n_iter_),seconds=time.perf_counter()-t0,vocabulary_size=len(fp['vectorizador'].vocabulary_),zero_train=int(np.sum(fp['vectorizador'].transform(normalized[cfg['normalization']][0]).getnnz(axis=1)==0)),zero_dev=int(np.sum(fp['vectorizador'].transform(normalized[cfg['normalization']][1]).getnnz(axis=1)==0)))
        records.append(row)
        interpretations.extend(experiment_interpretation(exp_id,cfg,fp,km,Xdv,dev,labels))
        pd.DataFrame(records).to_csv(OUT/'experiments.tsv',sep='\t',index=False)
        pd.DataFrame(interpretations).to_csv(OUT/'dev_cluster_interpretation.tsv',sep='\t',index=False)
        print(exp_id, stage, cfg, 'sil=', q['silhouette'], 'sizes=',q['sizes'], 'eligible=',eligible, flush=True)

    def best():
        good = [r for r in records if r['eligible']]
        if not good:
            raise RuntimeError('Ninguna configuración admisible. Revisar resultados dev sin acceder a test.')
        return max(good,key=lambda r:r['silhouette'])

    def cfg_of(row):
        return {k:row[k] for k in ['normalization','representation','ngram','svd','k','n_init','init','max_iter']}

    for mode,rep,ngram,k in itertools.product(NORMS,REPS,GRAMS,[4,8,12]):
        trial(dict(normalization=mode,representation=rep,ngram=ngram,svd=100,k=k,n_init=10,init='k-means++',max_iter=300),'factorial')
    base = cfg_of(best())
    for svd in [None,50,150]:
        trial({**base,'svd':svd},'svd')
    for k in [6,10]:
        trial({**base,'k':k},'k_refinement')
    base = cfg_of(best())
    for value in [1,20]:
        trial({**base,'n_init':value},'n_init')
    trial({**base,'init':'random'},'init')
    for value in [100,500]:
        trial({**base,'max_iter':value},'max_iter')
    winner = best()
    winner['selected'] = True
    cfg = cfg_of(winner)
    pd.DataFrame(records).to_csv(OUT/'experiments.tsv',sep='\t',index=False)
    write_json(OUT/'seleccion_dev.json',dict(experiment_id=winner['experiment_id'],config=cfg,selection=plan['selection'],silhouette_dev=winner['silhouette']))
    # Estabilidad: ARI entre asignaciones, sin usar categorías gold.
    fp = feature_pipeline(cfg)
    Xtr = fp.fit_transform(normalized[cfg['normalization']][0])
    Xdv = fp.transform(normalized[cfg['normalization']][1])
    assignments = {}
    stability = []
    for seed in [0,1,2]:
        km = kmeans(cfg,seed).fit(Xtr)
        labels = km.predict(Xdv)
        assignments[seed] = labels
        stability.append(dict(seed=seed,**quality(Xdv,labels,cfg['k'])))
    write_json(OUT/'estabilidad_dev.json',dict(trials=stability,pairwise_ari=[dict(seed_a=a,seed_b=b,ari=adjusted_rand_score(assignments[a],assignments[b])) for a,b in itertools.combinations(assignments,2)]))
    # Se congela la configuración ANTES de leer los textos y etiquetas de test para evaluación.
    train_dev = docs[docs.split.isin(['train','dev'])].reset_index(drop=True)
    final = Pipeline([('normalizacion',NormalizadorNoticias(modo=cfg['normalization'])),*feature_pipeline(cfg).steps,('kmeans',kmeans(cfg))])
    print('Ajuste final del pipeline completo en train + dev:',cfg,flush=True)
    final.fit(train_dev.text)
    joblib.dump(final,MOD/'kmeans_pipeline.joblib')
    write_json(OUT/'configuracion_final.json',dict(selected_experiment=winner['experiment_id'],config=cfg,train_dev_count=len(train_dev),test_used_for_selection=False,vectorizer_refitted_on='train+dev',source_sha256=plan['source_sha256'],python=platform.python_version(),versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','spacy','en-core-web-sm','joblib','reportlab','matplotlib','threadpoolctl']},seed=SEED))
    # Transformación única de todo el corpus con el pipeline final ya ajustado.
    normalized_all = final['normalizacion'].transform(docs.text)
    Xwords = final['vectorizador'].transform(normalized_all)
    X = Xwords
    if 'svd' in final.named_steps:
        X = final['svd'].transform(X)
    X = final['norma_l2'].transform(X)
    labels = final['kmeans'].predict(X)
    results = docs[['document_id','split']].copy()
    results['cluster_id'] = labels
    results['method/configuration'] = json.dumps(dict(method='KMeans',**cfg,random_state=SEED),ensure_ascii=False)
    results.to_csv(OUT/'clustering_results.tsv',sep='\t',index=False)
    metrics = []
    for split in ['train','dev','test']:
        mask = docs.split.eq(split).to_numpy()
        q = quality(X[mask],labels[mask],cfg['k'])
        metrics.append(dict(split=split,model='final_refit_train_dev',feature_space=f"svd={cfg['svd']} + L2",distance='euclidean',**q,ari=adjusted_rand_score(docs.loc[mask,'category'],labels[mask]),nmi=normalized_mutual_info_score(docs.loc[mask,'category'],labels[mask]),zero_vectors=int(np.sum(Xwords[mask].getnnz(axis=1)==0))))
        pd.crosstab(pd.Series(labels[mask],name='cluster_id'),pd.Series(docs.loc[mask,'category'].to_numpy(),name='category')).reindex(range(cfg['k']),fill_value=0).to_csv(OUT/f'contingencia_{split}.tsv',sep='\t')
    pd.DataFrame(metrics).to_csv(OUT/'metricas_finales.tsv',sep='\t',index=False)
    counts = pd.crosstab(results.cluster_id,results.split).reindex(range(cfg['k']),fill_value=0)
    counts['total'] = counts.sum(axis=1)
    counts.to_csv(OUT/'cluster_sizes.tsv',sep='\t')
    words = lexical_summary(final['vectorizador'],final.named_steps.get('svd'),final['kmeans'].cluster_centers_)
    top_rows, rep_rows = [], []
    distances = final['kmeans'].transform(X)
    for c in range(cfg['k']):
        for rank, word in enumerate(words[c],1):
            top_rows.append(dict(cluster_id=c,rank=rank,term=word,space='centroid inverse-projected to original word features' if cfg['svd'] else 'word-feature centroid'))
        # Interpretación final usando documentos de ajuste. Test queda como evidencia de generalización.
        ids = np.flatnonzero((labels==c) & docs.split.ne('test').to_numpy())
        near = ids[np.argsort(distances[ids,c])[:2]]
        far = ids[np.argsort(distances[ids,c])[-1:]]
        for role, indices in [('representativo',near),('frontera',far)]:
            for rank,i in enumerate(indices,1):
                rep_rows.append(dict(cluster_id=c,role=role,rank=rank,document_id=docs.iloc[i].document_id,split=docs.iloc[i].split,distance=float(distances[i,c]),text=docs.iloc[i].text,url=docs.iloc[i].url))
    pd.DataFrame(top_rows).to_csv(OUT/'top_words.tsv',sep='\t',index=False)
    pd.DataFrame(rep_rows).to_csv(OUT/'representative_documents.tsv',sep='\t',index=False)
    pd.DataFrame({'term':final['vectorizador'].get_feature_names_out()}).to_csv(OUT/'vocabulario.tsv',sep='\t',index=False)
    pd.DataFrame({'document_id':docs.document_id,'split':docs.split,'text_normalized':normalized_all}).to_csv(OUT/'corpus_normalizado_final.tsv',sep='\t',index=False)
    for split in ['train','dev','test']:
        mask = docs.split.eq(split).to_numpy()
        sparse.save_npz(MOD/f'X_words_{split}.npz',Xwords[mask])
    print('Finalizado. Experimentos:',len(records),'Ganador:',winner['experiment_id'],flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
