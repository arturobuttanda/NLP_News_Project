"""Comprueba integridad de entregables y equivalencia del modelo recargado."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'soporte'))
from normalizacion import NormalizadorNoticias


def main():
    out = HERE/'resultados'
    docs = pd.read_csv(out/'labeled_documents.tsv',sep='\t',keep_default_na=False)
    results = pd.read_csv(out/'clustering_results.tsv',sep='\t')
    assert results.document_id.is_unique
    assert list(zip(docs.document_id,docs.split)) == list(zip(results.document_id,results.split))
    sizes = pd.read_csv(out/'cluster_sizes.tsv',sep='\t')
    assert sizes.total.sum() == len(docs) == 4645
    for split in ['train','dev','test']:
        assert sizes[split].sum() == docs.split.eq(split).sum()
    words = pd.read_csv(out/'top_words.tsv',sep='\t')
    reps = pd.read_csv(out/'representative_documents.tsv',sep='\t')
    assert words.groupby('cluster_id').size().eq(10).all()
    assert reps[reps.role.eq('representativo')].groupby('cluster_id').size().eq(2).all()
    for row in reps.itertuples():
        assert results.loc[results.document_id.eq(row.document_id),'cluster_id'].item() == row.cluster_id
        assert row.split != 'test'
    exp = pd.read_csv(out/'experiments.tsv',sep='\t')
    winner = exp[exp.selected]
    assert len(winner)==1
    admissible = exp[exp.eligible]
    assert winner.silhouette.item() == admissible.silhouette.max()
    assert set(exp.representation) == {'binaria','frecuencia','tfidf'}
    assert set(exp.normalization) == {'minima','sin_stopwords','lemas'}
    assert exp.n_init.nunique() >= 3 and exp['init'].nunique() == 2 and exp.max_iter.nunique() >= 3
    model = joblib.load(HERE/'modelos/kmeans_pipeline.joblib')
    # Una muestra estratificada por partición verifica la persistencia sin volver a evaluar test.
    sample = docs[docs.split.ne('test')].groupby('split').head(12)
    predictions = model.predict(sample.text)
    expected = results.set_index('document_id').loc[sample.document_id,'cluster_id'].to_numpy()
    assert np.array_equal(predictions,expected)
    # Vocabulario final sólo contiene términos observados en train+dev (el ajuste final).
    seen = set()
    analyzer = model['vectorizador'].build_analyzer()
    train_dev = docs[docs.split.ne('test')]
    for text in model['normalizacion'].transform(train_dev.text):
        seen.update(analyzer(text))
    assert set(model['vectorizador'].vocabulary_).issubset(seen)
    metrics = pd.read_csv(out/'metricas_finales.tsv',sep='\t')
    for split in ['train','dev','test']:
        mask = docs.split.eq(split)
        row = metrics[metrics.split.eq(split)].iloc[0]
        assert np.isclose(row.ari,adjusted_rand_score(docs.loc[mask,'category'],results.loc[mask,'cluster_id']))
        assert np.isclose(row.nmi,normalized_mutual_info_score(docs.loc[mask,'category'],results.loc[mask,'cluster_id']))
    message = f'OK: {len(exp)} experimentos; {len(docs)} asignaciones; {len(sizes)} grupos; 10 términos y 2 noticias por grupo; selección en dev, persistencia y métricas externas verificadas.'
    (out/'validacion.txt').write_text(message+'\n',encoding='utf-8')
    print(message)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
