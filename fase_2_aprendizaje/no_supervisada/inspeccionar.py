"""Interfaz de consulta de métricas, grupos y asignación de noticias nuevas."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'soporte'))
from normalizacion import NormalizadorNoticias  # Necesario para deserializar el pipeline.

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cluster',type=int,help='Ver términos y noticias del grupo indicado.')
parser.add_argument('--texto',help='Asignar una noticia nueva (título + descripción) al modelo final.')
args = parser.parse_args()
if args.texto:
    model = joblib.load(HERE/'modelos/kmeans_pipeline.joblib')
    print('Cluster:',int(model.predict([args.texto])[0]))
elif args.cluster is not None:
    words = pd.read_csv(HERE/'resultados/top_words.tsv',sep='\t')
    reps = pd.read_csv(HERE/'resultados/representative_documents.tsv',sep='\t')
    selected = words[words.cluster_id.eq(args.cluster)]
    if selected.empty:
        parser.error('El cluster no existe. Consulta primero el resumen sin argumentos.')
    print('Términos:',', '.join(selected.term))
    for _, row in reps[reps.cluster_id.eq(args.cluster)].iterrows():
        print(f'\n{row.role}: {row.document_id} ({row.split})\n{row.text}\n{row.url}')
else:
    print((HERE/'resultados/seleccion_dev.json').read_text())
    print(pd.read_csv(HERE/'resultados/metricas_finales.tsv',sep='\t').to_string(index=False))
    print('\nTamaños por partición:')
    print(pd.read_csv(HERE/'resultados/cluster_sizes.tsv',sep='\t').to_string(index=False))
