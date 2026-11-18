# Proyecto PLN
### - Hernandez Espinoza Ricardo
### - Espinoza Diaz Diana Karen
### - Robles Lozano Byron
### - Saenz Buttanda Hector Arturo
### - Zurita Salinas Diamanda Abigail
El proyecto esta separado en dos areas independientes:

```text
2. Normalizacion/
│   ├── codigo/              # Pipeline y dependencias
│   ├── data/raw/            # Corpus de entrada
│   └── resultados/          # Corpus normalizado
3. Benchmark/
│   ├── codigo/              # Recuperacion, modelos y evaluacion
│   ├── data/                # Consultas y juicios
│   ├── resultados/           # Modelos, rankings y metricas
│   └── main.py               # Punto de entrada del sistema
```

La documentacion especifica de cada etapa se encuentra en
`2. Normalizacion\codigo\README.md` y
`3. Benchmark\codigo\README.md`. Esos archivos describen las columnas y el
contenido de cada entrada y salida.

## Instalacion

```powershell
pip install -r "2. Normalizacion\codigo\requirements.txt"
python -m spacy download en_core_web_sm
```

El benchmark usa las mismas dependencias y tiene su propia lista en
`3. Benchmark\codigo\requirements.txt`.

## Flujo de trabajo

Desde la raiz del proyecto:

```powershell
python "2. Normalizacion\codigo\normalize_corpus.py"
python "3. Benchmark\codigo\vector_space_models.py"
python "3. Benchmark\codigo\benchmark.py" --help
python "3. Benchmark\main.py" search --query "artificial intelligence regulation"
python "3. Benchmark\main.py" benchmark
```

La normalizacion lee `2. Normalizacion\data\raw` y escribe en
`2. Normalizacion\resultados`. El benchmark lee sus consultas y juicios desde
`3. Benchmark\data` y guarda sus salidas en `3. Benchmark\resultados`.
Los archivos de resultados previos se conservaron dentro de
`3. Benchmark\resultados\evidence`.
Las salidas activas del benchmark son `resultados\rankings.tsv`,
`resultados\evaluation_results.tsv` y `resultados\comparison.tsv`; las
carpetas por configuración contienen los rankings y métricas de cada
experimento. `REPORT_STAGE_1.md` es un informe histórico y no sustituye las
tablas TSV regeneradas por el código.

Los juicios del benchmark son asistidos por IA y deben revisarse antes de la
entrega.
