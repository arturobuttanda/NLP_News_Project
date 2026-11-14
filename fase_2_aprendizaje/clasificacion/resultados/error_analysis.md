# 🔍 Análisis Cualitativo de Errores en Validación (Dev)

**Modelo evaluado:** `LinearSVC` (Experimento `C239`)

**Tasa de error global en Dev:** 26.54% (185 errores de 697 documentos)

## Casos de Error Inspeccionados en Detalle

### Caso #1: Documento `N0006`
- **Categoría Real (Gold):** `politica`
- **Categoría Predicha:** `economia`
- **Fuente:** BBC News
- **Texto:** *"Banks expect to be hit with higher taxes in Budget Senior banking executives have told the BBC they expect to be "on the menu" in the chancellor's Budget tax la..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://bbc.co.uk/news/articles/c32l4k2xdvrzo](https://bbc.co.uk/news/articles/c32l4k2xdvrzo)

### Caso #2: Documento `N0040`
- **Categoría Real (Gold):** `politica`
- **Categoría Predicha:** `salud`
- **Fuente:** The Guardian
- **Texto:** *"Yvette Cooper pledges £140m to boost urgent care as NHS faces ‘tough’ winter Health secretary says government has been preparing early and will spend money on a..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://theguardian.com/society/2026/oct/18/yvette-cooper-140m-boost-urgent-care-nhs-tough-winter](https://theguardian.com/society/2026/oct/18/yvette-cooper-140m-boost-urgent-care-nhs-tough-winter)

### Caso #3: Documento `N0120`
- **Categoría Real (Gold):** `politica`
- **Categoría Predicha:** `mundo_conflictos`
- **Fuente:** CBS News
- **Texto:** *"Trump waives tariffs on largest buyers of Russian oil President Trump prevented some new sanctions on Russia and hefty tariffs on large-scale buyers of Russian ..."*
- **Diagnóstico de la Causa:** Intersección geopolítica: Decisiones de líderes políticos en contextos de conflicto o guerra.
- **Enlace:** [https://cbsnews.com/news/trump-waives-sanctions-tariffs-russian-oil-lindsey-graham](https://cbsnews.com/news/trump-waives-sanctions-tariffs-russian-oil-lindsey-graham)

### Caso #4: Documento `N0176`
- **Categoría Real (Gold):** `politica`
- **Categoría Predicha:** `cultura_espectaculos`
- **Fuente:** The Hindu
- **Texto:** *"Actor Imran Khan moves Bombay HC against ‘arbitrary, illegal’ notice to leave India Notice followed politically motivated complaint, led to violation of Section..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://thehindu.com/news/national/maharashtra/actor-imran-khan-moves-bombay-hc-against-arbitrary-illegal-notice-to-leave-india/article71597273.ece](https://thehindu.com/news/national/maharashtra/actor-imran-khan-moves-bombay-hc-against-arbitrary-illegal-notice-to-leave-india/article71597273.ece)

### Caso #5: Documento `N0180`
- **Categoría Real (Gold):** `politica`
- **Categoría Predicha:** `sociedad`
- **Fuente:** The Hindu
- **Texto:** *"ABVP accuses Left outfits of defacing Vivekananda poster at Presidency University; probe panel formed The university has asked the five-member panel to submit f..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://thehindu.com/news/national/west-bengal/abvp-accuses-left-outfits-of-defacing-vivekananda-poster-at-presidency-university-probe-panel-formed/article71588385.ece](https://thehindu.com/news/national/west-bengal/abvp-accuses-left-outfits-of-defacing-vivekananda-poster-at-presidency-university-probe-panel-formed/article71588385.ece)

### Caso #6: Documento `N0227`
- **Categoría Real (Gold):** `economia`
- **Categoría Predicha:** `tecnologia_ia`
- **Fuente:** BBC News
- **Texto:** *"I find it quite demoralising in store - why I do all my shopping online After Primark announced plans for home delivery, we hear from women who never go into cl..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://bbc.co.uk/news/articles/c36lk280kd1o](https://bbc.co.uk/news/articles/c36lk280kd1o)

### Caso #7: Documento `N0286`
- **Categoría Real (Gold):** `economia`
- **Categoría Predicha:** `cultura_espectaculos`
- **Fuente:** The Guardian
- **Texto:** *"Homes with striking interiors for sale in England – in pictures From a brick stable block renovated into an eco home to a Georgian townhouse painted in atmosphe..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://theguardian.com/money/gallery/2026/oct/16/homes-with-striking-interiors-for-sale-in-england](https://theguardian.com/money/gallery/2026/oct/16/homes-with-striking-interiors-for-sale-in-england)

### Caso #8: Documento `N0460`
- **Categoría Real (Gold):** `tecnologia_ia`
- **Categoría Predicha:** `cultura_espectaculos`
- **Fuente:** Ars Technica
- **Texto:** *"Plex now helps sell movie theater tickets Discovery data from Plex users will help studios, advertisers...."*
- **Diagnóstico de la Causa:** Predominio léxico de términos característicos de cultura_espectaculos sobre la categoría asignada por el feed (tecnologia_ia).
- **Enlace:** [https://arstechnica.com/gadgets/2026/10/plex-now-helps-sell-movie-theater-tickets](https://arstechnica.com/gadgets/2026/10/plex-now-helps-sell-movie-theater-tickets)

### Caso #9: Documento `N0470`
- **Categoría Real (Gold):** `tecnologia_ia`
- **Categoría Predicha:** `economia`
- **Fuente:** The Verge
- **Texto:** *"SpaceX’s ambitious, expensive gamble to replace AT&T, T-Mobile, and Verizon Today, I’m talking with Walter Piecyk, a partner and analyst at LightShed Partners w..."*
- **Diagnóstico de la Causa:** Economía política: Medidas regulatorias, presupuestos estatales o tratados comerciales.
- **Enlace:** [https://theverge.com/podcast/1012416/spacex-starlink-elon-musk-cell-carrier-wireless-mobile-att-verizon-t-mobile](https://theverge.com/podcast/1012416/spacex-starlink-elon-musk-cell-carrier-wireless-mobile-att-verizon-t-mobile)

### Caso #10: Documento `N0475`
- **Categoría Real (Gold):** `tecnologia_ia`
- **Categoría Predicha:** `ciencia`
- **Fuente:** Wired
- **Texto:** *"The Best LED Grow Lights for Houseplants This Winter (2026) Shorter days don’t have to spell trouble for your houseplants. These are the lights we’ve used and l..."*
- **Diagnóstico de la Causa:** Frontera difusa: Noticias de tecnología basada en investigación científica o IA.
- **Enlace:** [https://wired.com/story/best-led-grow-lights](https://wired.com/story/best-led-grow-lights)

### Caso #11: Documento `N0481`
- **Categoría Real (Gold):** `tecnologia_ia`
- **Categoría Predicha:** `deportes`
- **Fuente:** Wired
- **Texto:** *"The Leap Second Is No More The metric might not mean much to laypeople, but leap seconds have been a source of contention in the timekeeping community for years..."*
- **Diagnóstico de la Causa:** Predominio léxico de términos característicos de deportes sobre la categoría asignada por el feed (tecnologia_ia).
- **Enlace:** [https://wired.com/story/leap-second-is-abolished-bipm-says](https://wired.com/story/leap-second-is-abolished-bipm-says)

### Caso #12: Documento `N0495`
- **Categoría Real (Gold):** `tecnologia_ia`
- **Categoría Predicha:** `medio_ambiente`
- **Fuente:** Wired
- **Texto:** *"Egypt’s Climate-Resilient ‘Super Corals’ Are Starting to Feel the Heat Scientists can’t agree why reefs in the Red Sea have long been immune to bleaching. The a..."*
- **Diagnóstico de la Causa:** Ecología científica: Reportes sobre cambio climático con terminología biológica y de física.
- **Enlace:** [https://wired.com/story/egypts-climate-resilient-super-corals-are-starting-to-feel-the-heat](https://wired.com/story/egypts-climate-resilient-super-corals-are-starting-to-feel-the-heat)
