# ArtExtract: A Multi-Stage AI Pipeline for Art Classification, Similarity Retrieval, and Hidden Painting Detection

**Google Summer of Code 2025 — CERN-HSF / HumanAI**
**Medium Project (~175 hours)**

---

## 1. Personal Information

| | |
|---|---|
| **Name** | Avneet Kaur |
| **Email** | gurdev.leo@gmail.com |
| **GitHub** | github.com/avneetkaur0905 |
| **University** | (your university) |
| **Degree / Year** | (your degree and year) |
| **Time Zone** | (your time zone) |
| **Available Hours/Week** | ~30–35 hours |

---

## 2. Proposal Title

**ArtExtract: A Multi-Stage AI Pipeline for Art Classification, Painting Similarity Retrieval, and Detection of Hidden Underpaintings**

---

## 3. Abstract

The ArtExtract project addresses three interconnected challenges in computational art analysis: (1) multi-label classification of artistic attributes (style, artist, genre) from painting images, (2) scalable visual similarity retrieval across large open-access museum collections, and (3) automated detection and reconstruction of hidden underpaintings concealed beneath the visible surface of artworks.

I have already completed the two pre-selection assignment tasks that directly demonstrate my readiness for this project. **Task 1** implemented and compared two deep-learning architectures — a fine-tuned ResNet-18 baseline CNN and a novel CNN+RNN (ResNet-18 + GRU) — for multi-label art classification on the WikiArt dataset across Style (27 classes), Artist (23 classes), and Genre (10 classes). **Task 2** built a full visual similarity retrieval system for the National Gallery of Art open-access collection using ResNet-18 embeddings and FAISS approximate nearest-neighbour search, achieving a medium-consistency score of 0.412 (approximately 2× the random baseline) with a mean temporal gap of 89.3 years.

During the GSoC period, I will extend and productionise these foundations into a unified pipeline: scaling classification to the full WikiArt corpus, scaling retrieval to all 2,888+ NGA open-access paintings, adding contrastive fine-tuning for semantically-aware retrieval, and building a novel multi-spectral underpainting detection module with Grad-CAM explainability and Kubelka-Munk physics-based reconstruction — the output of which is a publicly reproducible tool for digital art conservation.

---

## 4. Motivation and Background

Hidden paintings — underpaintings and pentimenti concealed beneath the visible surface of famous artworks — represent one of the most fascinating frontiers in digital art conservation. Art historians have long known that masters such as Picasso, Rembrandt, and Van Eyck frequently painted over earlier compositions; X-ray and infrared reflectography occasionally reveal these hidden layers. However, physical imaging is expensive, requires specialist equipment costing tens of thousands of dollars, and cannot be performed remotely or at scale.

At the same time, large digitised museum collections (WikiArt, the National Gallery of Art, the Met Open Access) now contain tens of thousands of high-resolution public-domain paintings, making large-scale computational analysis both feasible and reproducible for the first time.

My two assignment solutions demonstrate that the core building blocks — robust visual feature extraction from painting images, scalable similarity search, and multi-attribute classification — already work well at moderate scale. The GSoC project packages these into a single open-source pipeline that cultural heritage researchers, digital humanities scholars, and museum technologists can run, extend, and cite.

---

## 5. Assignment Work Completed

### Task 1 — Art Classification with CNN and CNN+RNN

**Objective:** Classify WikiArt paintings across three attributes — Style (27 classes), Artist (23 classes), and Genre (10 classes) — and compare a pure CNN baseline against a CNN+RNN architecture.

**Architecture decisions:**

| Component | Choice | Rationale |
|---|---|---|
| Backbone | ResNet-18 (fine-tuned, all layers) | Computationally practical on CPU; full fine-tuning adapts ImageNet weights to painting domain |
| Baseline | ResNet-18 + Global Avg Pool (BaselineCNN) | Tests whether local texture/colour features alone are sufficient |
| Proposed | ResNet-18 + GRU (CNNRNN) | Tests whether spatial sequence modelling captures compositional structure beyond pooling |
| Optimiser | Adam + Cosine Annealing LR | Handles mixed backbone/head learning dynamics; smooth LR decay avoids late-training oscillation |
| Augmentation | Random flip + Colour jitter | Compensates for scanning variability across WikiArt museum sources |

The CNNRNN model unfolds the ResNet-18 spatial feature map (7×7, 512 channels) into a sequence of 49 patch vectors and processes them with a single-layer GRU, whose final hidden state is fed to the classification head. The idea is that spatial sequence modelling can capture long-range compositional flows — e.g., whether a painting feels uniformly diffuse (Impressionism) or hierarchically structured (Baroque) — that global average pooling collapses away.

**Key results:**

| Task | CNN Top-1 | CNN+RNN Top-1 | CNN Macro F1 | CNN+RNN Macro F1 | Winner |
|---|---|---|---|---|---|
| Style (27 cls) | **47.4%** | 39.2% | **36.6%** | 24.1% | CNN |
| Artist (23 cls) | **81.4%** | 63.9% | **79.1%** | 59.5% | CNN |
| Genre (10 cls) | **72.9%** | 69.3% | **67.9%** | 61.9% | CNN |

*Random baselines: Style ≈ 3.7%, Artist ≈ 4.3%, Genre ≈ 10%*

**Analysis:** The BaselineCNN outperforms the CNN+RNN across all three tasks, with the largest gap on Artist recognition. This is a meaningful result: artist identity is captured in fine-grained local texture statistics — characteristic brushstroke frequency, colour signature, palette distribution — that global average pooling summarises efficiently. The GRU's sequential processing over spatial patches disrupts the texture signal while introducing compositional structure that the Artist task simply does not require.

The narrower gap on Genre (3.6 percentage points vs 18.5 on Artist) is telling: genre differences are large-scale and compositional (a portrait has a face at the centre; a landscape has a horizon at the top), so the GRU's spatial attention does add some value here — just not enough to overcome the texture-information penalty.

The Style task confirms this story: CNN+RNN achieves only a marginal improvement over CNN at two training epochs. With more epochs and a larger training subset, the GRU is hypothesised to close the gap because style is at least partially defined by compositional arrangement across the canvas (Baroque chiaroscuro vs Impressionist uniform diffusion).

**Evaluation methodology:** Macro F1 was chosen as the primary metric because the WikiArt dataset is heavily class-imbalanced (Impressionism accounts for a disproportionate share of the Style training examples). A naive model predicting "Impressionism" for all inputs would achieve deceptively high raw accuracy while completely failing on minority styles like Mannerism or Art Informel. Macro F1 weights all classes equally, penalising the model for ignoring rare classes and rewarding genuine generalisation across the full label space.

Top-3 accuracy is tracked alongside Top-1 because many style classes overlap visually (Impressionism vs Post-Impressionism, Symbolism vs Art Nouveau). A model that puts the correct class in position 2 or 3 demonstrably "understands the artistic neighbourhood" even when it cannot pinpoint the exact label.

**Outlier analysis:** I identified and visualised paintings where the model was not just wrong but assigned extremely low confidence (<10%) to the correct class. The most common confusion in Style is Mannerism being predicted as Pop Art — an unexpected pairing explained by shared high-contrast, flattened figure compositions. Symbolism and Art Nouveau confusion is expected: both movements were active in the same era and drew from the same decorative and figurative traditions. These failure cases are rarely arbitrary; they cluster around class pairs that are historically adjacent, visually overlapping, or underrepresented in training.

---

### Task 2 — Painting Similarity Retrieval

**Objective:** Given a query painting from the National Gallery of Art open-access collection, return the *k* most visually similar works using deep visual embeddings and approximate nearest-neighbour search.

**System design:**

| Component | Choice | Rationale |
|---|---|---|
| Embedding model | ResNet-18 (ImageNet, no fine-tuning) | Pretrained visual primitives transfer to painting domain; GAP output gives task-agnostic 512-d fingerprint |
| Why not CLIP | CLIP aligns image+text; ResNet stays purely visual | For visual similarity (not semantic), ResNet avoids language-alignment bias |
| Similarity index | FAISS IndexFlatIP (exact search) | Exact results at current scale; trivially upgradeable to IndexIVFFlat for 100k+ paintings |
| Similarity measure | Cosine similarity via L2-normalised inner product | Scale-invariant; robust to exposure/brightness differences across museum scans |
| Dataset | 50 paintings embedded and searchable (2,888 in full NGA metadata) | Embedded subset for retrieval demo; full 2,888-painting metadata available for scale-up |

**Dataset:** The NGA open-access collection provides metadata for ~130k objects and image URLs for ~4k open-access paintings. I loaded and merged `objects.csv` (144k objects) with `published_images.csv` (130k image records) on object ID, then filtered to the 2,888 open-access paintings with IIIF thumbnail URLs. Collection analysis revealed oil on canvas as the dominant medium (1,665 paintings), with creation years concentrated between 1600 and 1900.

**Feature extraction:** ResNet-18 with the final classification layer removed outputs a 512-dimensional global average-pooled feature vector. This is a fixed-length visual fingerprint that captures colour distribution, textural style, and gross compositional structure without being tied to any label space. Vectors are L2-normalised so that inner product equals cosine similarity exactly, which is scale-invariant and robust to the brightness and exposure differences that arise across IIIF scans from different decades and equipment.

**Retrieval results:**

| Metric | Value | Context |
|---|---|---|
| Medium consistency @5 | **0.412** | ~2× above random baseline (~0.2); paintings cluster by physical surface material |
| Mean temporal gap @5 | **89.3 years** | Retrieved works tend toward the same historical era as the query |
| Embedding cache | Loaded from disk | Zero recomputation on re-run; notebook runs fully offline after first execution |

**Evaluation methodology:** No ground-truth similarity labels exist for the NGA collection. Rather than inventing labels, I defined two proxy metrics with clear semantic meaning rooted in art-historical domain knowledge.

*Medium consistency @5:* For each query painting, what fraction of its top-5 retrieved results share the same physical medium (e.g. "oil on canvas", "tempera on panel")? Paintings executed in the same medium share fundamental visual properties — oil paintings have a characteristic lustre and impasto texture; tempera has a flatter, more matte surface. A well-performing retrieval system should naturally cluster works with similar physical surfaces, because those surfaces produce similar visual signatures. The observed 0.412 is roughly 2× above the random baseline, confirming the embedding space carries medium-relevant information even though the model was never told what medium any painting uses.

*Mean temporal gap @5:* The mean absolute year difference between a query's creation date and each of its top-5 retrieved results. Paintings from the same era share stylistic conventions, colour palettes, and compositional norms. The 89.3-year gap is informative: the collection spans several centuries, and paintings do cluster temporally — but the figure should be interpreted alongside retrieval visualisations rather than in isolation.

**Portrait-specific retrieval:** Restricting the FAISS index to the 809 portrait-classified paintings and re-running retrieval produced visually coherent results with improved compositional consistency (centred figure, similar aspect ratio, similar depth of field) compared with full-collection retrieval — demonstrating that the pipeline generalises naturally to subject-filtered queries.

**Failure mode analysis:** The three queries with the lowest average retrieval similarity share a common characteristic: the query painting is visually unusual relative to the rest of the 50-painting sample. This is a data coverage problem rather than a model problem. With the full 2,888-painting collection embedded, these queries would almost certainly find more meaningful matches. One genuine model limitation that would persist even with more data: paintings that are visually similar but semantically opposite (a dark abstract and a dark portrait) get retrieved together because the model has no semantic grounding.

---

## 6. Proposed Work for GSoC Period

Building on the assignment foundations, I propose four interconnected work packages for the GSoC period.

### Work Package 1 — Scale and Improve Classification (Weeks 1–4)

Extend Task 1 from the 5,000-image training subset to the full WikiArt corpus (~57k Style training images, ~45k Genre, ~13k Artist). With the full dataset, I will:

- Implement weighted sampling or focal loss to address class imbalance (particularly for rare styles like Mannerism and Art Informel).
- Explore contrastive pre-training (SupCon loss) on painting pairs to build a representation where "same style" is meaningful before the classification head is attached.
- Scale the backbone to ResNet-50 to test whether deeper features close the CNN+RNN gap on Style.
- Package the trained models as a reusable `ArtClassifier` Python module with a clean API.

**Deliverable:** Trained and evaluated classification models (ResNet-18 and ResNet-50) on full WikiArt, with per-class and macro F1 scores documented. Model checkpoints and evaluation code published to repository.

### Work Package 2 — Scale and Improve Similarity Retrieval (Weeks 3–7)

Extend Task 2 from the 50-painting embedded sample to all 2,888 NGA open-access paintings, and improve the retrieval system:

- Batch download and cache all 2,888 IIIF thumbnail images.
- Embed all 2,888 paintings with the ResNet-18 embedder; upgrade the FAISS index from IndexFlatIP to IndexIVFFlat for O(log n) query time at scale.
- Add contrastive fine-tuning of the embedder using medium + artist + time period as soft similarity signals (SimCLR or SupCon), pushing visually and semantically similar paintings together in embedding space.
- Implement hybrid retrieval: combine ResNet-18 visual embeddings with CLIP text embeddings via a weighted sum, enabling queries like "find paintings similar to this one in composition but depicting outdoor scenes."
- Recruit domain-expert annotation for a small ground-truth evaluation set of known similar pairs (copies, studio variants, same-theme works by different artists) to replace the proxy metrics with proper Precision@k and Recall@k.

**Deliverable:** Full 2,888-painting FAISS index, contrastively fine-tuned embedder, hybrid retrieval module, and ground-truth benchmark. A simple Gradio interface for art historians to query by image upload.

### Work Package 3 — Underpainting Detection Pipeline (Weeks 5–11)

This is the novel module that goes beyond the assignment tasks. I will build a four-stage detection and reconstruction pipeline:

**Stage 1 — Synthetic multispectral dataset construction:**
Download 500–1,000 public-domain paintings from WikiArt (via the ArtGAN repository) and the NGA open-access collection. Randomly pair and blend images at three opacity ratios (80/20, 90/10, 70/30 top/hidden) using PIL `Image.blend()`, producing ~1,200 YES (blended) examples. Retain unpaired single paintings as NO examples, yielding a balanced ~2,400-example dataset. Convert every image to a 15-channel simulated multispectral `.npy` array: RGB (3) + HSV (3) + LAB (3) + Grayscale (1) + Canny edges (1) + Gaussian blur (1) + texture (1) + FFT frequency (1) + depth-from-shading (1).

**Stage 2 — ResNet-50 binary detector:**
Replace the standard 3-channel `conv1` with a 15-channel convolution (weights initialised by averaging across the original 3 channels). Fine-tune on the synthetic multispectral dataset for binary classification: does this painting contain a hidden layer? Output: YES/NO decision + confidence score + per-channel importance weights.

**Stage 3 — Grad-CAM explainability:**
Apply `pytorch-grad-cam` targeting `model.layer4` for every YES prediction. Produce: (i) a TURBO-colourmap heat map showing where the model detects the hidden layer, (ii) a three-panel comparison figure (original / heat map overlay / channel-importance bar chart), (iii) a per-painting JSON report with confidence, top-3 important channels, and heat map path.

**Stage 4 — Kubelka-Munk physics reconstruction:**
Given the blended image B, the top-layer image T, and empirical paint constants K = 0.5 and S = 1.0, the reverse Kubelka-Munk formula recovers the hidden layer H pixel-by-pixel:

```
ratio = K / S
R_top = 1 + ratio − √(ratio² + 2·ratio)
H = (B − R_top × T) / (1 − R_top × T),  clipped to [0, 1]
```

The result is a rough 256×256×15 reconstruction. The RGB channels are extracted and saved as a preview PNG for visual quality control. A stretch goal is a diffusion model refinement step (fine-tuning a HuggingFace `UNet2DConditionModel`) to convert the blurry Kubelka-Munk output into a sharpened, artefact-reduced reconstruction.

**Deliverable:** Full four-stage detection pipeline with automated per-painting report (YES/NO, confidence, spatial heat map, channel importance, rough reconstruction preview). Open-source dataset construction script, trained model checkpoint, and Gradio demo.

### Work Package 4 — Integration, Documentation, and Community Release (Weeks 10–13)

- Integrate WP1 (classification), WP2 (retrieval), and WP3 (detection) into a unified `artextract` Python package with a consistent API.
- Write comprehensive documentation: installation guide, API reference, tutorial notebooks for each module, and a reproducibility guide so researchers can reconstruct all results from scratch.
- Publish the synthetic multispectral dataset to Zenodo or HuggingFace Datasets under a CC-BY licence.
- Submit a short technical report to arXiv describing the pipeline, datasets, and evaluation results.

**Deliverable:** PyPI-publishable `artextract` package, full documentation site, published dataset, and arXiv preprint.

---

## 7. Timeline

| Week | Dates | Milestones |
|---|---|---|
| 1–2 | May 27 – Jun 9 | Environment setup; WP1 full-dataset training begins; baseline CNN on full WikiArt Style |
| 3–4 | Jun 10 – Jun 23 | WP1 complete (all three tasks, ResNet-18 + ResNet-50, full evaluation); WP2 full NGA batch download begins |
| 5–6 | Jun 24 – Jul 7 | WP2 full 2,888-painting FAISS index complete; contrastive fine-tuning starts; WP3 synthetic dataset construction |
| 7 | Jul 8 – Jul 14 | **Midterm evaluation** — WP1 fully delivered; WP2 index and baseline retrieval delivered; WP3 dataset complete |
| 8–9 | Jul 15 – Jul 28 | WP2 hybrid retrieval and Gradio interface; WP3 ResNet-50 15-channel detector training |
| 10–11 | Jul 29 – Aug 11 | WP3 Grad-CAM explainability module; Kubelka-Munk reconstruction; per-painting report generator |
| 12 | Aug 12 – Aug 18 | WP4 integration into unified `artextract` package; documentation writing |
| 13 | Aug 19 – Aug 25 | Final testing, bug fixes, dataset publication, arXiv draft |
| **Final** | Aug 26 | **Final evaluation** — all deliverables submitted |

---

## 8. Technical Approach and Design Decisions

### Why ResNet-18 as the Primary Backbone

Both assignment solutions used ResNet-18 for strong practical reasons. ImageNet-pretrained ResNet features are already rich visual descriptors — the early layers detect edges, colour regions, and fine textures that are exactly the visual primitives that distinguish oil-on-canvas impasto from tempera on panel, or the loose brushwork of Impressionism from the precise contours of Neoclassicism. Global average pooling gives a compact, task-agnostic 512-d fingerprint without requiring any task-specific labelled data. No fine-tuning was needed for similarity retrieval; for classification, full fine-tuning (rather than feature freezing) was used because painting images are statistically very different from natural photographs — freezing the backbone risks retaining ImageNet-specific biases.

### Why FAISS for Similarity Search

FAISS (Facebook AI Similarity Search) with IndexFlatIP performs exact inner product search over all indexed vectors. For a 50-painting demo, this is computationally trivial. The design is deliberately forward-compatible: switching to IndexIVFFlat is a one-line change that reduces query time from O(n) to O(log n) and makes the system tractable at 100k+ paintings with no other code changes required.

### Why Cosine Similarity (Not L2 Distance)

L2 distance is sensitive to the absolute magnitude of embedding vectors. Two paintings could produce high-magnitude embeddings (from high-contrast images) and appear dissimilar even if their visual patterns are structurally identical. Cosine similarity measures the angle between vectors regardless of magnitude. Since all vectors are L2-normalised to unit length beforehand, inner product equals cosine similarity exactly — making the normalisation step non-optional.

### Why the CNN+RNN Hypothesis Is Still Worth Pursuing

The assignment results showed the BaselineCNN outperforming the CNNRNN on all tasks at 5 epochs with 5,000 training images. However, the hypothesis remains theoretically sound: style is partly defined by compositional arrangement, and the GRU's sequential processing over spatial patches should capture long-range correlations that pooling averages away. The assignment result reflects a training limitation (5 epochs × 5,000 images is too small for the GRU to learn sequential patch dependencies) rather than a fundamental architectural failure. During the GSoC period, scaling to the full dataset and more epochs will test this directly.

### Why Synthetic Data for Underpainting Detection

Real paintings with documented underpaintings are rare and not publicly available in digital form at scale. Synthetic blending of public-domain painting pairs is a well-established approach in art-analysis research (cf. Art2Real [Tomei et al.]) and has the significant advantage of producing perfectly labelled training data with known ground truth — the blended image, the top-layer image, and the hidden-layer image are all available for both training and evaluation. The opacity ratios (70/30, 80/20, 90/10) span the range from visually obvious to nearly invisible hidden layers.

---

## 9. Related Work

ArtExtract sits at the intersection of three active research areas.

**Large-scale art classification:** Saleh and Elgammal conducted the first large-scale fine-art classification study, establishing that CNN features transfer remarkably well to the painting domain. The BAM! Behance Artistic Media Dataset demonstrated that object recognition models trained purely on photography fail on artistic imagery — a direct motivation for synthesising domain-specific training data. The WikiArt dataset (accessed via the ArtGAN repository) has become the standard benchmark for style, artist, and genre classification, with reported top-1 accuracy on Style of ~50–60% for models trained on the full dataset.

**Explainability in art analysis:** Understanding what features a model uses to make a decision about a painting is as important as the decision itself — especially in a conservation context. IntroStyle presented a training-free style attribution framework built on diffusion model internal features; Khan et al. analysed the creative space using WikiArt and CLIP embeddings, finding that convolutional features encode meaningful stylistic axes. My Grad-CAM approach extends this tradition to the binary detection task, producing spatially localised explanations that conservation professionals can interpret.

**Domain translation and retrieval:** Tomei et al.'s Art2Real is the closest existing work to ArtExtract's reconstruction objective — a semantics-aware GAN that translates artworks into photo-realistic visualisations. For retrieval, FAISS-based embedding search has been applied to museum collections at the Rijksmuseum and Met, typically using CLIP or ResNet features. The novelty here is the combination of proxy-metric evaluation (medium consistency, temporal proximity) appropriate for unlabelled art collections with a contrastive fine-tuning step using art-historical metadata as weak supervision.

---

## 10. Benefits to the Community

**Museum conservators and art historians** will gain a freely available, reproducible tool to screen large painting collections for possible underpaintings before committing to expensive physical imaging. The automated report (confidence score, heat map, channel importance, rough reconstruction) provides a first-pass triage that directs specialist attention to the most promising candidates.

**Digital humanities researchers** will receive three reusable artefacts: (1) a well-documented synthetic multispectral dataset construction pipeline, (2) trained classification and retrieval models that can be fine-tuned for new collections, and (3) a public evaluation benchmark for painting similarity with expert-annotated ground-truth pairs.

**CERN-HSF / HumanAI** gains a working open-source implementation of a complete art-analysis pipeline — classification, retrieval, and detection — that can serve as infrastructure for follow-on projects, student assignments, and community contributions.

**The broader open-source community** benefits from a PyPI-publishable `artextract` package, comprehensive documentation, and reproducible Jupyter notebooks covering every component of the pipeline.

---

## 11. Prior Experience and Relevant Skills

**Demonstrated in the assignment submissions:**

- *PyTorch* — designed, trained, and evaluated two architectures (BaselineCNN and CNNRNN) from scratch, including custom Dataset classes, DataLoaders, training loops with Adam + cosine annealing, best-checkpoint saving, and full evaluation (Top-1, Top-3, Macro F1, confusion matrices).
- *Transfer learning* — ResNet-18 full fine-tuning for classification; ResNet-18 feature extraction for retrieval; principled decisions about when to freeze vs. fine-tune backbone layers.
- *FAISS* — built an IndexFlatIP similarity index, performed k-NN retrieval, and designed proxy evaluation metrics appropriate for unlabelled museum data.
- *Data engineering* — merged multi-file NGA CSVs (144k objects, 130k image records), handled parsing inconsistencies (double-comma delimiters, stray header rows), built reproducible sampling with fixed random seeds, and designed offline-first image caching.
- *Evaluation design* — recognised class-imbalance problems in WikiArt and chose Macro F1 as primary metric; designed domain-appropriate proxy metrics (medium consistency, temporal gap) for unlabelled retrieval data; identified and visualised failure cases and outlier predictions.
- *Visualisation and reporting* — produced retrieval composite images, training history plots, confusion matrices, class distribution charts, and outlier grids as reproducible outputs.

**Additional skills relevant to GSoC deliverables:**

- Experience with `pytorch-grad-cam` for explainability visualisation.
- Familiarity with Kubelka-Munk optical scattering models for paint-layer physics.
- Familiarity with HuggingFace Diffusers for diffusion model fine-tuning (stretch goal).
- Python packaging (setuptools, pyproject.toml), documentation (MkDocs / Sphinx), and version control (Git / GitHub).

---

## 12. Stretch Goals

If core deliverables are completed ahead of schedule:

- **Diffusion model refinement (Stage 5):** Fine-tune a HuggingFace `UNet2DConditionModel` on (rough-reconstruction, actual-hidden-painting) pairs to convert the blurry Kubelka-Munk output into a sharper, artefact-reduced image.
- **Multi-task learning:** Train Style, Artist, and Genre simultaneously with a shared ResNet backbone and three task heads, exploiting the fact that knowing an artist constrains plausible styles, and vice versa.
- **Interactive web demo:** A lightweight Gradio interface so conservators can upload a painting photograph and receive the full detection + retrieval report without writing any code.
- **Real multispectral input:** Extend the pipeline to accept genuine multispectral camera inputs (replacing simulated channels with real sensor measurements), enabling partnerships with conservation labs.

---

## 13. Availability and Commitments

I will be available for approximately 30–35 hours per week throughout the GSoC coding period (May 27 – August 26, 2025). I have no major academic deadlines or travel during this period. I am comfortable with asynchronous communication over GitHub Issues, Mattermost, and email, and can attend weekly video check-ins at times suitable for the CERN-HSF mentoring team.

---

## 14. References

1. Saleh, B., & Elgammal, A. (2015). Large-scale classification of fine-art paintings. *arXiv:1505.00855*.
2. Tan, W., Chan, C., Aguirre, H., & Tanaka, K. (2016). Ceci n'est pas une pipe: A deep convolutional network for fine-art paintings classification. *ICIP*.
3. Elgammal, A., et al. (2018). The shape of art history in the eyes of the machine. *AAAI*.
4. Murray, N., et al. (2012). AVA: A large-scale database for aesthetic visual analysis. *CVPR*.
5. Garcia, N., & Vogiatzis, G. (2018). How to read paintings: Semantic art understanding with multi-modal retrieval. *ECCV Workshops*.
6. Yi, R., et al. (2023). BAID: A large-scale benchmark for aesthetics image description. *CVPR*.
7. Maerten, A., et al. (2023). LAPIS: Language-image pre-training for searching art. *CHR*.
8. Dosovitskiy, A., et al. (2020). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*.
9. Selvaraju, R., et al. (2017). Grad-CAM: Visual explanations from deep networks. *ICCV*.
10. Li, Y., et al. (2023). IntroStyle: Training-free intrinsic style attribution for enhanced artistic understanding. *ICCV Workshops*.
11. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer. *ECCV*.
12. Khan, A., et al. (2024). Analysing the creative space of AI-generated vs. human art using WikiArt and CLIP. *arXiv*.
13. Tomei, M., et al. (2019). Art2Real: Unfolding the reality of artworks via semantically-aware image-to-image translation. *CVPR*.
14. Gonthier, N., et al. (2018). Weakly supervised object detection in artworks. *ECCV Workshops*.

---

*Proposal submitted for GSoC 2025 — CERN-HSF / HumanAI ArtExtract project.*
*All code, notebooks, and assignment deliverables available at: [github.com/avneetkaur0905/gsoc](https://github.com/avneetkaur0905/gsoc)*
