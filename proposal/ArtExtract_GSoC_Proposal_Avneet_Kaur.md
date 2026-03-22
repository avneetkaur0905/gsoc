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

I have already completed the two pre-selection assignment tasks that directly demonstrate my readiness for this project. Task 1 implemented and compared two deep-learning architectures — a fine-tuned ResNet-18 baseline CNN and a CNN+RNN (ResNet-18 + GRU) — for multi-label art classification on the WikiArt dataset. Task 2 built a full visual similarity retrieval system for the National Gallery of Art open-access collection using ResNet-18 embeddings and FAISS search, achieving a medium-consistency score of 0.412 at approximately 2× the random baseline.

During the GSoC period, I will extend these foundations into a unified six-stage pipeline: scaling classification and retrieval, and building a novel multi-spectral underpainting detection system incorporating a ResNet-50 binary detector, Grad-CAM explainability, Kubelka-Munk physics-based reconstruction, and a diffusion model refinement stage — producing a publicly reproducible tool for digital art conservation.

---

## 4. Motivation and Background

Hidden paintings — underpaintings and pentimenti concealed beneath the visible surface of famous artworks — represent one of the most fascinating frontiers in digital art conservation. Art historians have long known that masters such as Picasso, Rembrandt, and Van Eyck frequently painted over earlier compositions; X-ray and infrared reflectography occasionally reveal these hidden layers. However, physical imaging is expensive, requires specialist equipment costing tens of thousands of dollars, and cannot be performed remotely or at scale.

At the same time, large digitised museum collections (WikiArt, the National Gallery of Art, the Met Open Access) now contain tens of thousands of high-resolution public-domain paintings, making large-scale computational analysis both feasible and reproducible for the first time.

My two assignment solutions demonstrate that the core building blocks — robust visual feature extraction from painting images, scalable similarity search, and multi-attribute classification — already work well at moderate scale. The GSoC project packages these into a single open-source pipeline that cultural heritage researchers, digital humanities scholars, and museum technologists can run, extend, and cite.

---

## 5. Assignment Work Completed

### Task 1 — Art Classification with CNN and CNN+RNN

For Task 1, I implemented and compared two deep learning architectures for classifying WikiArt paintings across Style (27 classes), Artist (23 classes), and Genre (10 classes). The baseline CNN used ResNet-18 fully fine-tuned with a global average pooling head, while the proposed CNN+RNN unfolded the 7×7 spatial feature map into a sequence of 49 patch vectors processed by a single-layer GRU. Both models were trained with Adam + cosine annealing and standard augmentation (random flip, colour jitter).

The baseline CNN outperformed the CNN+RNN across all three tasks: Artist 81.4% Top-1 / 79.1% Macro F1, Genre 72.9% / 67.9%, and Style 47.4% / 36.6%, compared to the CNN+RNN's Artist 63.9% / 59.5%, Genre 69.3% / 61.9%, and Style 39.2% / 24.1%. All results substantially exceeded random baselines (Style ~3.7%, Artist ~4.3%, Genre ~10%). The largest gap appeared on Artist recognition, where fine-grained local texture statistics — characteristic brushstroke frequency, colour signature, palette distribution — are captured efficiently by global average pooling but disrupted by the GRU's sequential processing. Macro F1 was chosen as the primary metric to guard against class-imbalance artifacts in the heavily skewed WikiArt distribution.

### Task 2 — Painting Similarity Retrieval

For Task 2, I built a full visual similarity retrieval system for the NGA open-access collection. ResNet-18 (ImageNet-pretrained, classification layer removed) was used to extract 512-dimensional embeddings from painting images, L2-normalised for cosine similarity, and indexed with FAISS IndexFlatIP. The system was evaluated on 50 embedded paintings drawn from the 2,888-painting NGA open-access metadata.

In the absence of ground-truth similarity labels, I designed two proxy metrics grounded in art-historical domain knowledge: medium consistency @5 (what fraction of the top-5 retrieved results share the query's physical medium) and mean temporal gap @5 (mean absolute year difference between query and retrieved results). The system achieved medium consistency @5 of 0.412 (~2× the random baseline of ~0.2) and a mean temporal gap of 89.3 years, confirming that the embedding space carries meaningful art-historical signal. Portrait-filtered retrieval (restricted to the 809 portrait-classified paintings) produced improved compositional consistency, demonstrating natural generalisation to subject-filtered queries.

---

## 6. Proposed Work for GSoC Period

Building on the assignment foundations, I propose four interconnected work packages structured around a six-stage ArtExtract pipeline.

### Work Package 1 — Scale and Improve Classification (Weeks 1–4)

Extend Task 1 from the 5,000-image training subset to the full WikiArt corpus (~57k Style training images, ~45k Genre, ~13k Artist). With the full dataset, I will implement weighted sampling or focal loss to address class imbalance (particularly for rare styles like Mannerism and Art Informel), explore contrastive pre-training (SupCon loss) on painting pairs to build a representation where "same style" is meaningful before the classification head is attached, and scale the backbone to ResNet-50 to test whether deeper features close the CNN+RNN gap on Style. The trained models will be packaged as a reusable `ArtClassifier` Python module with a clean API.

**Deliverable:** Trained and evaluated classification models (ResNet-18 and ResNet-50) on full WikiArt, with per-class and macro F1 scores documented. Model checkpoints and evaluation code published to repository.

### Work Package 2 — Scale and Improve Similarity Retrieval (Weeks 3–7)

Extend Task 2 from the 50-painting embedded sample to all 2,888 NGA open-access paintings. I will batch download and cache all IIIF thumbnail images, embed all 2,888 paintings, and upgrade the FAISS index from IndexFlatIP to IndexIVFFlat for O(log n) query time at scale. I will add contrastive fine-tuning of the embedder using medium, artist, and time period as soft similarity signals (SimCLR or SupCon), and implement hybrid retrieval combining ResNet-18 visual embeddings with CLIP text embeddings for semantically-guided queries. A Gradio interface will allow art historians to query by image upload without writing code.

**Deliverable:** Full 2,888-painting FAISS index, contrastively fine-tuned embedder, hybrid retrieval module, and a simple query interface.

### Work Package 3 — Underpainting Detection Pipeline (Weeks 5–11)

This is the novel module that goes beyond the assignment tasks. I will build a complete six-stage pipeline for detecting and reconstructing hidden underpaintings, as described in detail in Section 8 below. The six stages are: (1) synthetic multispectral dataset construction, (2) ResNet-50 binary detector, (3) Grad-CAM explainability, (4) Kubelka-Munk physics reconstruction, (5) diffusion model refinement, and (6) full pipeline integration.

**Deliverable:** Complete six-stage detection pipeline with automated per-painting report (YES/NO decision, confidence score, spatial heat map, channel importance chart, rough reconstruction, and refined final reconstruction). Open-source dataset construction script, trained model checkpoints, and Gradio demo.

### Work Package 4 — Integration, Documentation, and Community Release (Weeks 10–13)

Integrate WP1 (classification), WP2 (retrieval), and WP3 (detection) into a unified `artextract` Python package with a consistent API. I will write comprehensive documentation including installation guides, API reference, and tutorial notebooks for each module, publish the synthetic multispectral dataset to Zenodo or HuggingFace Datasets under a CC-BY licence, and submit a short technical report to arXiv describing the pipeline, datasets, and evaluation results.

**Deliverable:** PyPI-publishable `artextract` package, full documentation site, published dataset, and arXiv preprint.

---

## 7. Timeline

| Week | Dates | Milestones |
|---|---|---|
| 1–2 | May 27 – Jun 9 | Environment setup; WP1 full-dataset training begins; baseline CNN on full WikiArt Style |
| 3–4 | Jun 10 – Jun 23 | WP1 complete (all three tasks, ResNet-18 + ResNet-50, full evaluation); WP2 full NGA batch download begins |
| 5–6 | Jun 24 – Jul 7 | WP2 full 2,888-painting FAISS index complete; contrastive fine-tuning starts; WP3 Stage 1 dataset construction |
| 7 | Jul 8 – Jul 14 | **Midterm evaluation** — WP1 fully delivered; WP2 index and baseline retrieval delivered; WP3 dataset and Stage 2 detector training complete |
| 8–9 | Jul 15 – Jul 28 | WP2 hybrid retrieval and Gradio interface; WP3 Stage 3 Grad-CAM and Stage 4 Kubelka-Munk reconstruction |
| 10–11 | Jul 29 – Aug 11 | WP3 Stage 5 diffusion model training and refinement; Stage 6 full pipeline integration and automated report generator |
| 12 | Aug 12 – Aug 18 | WP4 integration into unified `artextract` package; documentation writing |
| 13 | Aug 19 – Aug 25 | Final testing, bug fixes, dataset publication, arXiv draft |
| **Final** | Aug 26 | **Final evaluation** — all deliverables submitted |

---

## 8. Technical Approach and Design Decisions

The core contribution of this GSoC project is a six-stage pipeline for hidden underpainting detection and reconstruction. Each stage is described below in full technical detail.

### Stage 1 — Synthetic Multispectral Dataset Construction

Real paintings with documented underpaintings are rare and not available at scale in digital form. The solution is to construct a large synthetic dataset by downloading 500–1,000 public-domain paintings from WikiArt and the NGA open-access collection, then randomly pairing them to simulate a hidden layer beneath a top layer. Each pair is blended at three opacity ratios (80/20, 90/10, and 70/30 top/hidden) using PIL `Image.blend()`, producing three training examples per pair and yielding approximately 1,200 positive (hidden layer present) examples. Unpaired single paintings are retained as negative examples, producing a balanced ~2,400-example dataset.

Every image in the dataset — blended and single alike — is converted from standard RGB to a 15-channel simulated multispectral array and saved as a `.npy` file. The 15 channels capture different aspects of the painting's visual signal: RGB (channels 0–2), HSV (channels 3–5), LAB colour space (channels 6–8), greyscale (channel 9), Canny edge detection (channel 10), Gaussian blur (channel 11), texture via local binary patterns (channel 12), FFT frequency magnitude (channel 13), and depth-from-shading (channel 14). This multispectral representation simulates the kind of multi-band imaging data that museum conservators collect using specialist cameras, and gives the detection model far more information than RGB alone. The dataset is split 80/20 into train and test sets, with separate CSV label files recording filename and YES/NO ground truth for each split.

### Stage 2 — ResNet-50 Binary Detector

The detection model is a ResNet-50 with one critical modification: the standard 3-channel `conv1` is replaced with a 15-channel convolution whose weights are initialised by averaging the original 3-channel weights across the new 15 input positions. This is a standard and well-established technique for adapting pretrained CNNs to multispectral input, preserving the learned low-level feature detectors while accepting the expanded channel count.

Training uses Adam at a learning rate of 0.0001, batch size 16, and 30 epochs with CrossEntropyLoss for the binary YES/NO task. A data loader reads the labels CSV, loads `.npy` files as PyTorch tensors with shape (batch, 15, 256, 256), and shuffles randomly each epoch. Training loss should decrease steadily; model checkpoints are saved every 5 epochs. After training, the model is evaluated on the held-out test set using accuracy, precision, recall, and F1 score (targets: above 85% accuracy, above 0.80 F1). The best checkpoint is saved as `best_detection_model.pth`.

### Stage 3 — Grad-CAM Explainability

A binary YES/NO decision alone is insufficient for conservation use. A museum conservator needs to know *where* in the painting the hidden layer was detected, and *which imaging channels* provided the strongest evidence. Grad-CAM (Gradient-weighted Class Activation Mapping) answers both questions by computing, for every pixel area in the image, how much the YES confidence score would change if that area were perturbed.

Using `pytorch-grad-cam` targeting `model.layer4` (the last convolutional layer of ResNet-50), the system produces a 256×256 importance map for each YES prediction. This raw map is converted to a TURBO-colourmap heat map — red and bright yellow indicate areas of strong evidence, blue indicates areas the model ignored — and blended onto the original painting at 60/40 (painting/heat map) weighting. The output is a three-panel comparison figure showing the original painting, the heat map alone, and the overlay together.

Beyond spatial localisation, the system also computes per-channel importance by blanking each of the 15 input channels in turn and measuring how much the YES score drops. Channels producing the largest score drop when blanked are the most informative. A horizontal bar chart of channel importances tells conservators which imaging modality (edge detection, texture, FFT frequency, etc.) to prioritise in real camera equipment. All results for YES predictions in the test set are saved in an organised `gradcam_results/` folder, with one subfolder per painting containing the comparison image, channel chart, and overlay.

### Stage 4 — Kubelka-Munk Physics Reconstruction

Stage 4 begins the job of actually showing what the hidden painting looks like. The Kubelka-Munk optical model describes how light interacts with stacked paint layers: some light reflects from the top layer surface, some passes through and reflects off the hidden layer, and a small fraction returns through the top layer creating faint traces of the hidden composition. Given the blended multispectral image B, the top-layer image T, and paint absorption and scattering constants K = 0.5 and S = 1.0, the reverse Kubelka-Munk formula recovers the hidden layer H pixel-by-pixel across all 15 channels:

```
ratio = K / S
R_top = 1 + ratio − √(ratio² + 2·ratio)
H = (B − R_top × T) / (1 − R_top × T),  clipped to [0, 1]
```

NumPy vectorised operations apply this formula to the entire 256×256×15 array simultaneously, making computation near-instantaneous. The result is a rough 15-channel reconstruction: the main shapes and layout of the hidden painting should be recognisable, though the output will appear blurry and colour-imperfect — as expected from a physics model that cannot account for pigment interactions not captured by K and S. The RGB channels are extracted and saved as a preview PNG for visual quality control; the full 15-channel array is saved as a `.npy` file for Stage 5.

### Stage 5 — Diffusion Model Final Reconstruction

Stage 5 takes the rough Kubelka-Munk output and refines it into a sharp, detailed, visually clear image. A diffusion model learns to run a denoising process in reverse: during training, known amounts of Gaussian noise are progressively added to training images over 1,000 timesteps, and the model learns at each timestep to predict what noise was added. Once trained, the model can reverse this process — starting from a noisy image and iteratively removing noise over 1,000 steps to recover a clean result.

For this project, the model begins from the rough Kubelka-Munk reconstruction rather than from pure random noise. Because the reconstruction already has the correct shapes and layout (just blurry and colour-imperfect), the model starts from a far more meaningful initial state than random static, producing significantly more accurate final reconstructions. Training data consists of matched pairs of rough reconstructions (input) and actual hidden paintings from the synthetic dataset (target). Using HuggingFace Diffusers, the model is trained at learning rate 0.0001, batch size 4, and 50 epochs with 1,000 timesteps, saving checkpoints every 10 epochs. The final model is evaluated on held-out test reconstructions using SSIM (structural similarity, target above 0.7), PSNR (pixel accuracy, target above 25 dB), and LPIPS (perceptual quality, target below 0.2). Three-panel comparison figures (original painting | rough reconstruction | final refined output) make the contribution of this stage visually self-evident.

### Stage 6 — Full Pipeline Integration

Stage 6 connects all five previous stages into a single seamless system. A user provides any painting image; the pipeline automatically runs all stages and returns a complete analysis report without any manual steps. The complete flow is:

| Stage | Input | Output | Tool |
|---|---|---|---|
| 1. Preprocessing | Regular painting image | 15-channel .npy file | NumPy + OpenCV |
| 2. Detection | 15-channel .npy file | YES/NO + confidence score | Trained ResNet-50 |
| 3. Grad-CAM | 15-channel .npy + YES result | Heat map overlay + channel importance | pytorch-grad-cam |
| 4. KM Reconstruction | 15-channel .npy + top layer .npy | Rough 15-channel reconstruction | Kubelka-Munk formula |
| 5. Diffusion Refinement | Rough reconstruction .npy | Clean final reconstruction | Trained diffusion model |
| 6. Output Report | All above results | Professional PDF report | Python + matplotlib |

If the detector returns NO, the pipeline stops at Stage 2 and reports no hidden layer found. If YES, all remaining stages run automatically. The final output report includes three pages: a detection summary (original painting, YES/NO decision, confidence percentage), a Grad-CAM explanation (three-panel comparison, channel importance chart, plain-language summary of where evidence was found), and a reconstruction page (three-panel comparison of original, rough reconstruction, and final cleaned output with SSIM/PSNR/LPIPS scores). This report format is designed so that a museum conservator with no programming background can read and act on it immediately — which is what makes this pipeline genuinely useful rather than merely technically interesting.

### Why ResNet-18 and ResNet-50

Both assignment solutions used ResNet-18 because ImageNet-pretrained ResNet features are already rich visual descriptors — the early layers detect edges, colour regions, and fine textures that are exactly the visual primitives that distinguish oil-on-canvas impasto from tempera on panel, or the loose brushwork of Impressionism from the precise contours of Neoclassicism. For the detection pipeline, ResNet-50 is used as the backbone because deeper features are better suited to the subtle, multi-channel signal of partially visible hidden layers. The 15-channel input modification (averaging original weights across new channels) is a standard technique that preserves pretrained low-level features while accepting multispectral input.

### Why FAISS and Cosine Similarity

FAISS IndexFlatIP performs exact inner product search over all indexed vectors. For a 50-painting demo, this is computationally trivial, but the design is deliberately forward-compatible: switching to IndexIVFFlat is a one-line change that reduces query time from O(n) to O(log n) for 100k+ paintings. Cosine similarity (implemented as L2-normalised inner product) is preferred over L2 distance because it is scale-invariant — two paintings can produce high-magnitude embeddings from high-contrast images and appear dissimilar under L2 even when their visual patterns are structurally identical, whereas cosine similarity measures the angle between vectors regardless of magnitude.

### Why Synthetic Data

Real paintings with documented underpaintings are rare and not publicly available in digital form at scale. Synthetic blending of public-domain painting pairs is a well-established approach in art-analysis research (cf. Art2Real, Tomei et al.) and has the significant advantage of producing perfectly labelled training data with known ground truth. The three opacity ratios (70/30, 80/20, 90/10) span the range from visually obvious to nearly invisible hidden layers, ensuring the model learns to detect the full range of blend intensities encountered in real conservation scenarios.

---

## 9. Related Work

ArtExtract sits at the intersection of three active research areas.

**Large-scale art classification:** Saleh and Elgammal conducted the first large-scale fine-art classification study, establishing that CNN features transfer remarkably well to the painting domain. The WikiArt dataset (accessed via the ArtGAN repository) has become the standard benchmark for style, artist, and genre classification, with reported top-1 accuracy on Style of ~50–60% for models trained on the full dataset. The BAM! Behance Artistic Media Dataset demonstrated that object recognition models trained purely on photography fail on artistic imagery — a direct motivation for synthesising domain-specific training data.

**Explainability in art analysis:** Understanding what features a model uses to classify a painting is as important as the classification itself — especially in a conservation context where decisions guide expensive physical imaging. IntroStyle presented a training-free style attribution framework built on diffusion model internal features; Khan et al. analysed the creative space using WikiArt and CLIP embeddings, finding that convolutional features encode meaningful stylistic axes. My Grad-CAM approach extends this tradition to the binary detection task, producing spatially localised explanations that conservation professionals can interpret and act on.

**Domain translation and retrieval:** Tomei et al.'s Art2Real is the closest existing work to ArtExtract's reconstruction objective — a semantics-aware GAN that translates artworks into photo-realistic visualisations. For retrieval, FAISS-based embedding search has been applied to museum collections at the Rijksmuseum and Met, typically using CLIP or ResNet features. The novelty here is the combination of proxy-metric evaluation appropriate for unlabelled art collections with a contrastive fine-tuning step using art-historical metadata as weak supervision, and the integration of a physics-based reconstruction step (Kubelka-Munk) with learned diffusion model refinement.

---

## 10. Benefits to the Community

**Museum conservators and art historians** will gain a freely available, reproducible tool to screen large painting collections for possible underpaintings before committing to expensive physical imaging. The automated report provides a first-pass triage that directs specialist attention to the most promising candidates, making AI-assisted conservation accessible without specialist programming knowledge.

**Digital humanities researchers** will receive three reusable artefacts: a well-documented synthetic multispectral dataset construction pipeline, trained classification and retrieval models that can be fine-tuned for new collections, and a public evaluation benchmark for painting similarity with expert-annotated ground-truth pairs.

**CERN-HSF / HumanAI** gains a working open-source implementation of a complete art-analysis pipeline — classification, retrieval, and detection — that can serve as infrastructure for follow-on projects, student assignments, and community contributions.

**The broader open-source community** benefits from a PyPI-publishable `artextract` package, comprehensive documentation, and reproducible Jupyter notebooks covering every component of the pipeline.

---

## 11. Prior Experience and Relevant Skills

**Demonstrated in the assignment submissions:** PyTorch — designed, trained, and evaluated two architectures (BaselineCNN and CNNRNN) from scratch, including custom Dataset classes, DataLoaders, training loops with Adam + cosine annealing, best-checkpoint saving, and full evaluation (Top-1, Top-3, Macro F1, confusion matrices). Transfer learning — ResNet-18 full fine-tuning for classification; ResNet-18 feature extraction for retrieval; principled decisions about when to freeze versus fine-tune backbone layers. FAISS — built an IndexFlatIP similarity index, performed k-NN retrieval, and designed proxy evaluation metrics appropriate for unlabelled museum data. Data engineering — merged multi-file NGA CSVs (144k objects, 130k image records), handled parsing inconsistencies, built reproducible sampling with fixed random seeds, and designed offline-first image caching. Evaluation design — recognised class-imbalance problems in WikiArt and chose Macro F1 as primary metric; designed domain-appropriate proxy metrics for unlabelled retrieval data; identified and visualised failure cases and outlier predictions.

**Additional skills relevant to GSoC deliverables:** Experience with `pytorch-grad-cam` for explainability visualisation; familiarity with Kubelka-Munk optical scattering models for paint-layer physics; familiarity with HuggingFace Diffusers for diffusion model training; Python packaging (setuptools, pyproject.toml), documentation (MkDocs / Sphinx), and version control (Git / GitHub).

---

## 12. Stretch Goals

If core deliverables are completed ahead of schedule:

- **Multi-task learning:** Train Style, Artist, and Genre simultaneously with a shared ResNet backbone and three task heads, exploiting the fact that knowing an artist constrains plausible styles, and vice versa.
- **Real multispectral input:** Extend the pipeline to accept genuine multispectral camera inputs (replacing simulated channels with real sensor measurements), enabling partnerships with conservation labs.
- **Ground-truth similarity benchmark:** Recruit domain-expert annotation for a small set of known similar pairs (copies, studio variants, same-theme works by different artists) to replace the proxy metrics with proper Precision@k and Recall@k.
- **Extended pigment modelling:** Replace the fixed K = 0.5, S = 1.0 Kubelka-Munk constants with pigment-specific values from published conservation databases, improving reconstruction accuracy for well-documented historical pigments.

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
