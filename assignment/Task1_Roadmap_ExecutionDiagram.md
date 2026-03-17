# Task 1 Roadmap Execution Diagram

## Full Task 1 Flow

```mermaid
flowchart TD
    A["Task 1 Goal<br/>Classify paintings by Style, Genre, Artist"] --> B["Step 1<br/>Load ArtGAN split files"]
    B --> C["Step 2<br/>Connect split rows to real WikiArt images"]
    C --> D["Step 3<br/>Create small working subset"]
    D --> E["Step 4<br/>Build baseline model<br/>ResNet18"]
    E --> F["Step 5<br/>Train and evaluate baseline"]
    F --> G["Step 6<br/>Build CNN + RNN model<br/>ResNet18 + GRU"]
    G --> H["Step 7<br/>Train and evaluate CNN + RNN"]
    H --> I["Step 8<br/>Compare both models"]
    I --> J["Step 9<br/>Error analysis / outliers"]
    J --> K["Step 10<br/>Write interpretation"]
    K --> L["Repeat same pipeline for Genre"]
    L --> M["Repeat same pipeline for Artist"]
```

## Current Progress View

```mermaid
flowchart LR
    A["Load Style Data"] --> B["Check Images"]
    B --> C["Build Baseline"]
    C --> D["Train Baseline"]
    D --> E["Build CNN + RNN"]
    E --> F["Train CNN + RNN"]
    F --> G["Compare Models"]
    G --> H["Confusion Matrix + Outliers"]
    H --> I["Write Summary"]

    A:::done
    B:::done
    C:::done
    D:::done
    E:::done
    F:::done
    G:::current
    H:::next
    I:::next

    classDef done fill:#cfeccf,stroke:#2f6f2f,color:#111;
    classDef current fill:#ffe7a8,stroke:#9a6b00,color:#111;
    classDef next fill:#d9e8ff,stroke:#2f5ea8,color:#111;
```

## Simple Meaning of the Diagram

- `Done` means the step is already completed for `Style`
- `Current` means this is the main thing to finish now
- `Next` means these are the immediate follow-up steps

## What To Do Next

1. Compare baseline vs CNN + RNN cleanly
2. Run confusion matrix
3. Inspect a few wrong predictions / outliers
4. Write a short interpretation
5. Then repeat the same notebook flow for `Genre`
6. Then repeat the same notebook flow for `Artist`
