# Solar-Flare-Prediction

From the selection of feature sets, the combination of the feature sets FS-I, FS-II, FS-IV resulted in the best True Skill Score. Descriptions for each feature set are provided:

**Current state (FS-I)**: Baseline magnetic properties of the active solar regions.<br/>
**Temporal evolution (FS-II)**: Changes in these properties over time, allowing for pattern recognition in the lead-up to a solar flare.<br/>
**Max-Min variability (FS-IV)**: The magnitude of fluctuations, which could indicate magnetic instability in the region.<br/>

<img width="1612" alt="image" src="https://github.com/user-attachments/assets/dedb67ac-223a-4ce6-a1f1-99d0cfd2d2f3">

**Cross-Validation and Performance Evaluation**: <br/>
Utilizes k-fold cross-validation to ensure reliable evaluation of the SVM model. For each fold, the model is trained and tested on different data splits, minimizing overfitting and ensuring consistent performance. The model's performance is assessed using the True Skill Score (TSS), which is particularly suited for imbalanced data, and confusion matrices are generated to analyze True Positives, False Positives, True Negatives, and False Negatives.

**Experiments**: <br/>
Two experiments are conducted: feature experimentation and data experimentation. In the feature experiment, different combinations of FS-I, FS-II, FS-III, and FS-IV are tested to find the best-performing feature set based on TSS scores and confusion matrices. In the data experiment, the best feature set is used to evaluate the model on datasets from 2010-2015 and 2020-2024, assessing the model's generalizability across different time periods of solar activity.
