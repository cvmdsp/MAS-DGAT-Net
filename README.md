# MAS-DGAT-Net
MAS-DGAT-Net: A Dynamic Graph Attention Network with Multibranch Feature Extraction and Staged Fusion for EEG Emotion Recognition
**Abstract**
In recent years, with the rise of deep learning technologies, EEG-based emotion recognition has garnered significant attention. However, most existing methods tend to focus on the spatiotemporal information of EEG signals while overlooking the potential topological information of brain regions. To address this issue, this paper proposes a dynamic graph attention network with multi-branch feature extraction and staged fusion (MAS-DGAT-Net), which integrates graph convolutional neural networks (GCN) for EEG emotion recognition. Specifically, the differential entropy (DE) features of EEG signals are first reconstructed into a correlation matrix using the Spearman correlation coefficient. Then, the brain-region connectivity-feature extraction (BCFE) module is employed to capture the brain connectivity features associated with emotional activation states. Meanwhile, this paper introduces a dual-branch cross-fusion feature extraction (CFFE) module, which consists of an attention-based cross-fusion feature extraction branch (A-CFFEB) and a crossfusion feature extraction branch (CFFEB). A-CFFEB efficiently extracts key channel-frequency information from EEG features by using an attention mechanism and then fuses it with the output features from the BCFE. The fused features are subsequently input into the proposed dynamic graph attention module with a broad learning system (DGAT-BLS) to mine the brain connectivity feature informationfurther. Finally, the deep features output by DGAT-BLS and CFFEB are combined for emotion classification. The proposed algorithm has been experimentally validated on SEED, SEED-IV, and DEAP datasets in subject-dependent and subject-independent settings, with the results confirming the model's effectiveness. The source code is publicly available at: https://github.com/cvmdsp/MAS-DGAT-Net.
