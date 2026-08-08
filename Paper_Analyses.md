# 📄 Comprehensive Research Paper Analysis
## AI-Powered Medical Report Analyzer & Disease Risk Predictor

Below is the detailed breakdown of all 22 papers used in this project, analyzed individually with their Abstract, Literature Review context, Insights, and Research Gaps.

---

## 🔵 Group 1: Multimodal Fusion (Image + Text + Clinical Data)

### 1. PMID: 37362656 - Multimodal medical tensor fusion network-based DL framework for abnormality prediction from the radiology CXRs and clinical text reports
**Abstract:** Investigated multimodal medical fusion strategies leveraging DL techniques to predict pulmonary abnormality from heterogeneous radiology Chest X-Rays (CXRs) and clinical text reports. Proposed unimodal and multimodal subnetworks. Multimodal models gave superior results over unimodal models.
**Literature Review Context:** Belongs to the early wave of combining unstructured image data with unstructured clinical text for pulmonary disease detection, moving away from single-modality vision models.
**Insights:** Fusing text and imaging significantly improves diagnostic accuracy compared to using either modality alone. Tensor fusion networks are highly effective at capturing cross-modal interactions.
**Research Gaps:** The study only focuses on pulmonary abnormalities. It does not address missing modalities (what if the text report is absent?) and lacks deep cross-modal explainability to show clinicians exactly why a prediction was made.

### 2. PMID: 42424829 - MedFusionNet: A hybrid transformer-based multimodal deep learning framework for chronic disease prediction in women's health
**Abstract:** Proposes MedFusionNet, combining medical images and clinical reports for women's health conditions (breast cancer, cervical cancer, PCOS). Uses EfficientNetB0, ViT, and a text transformer with cross-modal influence fusion mechanism.
**Literature Review Context:** Represents the state-of-the-art in applying hybrid transformer architectures (ViT + text transformers) to highly specific, multi-disease predictive frameworks within a specialized medical domain (women's health).
**Insights:** Two-stage training (fusion/classification first, fine-tuning second) yields extreme accuracy (>96%). Visual patterns are better interpreted when contextualized by textual descriptions.
**Research Gaps:** Restricted entirely to women's health. The model architecture is highly specialized and lacks a generalized approach for broader respiratory or metabolic diseases (like Pneumonia or Diabetes). No handling of missing tabular/lab data.

### 3. PMID: 40576670 - Early prediction of adverse outcomes in liver cirrhosis using a CT-based multimodal deep learning model
**Abstract:** Developed a deep learning-based triple-modal fusion liver cirrhosis network (TMF-LCNet) for the prediction of adverse outcomes using CT images, lab data, and clinical notes.
**Literature Review Context:** One of the few recent papers expanding beyond bimodal (image+text) to triple-modal fusion (image+text+tabular), specifically for predicting adverse clinical outcomes.
**Insights:** Adding tabular lab data to image and text pipelines creates a much stronger predictive tool for long-term adverse outcomes in chronic conditions like cirrhosis.
**Research Gaps:** Focuses solely on liver cirrhosis. The fusion technique is rigid; if lab data is missing, the triple-modal network cannot function properly. Lacks comprehensive explainability across the three modalities.

### 4. PMID: 38396486 - An Innovative and Efficient Diagnostic Prediction Flow for Head and Neck Cancer: A Deep Learning Approach for Multi-Modal Survival Analysis Prediction
**Abstract:** Proposed a multi-modal image-text fusion strategy using cross-attention for PET and CT images, and Q-former architecture for text and image fusion to improve Head and Neck Cancer prognosis.
**Literature Review Context:** Highlights the use of Q-former architectures (popularized by BLIP-2) in the medical domain to align textual clinical data with multi-center 3D imaging (PET/CT) for survival analysis.
**Insights:** Introducing time as a variable in multimodal models significantly improves survival analysis metrics (MFS, RFS, OS). Q-formers are highly effective for medical image-text alignment.
**Research Gaps:** Highly complex and computationally heavy due to 3D PET/CT processing. Not suited for real-time or edge deployment. The focus is strictly on cancer survival rather than general disease risk prediction.

### 5. PMID: 42310728 - Lung cancer multimodal auxiliary diagnosis based on entropy weight decision fusion
**Abstract:** Proposes a lung cancer multimodal auxiliary diagnosis model based on entropy weight decision fusion to overcome poor feature alignment caused by simply concatenating CT images and clinical text.
**Literature Review Context:** Addresses the "concatenation flaw" in early multimodal AI, where simply joining image and text vectors led to suboptimal learning, proposing mathematical weighting (entropy) instead.
**Insights:** Entropy weight decision fusion dynamically balances the importance of image vs. text features, preventing one dominant modality from suppressing the other during training.
**Research Gaps:** The entropy weighting is done at the decision level (late fusion), missing the deep feature interactions (early/intermediate fusion) that attention mechanisms provide.

### 6. PMID: 41957440 - K-STAMM: a knowledge-enhanced spatial - temporal attention model with multimodal fusion for pneumonia prediction
**Abstract:** Presents K-STAMM, which brings together biomedical knowledge from UMLS, attention-based spatial modeling of structured EHR, and temporal sequence modeling. Fuses CXR, clinical text, and knowledge embeddings.
**Literature Review Context:** The most advanced model for Pneumonia prediction, combining Knowledge Graphs (UMLS) with spatiotemporal EHR data and imaging.
**Insights:** Integrating structured medical knowledge (UMLS) forces the deep learning model to learn clinically relevant semantics, drastically improving AUROC (0.953) on MIMIC datasets.
**Research Gaps:** The architecture is limited to Pneumonia. Graph construction and UMLS embedding make the model extremely heavy and difficult to deploy in low-resource clinical settings.

### 7. PMID: 42141332 - MRI- and report-based multimodal model with SHAP-based explanation for preoperative prediction of deep stromal invasion in early-stage cervical cancer
**Abstract:** Aims to develop an explainable multimodal data fusion model integrating MRI, radiology reports, and clinical variables for preoperative assessment of DSI risk in cervical cancer.
**Literature Review Context:** A critical paper that explicitly attempts to solve the "black box" problem of multimodal medical AI by applying SHAP (SHapley Additive exPlanations).
**Insights:** Explaining multimodal AI is possible, and revealing feature importance builds clinical trust for preoperative surgical decisions.
**Research Gaps:** SHAP is applied to features *within* each modality, but the paper lacks a clear cross-modal attribution mechanism (e.g., showing if the MRI was mathematically more important than the report for a specific patient).

---

## 🔵 Group 2: Automated Radiology Report Generation & NLP

### 8. PMID: 41714517 - Automated Report Generation in Ophthalmology: Integrating Artificial Intelligence, Multimodal Imaging, and Clinical Data
**Abstract:** Summarizes recent advances in AI-driven report generation, emphasizing the integration of multimodal imaging (fundus, OCT) and clinical data to generate structured, personalized diagnostic reports.
**Literature Review Context:** Provides a comprehensive overview of how Generative AI (LLMs and Vision-Language Models) are taking over diagnostic reporting in specialized fields like ophthalmology.
**Insights:** Multimodal learning combined with LLMs reduces interobserver variability and streamlines clinical workflow significantly.
**Research Gaps:** Identifies data heterogeneity, model interpretability, and clinical integration as persistent challenges. Generative reports often suffer from hallucination, requiring strict validation frameworks.

### 9. PMID: 38829752 - Automated Radiology Report Generation: A Review of Recent Advances
**Abstract:** Methodological review of contemporary Automated Radiology Report Generation (ARRG) approaches, examining contrastive learning, reinforcement learning, CNN/transformer architectures, and knowledge graphs.
**Literature Review Context:** The definitive 2024/2025 survey on ARRG, establishing the baseline architectures (like DenseNet/ResNet for vision and Transformers for text) used in modern medical AI.
**Insights:** Reinforcement learning and contrastive learning are the current best approaches to force models to generate clinically accurate, rather than just linguistically fluent, reports.
**Research Gaps:** The paper notes that current evaluation metrics (like BLEU/ROUGE) are inadequate for clinical text. It highlights the need for models that can ingest multiple types of radiological modalities simultaneously.

### 10. PMID: 37162253 - Transformer versus traditional natural language processing: how much data is enough for automated radiology report classification?
**Abstract:** Compares state-of-the-art transformer deep-learning architectures against traditional NLP techniques, hypothesizing that traditional NLP may outperform transformers on smaller radiology report datasets.
**Literature Review Context:** A critical pragmatic study challenging the "transformers are always better" narrative, specifically in data-constrained medical environments.
**Insights:** Transformers require massive datasets to excel; for small, highly specific clinical datasets, traditional ML/NLP techniques (like TF-IDF + SVM) can achieve comparable or better results with less computational cost.
**Research Gaps:** Did not test lightweight or distilled transformers (like MobileBERT). The gap remains in finding highly efficient, low-data transformer architectures for niche medical tasks.

### 11. PMID: 42085853 - Artificial intelligence language models for medical text analysis: A systematic review
**Abstract:** Systematic review of AI-driven language models (BERT, GPT) for analyzing, classifying, and generating medical textual data. Shows they outperform conventional ML but face challenges in validation and interpretability.
**Literature Review Context:** A broad review confirming BERT and GPT as the undisputed standards for processing EHRs and clinical notes in 2025.
**Insights:** Pre-trained clinical models (like ClinicalBERT) drastically outperform generic models. The primary barrier to adoption is not accuracy, but clinical trust and explainability.
**Research Gaps:** Concludes that future research *must* prioritize hybrid AI systems (multimodal data sources) and incorporate explainable AI mechanisms—directly validating the premise of your MTech project.

### 12. PMID: 41933671 - Automated Detection and Classification of Radiology Report Discrepancies Using NLP: A Tool for Resident Education and Quality Assurance
**Abstract:** Developed an NLP system to automatically detect and classify discrepancies between preliminary and final radiology reports for resident education and quality assurance.
**Literature Review Context:** Focuses on the administrative and educational applications of clinical NLP rather than direct diagnosis.
**Insights:** NLP can reliably understand clinical semantics well enough to detect subtle human errors and discrepancies in medical narratives.
**Research Gaps:** The system is purely text-to-text. It cannot look at the actual X-ray to determine *which* report (preliminary or final) is actually visually correct.

---

## 🔵 Group 3: EHR + Explainable AI + Risk Prediction

### 13. PMID: 40911160 - Interpretable Semi-federated Learning for Multimodal Cardiac Imaging and Risk Stratification: A Privacy-Preserving Framework
**Abstract:** Introduces PerFed-Cardio, a lightweight semi-federated learning system for real-time cardiovascular risk stratification using multimodal data (cardiac imaging, physiological signals, EHR) with LIME and Grad-CAM for transparency.
**Literature Review Context:** Combines edge computing, federated privacy, multimodal fusion, and XAI into a single advanced framework for cardiac care.
**Insights:** High-capacity nodes (hospitals) and edge devices (wearables) can collaboratively train models. Achieved incredible inference latency (130ms) suitable for real-time use.
**Research Gaps:** The federated setup is complex to deploy. While it addresses privacy and latency, it assumes all data streams (wearables + EHR + imaging) are constantly available, ignoring missing modality scenarios.

### 14. PMID: 42078832 - Explainable machine learning for predicting disease flares in axial spondyloarthritis: A real-world electronic health record-based pilot study
**Abstract:** Developed and internally validated a machine learning model to forecast disease flares 3 to 12 months ahead using routinely collected electronic health record (EHR) data.
**Literature Review Context:** A strong example of using longitudinal, tabular EHR data for temporal risk forecasting, enhanced by explainable AI.
**Insights:** Routine, structured EHR data is highly predictive of long-term disease flares when processed by ML, and XAI helps clinicians understand the longitudinal risk factors.
**Research Gaps:** Purely tabular. Does not incorporate unstructured clinical notes or medical imaging, leaving significant predictive data on the table.

### 15. PMID: 41322476 - Evaluating XAI techniques under class imbalance using CPRD data
**Abstract:** Investigates the reliability of post-hoc XAI techniques (LIME, SHAP, PDPs) under real-world scenarios of class imbalance in tabular Electronic Health Record (EHR) data.
**Literature Review Context:** A critical methodological paper that exposes the vulnerabilities of popular explainability tools in healthcare settings.
**Insights:** XAI techniques like SHAP can produce inconsistent or misleading feature importance scores when trained on highly imbalanced medical data (where the disease class is a minority).
**Research Gaps:** Exposes a massive gap: We need XAI frameworks that are mathematically robust to class imbalance, particularly in multimodal settings where imbalance affects different modalities differently.

### 16. PMID: 41699573 - Machine learning and artificial intelligence for delirium prediction with Electronic Health Records (EHR): a scoping review
**Abstract:** Scoping review of ML models for delirium prediction using EHRs. Highlights reliance on structured data, limited use of unstructured narratives, and methodological variations posing challenges to generalizability.
**Literature Review Context:** Summarizes the state of predictive modeling for neurocognitive disorders in hospital settings.
**Insights:** The medical field is over-reliant on structured (tabular) data because it is easy to process, ignoring the rich, highly informative unstructured clinical notes.
**Research Gaps:** Explicitly calls for the development of integrated multimodal fusion models adaptable to dynamic patient states and emphasizes the need for models that handle incomplete/missing EHR data.

### 17. PMID: 41402001 - Integrating machine learning models to assess the combined risk of diabetes and tuberculosis in populations
**Abstract:** Suggests using ML models to assess the combined risk of diabetes and tuberculosis. Uses logistic regression, random forest, XGBoost, and deep learning with a multitask learning structure.
**Literature Review Context:** Directly validates the multi-disease aspect of your project by highlighting the clinical importance of predicting comorbid conditions (Diabetes + TB).
**Insights:** Multitask learning structures are highly effective at modelling the shared risk factors between two distinct diseases simultaneously.
**Research Gaps:** The study is mostly focused on population-level tabular data. It does not integrate medical imaging (chest X-rays for TB) into the multitask framework, which would vastly improve diagnostic accuracy.

### 18. PMID: 40828572 - Deep Learning and Image Generator Health Tabular Data (IGHT) for Predicting Overall Survival in Patients With Colorectal Cancer
**Abstract:** Transforms tabular electronic medical record (EMR) data into structured 2D image matrices, enabling the use of computer vision-based deep learning models for survival prediction.
**Literature Review Context:** Represents a novel, unconventional approach to multimodal learning—converting tabular data into an image format to utilize powerful CNN architectures.
**Insights:** Spatial encoding of clinical tabular features allows CNNs to capture complex, non-linear interactions among clinical variables that standard tabular ML (like XGBoost) might miss.
**Research Gaps:** While clever, this approach destroys the inherent structure and interpretability of lab values. A doctor cannot easily interpret a "pixel" that represents a blood glucose level.

---

## 🔵 Group 4: Specialized Imaging AI

### 19. PMID: 38381447 - Deep Learning and Machine Learning Algorithms for Retinal Image Analysis in Neurodegenerative Disease: Systematic Review of Datasets and Models
**Abstract:** Reviews deep learning models used for automated neurodegenerative disease diagnosis (Alzheimer's, Parkinson's) and risk prediction using retinal images.
**Literature Review Context:** Shows how deep learning is expanding beyond traditional radiology (X-rays/MRIs) to use accessible imaging (retinal scans) as biomarkers for systemic diseases.
**Insights:** The eye acts as a window to the brain; CNNs can detect micro-vascular changes in retinal images that correlate highly with cognitive decline years before symptoms appear.
**Research Gaps:** Mostly relies on single-modality retinal images. Integrating these images with cognitive clinical test scores (text/tabular) is noted as a necessary next step.

### 20. PMID: 39405390 - Patient-Specific Myocardial Infarction Risk Thresholds From AI-Enabled Coronary Plaque Analysis
**Abstract:** Deep learning provides automated quantification of coronary plaque from CT angiography to determine age- and sex-specific distributions for myocardial infarction risk prediction.
**Literature Review Context:** Focuses on using AI not just to "detect" a disease, but to quantify a continuous biomarker (plaque volume) and generate personalized risk thresholds.
**Insights:** AI is highly effective at precise, tedious volumetric segmentation tasks that humans struggle to perform consistently, translating pixels directly into personalized risk scores.
**Research Gaps:** The AI operates purely on the imaging physics. It does not ingest the patient's lifestyle data, clinical history, or genetic profile, limiting the holistic accuracy of the risk threshold.

### 21. PMID: 36442416 - Predicting time-to-conversion for dementia of Alzheimer's type using multi-modal deep survival analysis
**Abstract:** Used a deep-learning model for survival analyses to predict time-to-conversion to Dementia of Alzheimer's Type using MRI, genetic, and cognitive test data.
**Literature Review Context:** A landmark paper demonstrating how different modalities are dominant at different stages of a disease (e.g., genetics are predictive early, cognitive tests predictive later).
**Insights:** Combining MRI and genetic features improved prediction over single modalities. Interestingly, adding cognitive test data to any combination only worked as well as using cognitive data alone for late-stage subjects.
**Research Gaps:** The architecture does not dynamically weigh the modalities; it relies on manual feature concatenation. It highlights the need for attention mechanisms that automatically shift weight to the most informative modality depending on the patient's disease stage.

### 22. PMID: 41311632 - Federated learning in computational pathology: a literature review
**Abstract:** Examines the current state of the art in the application of Federated Learning (FL) within the healthcare domain, specifically focusing on computational pathology to mitigate privacy risks.
**Literature Review Context:** Addresses the massive data bottleneck in training medical AI: hospitals cannot legally share high-resolution pathology images due to patient privacy laws.
**Insights:** FL allows multiple hospitals to train a shared global model without ever moving raw patient data out of their local firewalls, successfully preserving privacy while achieving high accuracy.
**Research Gaps:** FL in pathology suffers from extreme data heterogeneity (different hospitals use different scanners and staining techniques). The review notes a severe lack of FL frameworks that can handle multimodal data (e.g., pathology images + local EHR text).
