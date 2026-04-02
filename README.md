Classification de Tissus Cancéreux — Deep Learning sur PathMNIST


🎯 Contexte & Problématique

Dataset : PathMNIST — benchmark MedMNIST, issu du NCT-CRC-HE-100K. Patches histopathologiques colorectaux 28×28 RGB, 9 classes de tissus.
Pertinence clinique : La classification automatique de tissus histologiques peut assister les pathologistes dans le diagnostic du cancer, quantifier la composition tumorale et accélérer l'analyse de lames entières.
Classes :
LabelTissuLabelTissu0Adipose5Smooth Muscle1Background6Normal Mucosa2Debris7Cancer Stroma3Lymphocytes8Tumor Epithelium4Mucus

🏗️ Architecture du Pipeline

[PathMNIST]         [Exploration]      [Modélisation]             [Analyse]
107 180 images  →   EDA + stats    →   1. MLP Baseline        →   Grad-CAM (hooks)
28×28 RGB           Distribution        2. CNN from scratch        Comparaison finale
9 classes           Pixels stats        3. ResNet-18 Transfer      Bootstrap CI 95%
Train/Val/Test      Q1.1 / Q1.2         4. ViT from scratch        Per-class F1
                                        TimeSeriesSplit n/a         Confusion Matrix

🗂️ Structure du Dépôt

deep-learning-pathmnist/
│
├── notebook/
│   └── DeepLearning_PathMNIST.ipynb   # Notebook complet (100 cellules, tous outputs)
│
├── slides/
│   └── deep_learning_project.pdf      # Sujet du projet
│
├── requirements.txt
└── README.md

📥 Le dataset PathMNIST est téléchargé automatiquement via medmnist au lancement du notebook.


🔬 Étapes du Projet

1️⃣ Data Exploration

107 180 images · 9 classes · dataset très équilibré (ratio max/min = 1.64)
Statistiques d'intensité par canal R, G, B vs statistiques ImageNet
Analyse visuelle Debris vs Background (texture hétérogène vs fond uniforme)

2️⃣ MLP Baseline

Architecture : Dense 3 couches + BatchNorm + Dropout
Entraînement : 20 epochs, batch_size=256
Validation loss : 2.693 (epoch 1) → 1.817 (epoch 10) → 1.520 (epoch 20)
Confusion principale : Tumor Epithelium → Normal Mucosa (421 cas)

3️⃣ CNN from Scratch

3 blocs Conv+BatchNorm+ReLU+Dropout2d+MaxPool
1 701 673 paramètres entraînables
Premier bloc : Conv2d(3, 32, kernel_size=3) → 864 paramètres (32×3×3×3)
Augmentations histologie : RandomHorizontalFlip · RandomVerticalFlip · RandomRotation(90°)
Gap train/val : ne dépasse jamais 15pp (BatchNorm + Dropout suffisent)
Résultat inattendu : le CNN sans augmentation surpasse le CNN avec augmentation

4️⃣ Transfer Learning — ResNet-18

Redimensionnement 28×28 → 224×224 (interpolation bicubique)
Expérience A : backbone gelé, seule la tête entraînée
Expérience B : fine-tuning complet du réseau

5️⃣ Vision Transformer from Scratch

Patch embedding + CLS token learnable + positional embeddings learnable
nn.TransformerEncoder — 4 layers, 4 heads, embed_dim=128
815 753 paramètres (2.09× moins que le CNN)
Ablation positional embeddings : -3.99pp d'accuracy

6️⃣ Grad-CAM Interpretability

Implémentation manuelle via hooks PyTorch (sans librairie externe)
Forward hook → activations dernière conv · Backward hook → gradients
Visualisation sur 4 types de tissus : Normal Mucosa, Tumor Epithelium, Debris, Lymphocytes
Analyse d'un faux négatif : Tumor Epithelium prédit comme Lymphocytes


📊 Résultats

Tableau Comparatif Final
ModèleParamètresTest AccuracyObjectifMLP Baseline—57.06 %≥ 55 % ✅CNN sans augmentation1 701 67385.11 %≥ 75 % ✅CNN avec augmentation1 701 67377.64 %≥ 75 % ✅ResNet-18 frozen512 (tête)87.46 %≥ 85 % ✅ResNet-18 fine-tune ⭐11 M91.68 %≥ 85 % ✅ViT patch=7815 75381.03 %—ViT patch=14815 75373.89 %—ViT sans pos. emb.815 75377.04 %—
🏆 Meilleur Modèle : ResNet-18 Fine-tune
ComparaisonValeurResNet frozen vs Fine-tune+4.22 pp en faveur du fine-tuneViT patch=7 vs patch=14+7.13 pp en faveur de patch=7ViT avec vs sans pos. emb.+3.99 pp en faveur des pos. emb.CNN sans aug. vs avec aug.+7.46 pp en faveur du sans augmentation
Premier Epoch ≥ 50% Validation Accuracy
ModèleEpochMLP2CNN (aug)1ResNet fine-tune1ViT patch=71

🛠️ Stack Technique

CoucheOutilsDeep LearningPyTorch · torchvisionDatasetMedMNIST (medmnist) — PathMNISTArchitecturesMLP · CNN · ResNet-18 · ViT (from scratch)InterprétabilitéGrad-CAM (implémentation manuelle via hooks PyTorch)ÉvaluationConfusion matrix · F1-Score per class · Bootstrap CI 95%VisualisationMatplotlib · SeabornEnvironnementGPU CUDA · Google Colab / Jupyter

🚀 Lancement

bashpip install -r requirements.txt
jupyter notebook notebook/DeepLearning_PathMNIST.ipynb
Le dataset PathMNIST est téléchargé automatiquement à la première exécution :
pythonPathMNIST(split="train", download=True)

📦 Dépendances

torch
torchvision
medmnist
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

💡 Compétences Démontrées

Deep Learning PyTorch : MLP, CNN, ResNet, ViT — implémentation et entraînement complets
Transfer Learning : fine-tuning partiel vs complet, gestion du mismatch de résolution (28→224)
Vision Transformer from scratch : patch embedding, CLS token, positional embeddings, ablation study
Grad-CAM manuel : implémentation via hooks PyTorch sans librairie externe
Évaluation rigoureuse : Bootstrap CI 95%, F1 per-class, matrices de confusion, courbes d'apprentissage
Analyse critique : résultats inattendus documentés (augmentation contre-productive, ablation pos. emb.)


📌 Pistes d'Amélioration

 Tester un ViT pré-entraîné (ViT-B/16) pour comparer au ResNet-18 fine-tuné
 Ajouter du label smoothing classique (ε=0.1) pour réduire l'overconfidence
 Explorer des augmentations spécifiques à l'histologie (ColorJitter, stain normalization)
 Déployer le meilleur modèle via une API d'inférence (FastAPI + ONNX)
