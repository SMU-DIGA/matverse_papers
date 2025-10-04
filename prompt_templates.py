models_data = [
    (
        "Linear Models",
        ["Linear Model", "Polynomial Model", "Generalized Linear Model"],
    ),
    (
        "Tree-based Models",
        [
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting Tree",
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ],
    ),
    (
        "Kernel-based Models",
        [
            "Support Vector Machine",
            "Gaussian Process",
            "Radial Basis Function Network",
        ],
    ),
    (
        "Probabilistic Models",
        [
            "Naive Bayes",
            "Bayesian Network",
            "Hidden Markov Model",
            "Markov Random Field",
            "Conditional Random Field",
            "Gaussian Mixture Model",
            "Latent Dirichlet Allocation",
        ],
    ),
    (
        "Basic Neural Networks",
        [
            "Perceptron",
            "Multi-Layer Perceptron",
            "Feedforward Neural Network",
            "Radial Basis Function Network",
        ],
    ),
    (
        "Convolutional Neural Networks",
        [
            "Convolutional Neural Network",
            "LeNet",
            "AlexNet",
            "VGG",
            "ResNet",
            "Inception",
            "DenseNet",
            "MobileNet",
            "EfficientNet",
            "SqueezeNet",
            "ResNeXt",
            "SENet",
            "NASNet",
            "U-Net",
        ],
    ),
    (
        "Recurrent Neural Networks",
        [
            "Recurrent Neural Network",
            "Long Short-Term Memory",
            "Gated Recurrent Unit",
            "Bidirectional RNN",
            "Bidirectional LSTM",
        ],
    ),
    (
        "Transformer Architectures",
        [
            "Transformer",
            "BERT",
            "GPT",
            "T5",
            "Vision Transformer",
            "CLIP",
            "DALL-E",
            "Swin Transformer",
        ],
    ),
    (
        "Attention Mechanisms",
        [
            "Attention Mechanism",
            "Self-Attention Network",
            "Multi-Head Attention",
            "Cross-Attention",
        ],
    ),
    (
        "Graph Neural Networks",
        [
            "Graph Neural Network",
            "Graph Convolutional Network",
            "Graph Attention Network",
            "GraphSAGE",
            "Message Passing Neural Network",
            "Graph Isomorphism Network",
            "Temporal Graph Network",
        ],
    ),
    (
        "Generative Models",
        [
            "Autoencoder",
            "Variational Autoencoder",
            "Generative Adversarial Network",
            "Conditional GAN",
            "Deep Convolutional GAN",
            "StyleGAN",
            "CycleGAN",
            "Diffusion Model",
            "Denoising Diffusion Probabilistic Model",
            "Normalizing Flow",
        ],
    ),
    (
        "Energy-based Models",
        ["Boltzmann Machine", "Restricted Boltzmann Machine", "Hopfield Network"],
    ),
    (
        "Memory Networks",
        [
            "Neural Turing Machine",
            "Memory Network",
            "Differentiable Neural Computer",
        ],
    ),
    (
        "Specialized Architectures",
        [
            "Capsule Network",
            "Siamese Network",
            "Triplet Network",
            "Attention Network",
            "Pointer Network",
            "WaveNet",
            "Seq2Seq",
            "Encoder-Decoder",
        ],
    ),
    (
        "Object Detection Models",
        [
            "YOLO",
            "R-CNN",
            "Fast R-CNN",
            "Faster R-CNN",
            "Mask R-CNN",
            "FPN",
            "RetinaNet",
        ],
    ),
    (
        "Time Series Models",
        [
            "ARIMA Model",
            "SARIMA Model",
            "State Space Model",
            "Temporal Convolutional Network",
            "Prophet",
        ],
    ),
    ("Point Cloud Models", ["PointNet", "PointNet++"]),
    (
        "Matrix Factorization",
        [
            "Matrix Factorization",
            "Non-negative Matrix Factorization",
            "Singular Value Decomposition",
        ],
    ),
]

# Learning Methods with categories
learning_methods_data = [
    (
        "Basic Learning Paradigms",
        [
            "Supervised Learning",
            "Unsupervised Learning",
            "Semi-Supervised Learning",
            "Self-Supervised Learning",
            "Reinforcement Learning",
        ],
    ),
    (
        "Advanced Learning Paradigms",
        [
            "Transfer Learning",
            "Multi-Task Learning",
            "Meta-Learning",
            "Few-Shot Learning",
            "Zero-Shot Learning",
            "One-Shot Learning",
            "Active Learning",
            "Online Learning",
            "Incremental Learning",
            "Continual Learning",
            "Lifelong Learning",
            "Curriculum Learning",
        ],
    ),
    (
        "Training Strategies",
        [
            "Batch Learning",
            "Mini-Batch Learning",
            "Stochastic Learning",
            "End-to-End Learning",
            "Adversarial Training",
            "Contrastive Learning",
            "Knowledge Distillation",
            "Fine-Tuning",
            "Pre-training",
            "Prompt Learning",
            "In-Context Learning",
        ],
    ),
    (
        "Optimization Methods",
        [
            "Gradient Descent",
            "Stochastic Gradient Descent",
            "Backpropagation",
            "Maximum Likelihood Estimation",
            "Maximum A Posteriori",
            "Expectation-Maximization",
            "Variational Inference",
            "Evolutionary Learning",
        ],
    ),
    (
        "Reinforcement Learning Methods",
        [
            "Q-Learning",
            "Policy Gradient",
            "Value Iteration",
            "Policy Iteration",
            "Temporal Difference Learning",
            "Monte Carlo Learning",
            "Actor-Critic",
            "Model-Free Learning",
            "Model-Based Learning",
            "Inverse Reinforcement Learning",
            "Imitation Learning",
            "Multi-Agent Learning",
        ],
    ),
    (
        "Special Learning Settings",
        [
            "Weakly Supervised Learning",
            "Noisy Label Learning",
            "Positive-Unlabeled Learning",
            "Cost-Sensitive Learning",
            "Imbalanced Learning",
            "Multi-Instance Learning",
            "Multi-View Learning",
            "Co-Training",
            "Self-Training",
            "Pseudo-Labeling",
        ],
    ),
    (
        "Domain and Distribution",
        [
            "Domain Adaptation",
            "Domain Generalization",
            "Covariate Shift Adaptation",
            "Out-of-Distribution Learning",
        ],
    ),
    (
        "Collaborative Learning",
        [
            "Federated Learning",
            "Distributed Learning",
            "Collaborative Learning",
            "Privacy-Preserving Learning",
        ],
    ),
    (
        "Ensemble Methods",
        ["Ensemble Learning", "Bagging", "Boosting", "Stacking", "Blending"],
    ),
    (
        "Representation Learning",
        [
            "Representation Learning",
            "Feature Learning",
            "Metric Learning",
            "Distance Learning",
            "Embedding Learning",
            "Dictionary Learning",
            "Manifold Learning",
        ],
    ),
    (
        "Learning Modes",
        [
            "Generative Learning",
            "Discriminative Learning",
            "Transductive Learning",
            "Inductive Learning",
        ],
    ),
]

# Tasks with categories
tasks_data = [
    (
        "Prediction Tasks",
        [
            "Regression",
            "Classification",
            "Binary Classification",
            "Multi-class Classification",
            "Multi-label Classification",
            "Ordinal Regression",
            "Time Series Forecasting",
            "Survival Analysis",
        ],
    ),
    (
        "Ranking and Retrieval",
        [
            "Ranking",
            "Information Retrieval",
            "Recommendation",
            "Collaborative Filtering",
            "Content-Based Filtering",
        ],
    ),
    ("Clustering and Grouping", ["Clustering", "Community Detection", "Grouping"]),
    (
        "Dimensionality Reduction",
        ["Dimensionality Reduction", "Feature Selection", "Feature Extraction"],
    ),
    (
        "Anomaly and Outlier",
        [
            "Anomaly Detection",
            "Outlier Detection",
            "Novelty Detection",
            "Fraud Detection",
        ],
    ),
    ("Density and Distribution", ["Density Estimation", "Distribution Estimation"]),
    (
        "Structured Prediction",
        [
            "Structured Prediction",
            "Sequence Labeling",
            "Named Entity Recognition",
            "Part-of-Speech Tagging",
            "Sequence-to-Sequence",
        ],
    ),
    (
        "Computer Vision Tasks",
        [
            "Image Classification",
            "Object Detection",
            "Object Localization",
            "Semantic Segmentation",
            "Instance Segmentation",
            "Panoptic Segmentation",
            "Pose Estimation",
            "Action Recognition",
            "Video Classification",
            "Optical Flow Estimation",
            "Depth Estimation",
            "Image Super-Resolution",
            "Image Denoising",
            "Image Inpainting",
            "Style Transfer",
            "Image-to-Image Translation",
            "Image Generation",
            "Video Generation",
        ],
    ),
    (
        "Natural Language Processing Tasks",
        [
            "Language Modeling",
            "Text Classification",
            "Sentiment Analysis",
            "Machine Translation",
            "Text Summarization",
            "Question Answering",
            "Reading Comprehension",
            "Dialog Generation",
            "Text Generation",
            "Paraphrase Generation",
            "Text-to-Speech",
            "Speech Recognition",
            "Speech Synthesis",
        ],
    ),
    (
        "Graph Tasks",
        [
            "Node Classification",
            "Link Prediction",
            "Graph Classification",
            "Graph Generation",
            "Graph Matching",
            "Influence Maximization",
        ],
    ),
    (
        "Decision Making",
        [
            "Decision Making",
            "Policy Learning",
            "Control",
            "Planning",
            "Optimization",
            "Resource Allocation",
        ],
    ),
    (
        "Design Tasks",
        [
            "Experimental Design",
            "Hyperparameter Optimization",
            "Architecture Search",
            "AutoML",
            "Neural Architecture Search",
        ],
    ),
    (
        "Association and Pattern",
        ["Association Rule Mining", "Pattern Recognition", "Motif Discovery"],
    ),
    (
        "Matching and Alignment",
        ["Entity Matching", "Entity Alignment", "Record Linkage", "Image Matching"],
    ),
    (
        "Generative Tasks",
        ["Data Generation", "Data Augmentation", "Synthetic Data Generation"],
    ),
    (
        "Causal Tasks",
        [
            "Causal Inference",
            "Treatment Effect Estimation",
            "Counterfactual Reasoning",
        ],
    ),
]


prompt_template_clean_draft = """You are an expert at extracting structured information from scientific papers in the AI for Science domain. Your task is to analyze the provided paper text and extract ALL AI-related information in JSON format, including models, learning methods, tasks, datasets, and their performances.

## Available Options

### Models (Architectures/Structures):
{models}

### Learning Methods (How to Learn):
{learning_methods}

### Tasks (What Problems to Solve):
{tasks}

## Task: Extract AI Components from AI4Science Paper

Analyze the paper and extract ALL instances of models, learning methods, tasks, datasets, and performances. For each component:
1. Identify which specific items from the provided lists are mentioned
2. Extract the exact context where they appear
3. Link models, learning methods, and tasks together when they are used in combination
4. Record any performance metrics reported

Output your response as a valid JSON object with this exact structure:

{{

    "datasets": [
        {{
            "name": "the specific datasets used from references or curated by this paper",
            "domain": "scientific domain",
            "size": "dataset size/scale/number of samples",
            "source": "dataset source or reference",
            "used_for": "which task(s) this dataset is used for"
        }}
    ],
    
    "tasks_addressed": [
        {{
            "name": "exact task name from the list above",
            "context": "relevant sentence or phrase from paper",
            "description": "specific problem formulation in this paper"
        }}
    ],
    
    "models_used": [
        {{
            "name": "exact model name from the list above",
            "context": "relevant sentence or phrase from paper",
            "variants": "any specific variants or modifications mentioned"
        }}
    ],
    
    "learning_methods_used": [
        {{
            "name": "exact learning method name from the list above",
            "context": "relevant sentence or phrase from paper",
            "details": "any specific details about how it's applied"
        }}
    ],
    

    "model_task_learning_combinations": [
        {{
            "model": "model name",
            "learning_method": "learning method name",
            "task": "task name",
            "performance": {{
                "metrics": {{"metric_name": "value"}},
                "context": "relevant sentence with results"
            }},
            "description": "brief description of this combination"
        }}
    ],
    
    "application_domains": [
        "list of application domains of this paper"
    ]
    
    "key_contributions": [
        "list the main AI/ML contributions of this paper"
    ]
}}

## Important Guidelines:
1. Only use model names, learning methods, and tasks from the provided lists above
2. If a concept is mentioned but not in the lists, note it in "key_contributions" instead
3. Extract ALL instances - a paper may use multiple models, methods, and address multiple tasks
4. Be precise with terminology - use exact names from the lists
5. When models are combined or modified, note this in "variants" or "details"
6. Link performance metrics to the specific model-method-task combination
7. If unsure whether something matches a list item, include your best match and explain in "description"

## Example Scenarios:
- If paper uses "ResNet with supervised learning for image classification", extract:
  - model: "ResNet"
  - learning_method: "Supervised Learning"
  - task: "Image Classification"
  
- If paper uses "Transformer with transfer learning and fine-tuning for text classification", extract:
  - model: "Transformer"
  - learning_methods: ["Transfer Learning", "Fine-Tuning"]
  - task: "Text Classification"

- If paper uses "GNN with semi-supervised learning for node classification", extract:
  - model: "Graph Neural Network"
  - learning_method: "Semi-Supervised Learning"
  - task: "Node Classification"

Now analyze the following paper text:

---
[PAPER TEXT]
---

Return only the JSON object, no additional text.
"""


all_models = []
for cate_models in models_data:
    all_models += cate_models[1]
all_learning_methods = []
for cate_learning_methods in learning_methods_data:
    all_learning_methods += cate_learning_methods[1]

all_tasks = []
for cate_tasks in tasks_data:
    all_tasks += cate_tasks[1]


# Format the template with the lists
prompt_template_clean = prompt_template_clean_draft.format(
    models="\n".join([f"- {m}" for m in all_models]),
    learning_methods="\n".join([f"- {lm}" for lm in all_learning_methods]),
    tasks="\n".join([f"- {t}" for t in all_tasks]),
)

prompt_template = """You are an expert at extracting structured information from scientific papers in the AI for Science domain. Your task is to analyze the provided paper text and extract ALL AI-related information in JSON format, including multiple models, datasets, and their performances.

## Task: Extract AI Components from AI4Science Paper

Analyze the paper and extract ALL instances of models, datasets, and performances. Output your response as a valid JSON object with this exact structure:

{
  "datasets": [
    {
      "name": "dataset name",
      "domain": "scientific domain",
      "size": "dataset size/scale/number of samples",
      "source": "dataset source or reference",
      "preprocessing": "preprocessing methods",
      "split": "train/val/test split if mentioned",
      "description": "brief description of dataset content"
    }
  ],
  "models": [
    {
      "model_name": "specific model name",
      "model_type": "type (e.g., CNN, Transformer, GNN, VAE, Diffusion)",
      "architecture_details": "detailed architecture description",
      "backbone": "backbone network if applicable",
      "parameters": "number of parameters if mentioned",
      "pretrained": "yes/no and pretrained source",
      "novel_contribution": "what's new about this model",
      "purpose": "baseline/proposed/comparison model"
    }
  ],
  "algorithms_and_methods": {
    "training_algorithms": ["list all training approaches used"],
    "optimization_methods": ["list all optimizers"],
    "loss_functions": [
      {
        "name": "loss function name",
        "description": "brief description or formula"
      }
    ],
    "special_techniques": ["all techniques like augmentation, regularization, etc."],
    "evaluation_metrics": ["all metrics used"]
  },
  "experiments": [
    {
      "experiment_name": "name or description of experiment",
      "dataset_used": "which dataset(s)",
      "models_compared": ["list of models in this experiment"],
      "task": "specific task (e.g., classification, regression, generation)",
      "results": [
        {
          "model_name": "model name",
          "dataset": "dataset name",
          "metrics": {
            "metric_name_1": "value",
            "metric_name_2": "value"
          },
          "additional_notes": "any important notes about results"
        }
      ]
    }
  ],
  "performance_summary": [
    {
      "model": "model name",
      "dataset": "dataset name",
      "task": "task description",
      "best_metric": "name of primary metric",
      "best_score": "score value",
      "comparison": "compared to baseline/sota",
      "improvement": "improvement percentage if mentioned"
    }
  ],
  "computational_details": {
    "hardware": "all hardware mentioned",
    "training_time": "training duration per model if specified",
    "inference_time": "inference speed if mentioned",
    "hyperparameters": [
      {
        "model": "model name",
        "learning_rate": "value",
        "batch_size": "value",
        "epochs": "value",
        "other_params": "other hyperparameters"
      }
    ]
  },
  "scientific_application": {
    "domain": "scientific field",
    "problem": "specific problem addressed",
    "ai_application": "how AI is applied to solve it",
    "key_findings": ["list of main findings"],
    "limitations": "limitations mentioned if any",
    "future_work": "future directions if mentioned"
  },
  "code_and_resources": {
    "code_available": "yes/no/url",
    "pretrained_models": "availability and links",
    "supplementary": "supplementary materials mentioned"
  }
}

## Instructions:
1. Extract ALL models mentioned in the paper (proposed, baselines, comparisons)
2. Extract ALL datasets used or mentioned
3. Extract ALL performance results, experiments, and comparisons
4. Capture performance metrics for each model-dataset combination
5. Use "not_specified" for unavailable information
6. Use empty array [] for list fields with no information
7. Ensure valid JSON format
8. Be comprehensive - don't miss any model or dataset mentioned
9. Include ablation studies and variant models if present

---
[PAPER TEXT]
---

Output only the JSON object, no additional text.
"""


# print(prompt_template_clean)
