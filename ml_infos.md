---
layout: default
title: ML Infos
permalink: /ml_infos/
---
            
<div align="center">
    <h1>Machine Learning Infos in AI4(M)S Papers</h1> 
    <h3>Update Time: 2025-10-05 02:18:58</h3>
    </div>

---

## [395. Activation entropy of dislocation glide in body-centered cubic metals from atomistic simulations]((https://doi.org/10.1038/s41467-025-62390-w)), Nature Communications *(September 24, 2025)*

| Category | Items |
|----------|-------|
| **Models** | Machine-Learning Interatomic Potentials (MLIP),<br>Embedded Atom Method (EAM) potentials |
| **Datasets** | Fe and W MLIP training datasets (extended from refs. 23, 26),<br>PAFI sampling datasets (finite-temperature sampled configurations along reaction coordinates),<br>Empirical potential (EAM) reference calculations,<br>Experimental yield stress datasets (from literature) |
| **Tasks** | Regression,<br>Data Generation,<br>Image Classification |
| **Learning Methods** | Supervised Learning,<br>Transfer Learning |
| **Performance Highlights** | activation_entropy_harmonic_regime_Fe: ΔS(z2) = 6.3 kB,<br>activation_entropy_difference_above_T0_Fe: ΔS(z2)-ΔS(z1) = 1.6 kB,<br>activation_entropy_harmonic_regime_W: approx. 8 kB,<br>MD_velocity_prefactor_fit_HTST: ν = 3.8×10^9 Hz (HTST fit),<br>MD_velocity_prefactor_fit_VHTST: ν = 9.2×10^10 Hz (variational HTST fit),<br>simulation_cell_size: 96,000 atoms (per atomistic simulation cell),<br>PAFI_computational_cost_per_condition: 5×10^4 to 2.5×10^5 CPU hours (for anharmonic Gibbs energy calculations),<br>Hessian_diagonalization_cost: ≈5×10^4 CPU-hours per atomic system using MLIP,<br>effective_entropy_variation_range_Fe_EAM: ΔSeff varies by ~10 kB between 0 and 700 MPa (Fe, EAM),<br>departure_from_harmonicity_temperature: marked departure from harmonic prediction above ~20 K (Fe, EAM),<br>inverse_Meyer_Neldel_TMNs: Fe: TMN = -406 K (effective fit), W: TMN = -1078 K (effective fit) |
| **Application Domains** | materials science,<br>computational materials / atomistic simulation,<br>solid mechanics / metallurgy (dislocation glide & yield stress in BCC metals),<br>physics of defects (dislocations, kink-pair nucleation) |

---


## [391. Active Learning for Machine Learning Driven Molecular Dynamics]((https://doi.org/10.48550/arXiv.2509.17208)), Preprint *(September 21, 2025)*

| Category | Items |
|----------|-------|
| **Models** | Graph Neural Network |
| **Datasets** | Chignolin protein (in-house benchmark suite) |
| **Tasks** | Regression,<br>Data Generation,<br>Dimensionality Reduction,<br>Distribution Estimation |
| **Learning Methods** | Active Learning,<br>Supervised Learning |
| **Performance Highlights** | TICA_W1_before: 1.15023,<br>TICA_W1_after: 0.77003,<br>TICA_W1_percent_change: -33.05%,<br>Bond_length_W1_before: 0.00043,<br>Bond_length_W1_after: 0.00022,<br>Bond_length_W1_percent_change: -48.84%,<br>Bond_angle_W1_before: 0.11036,<br>Bond_angle_W1_after: 0.10148,<br>Bond_angle_W1_percent_change: -8.05%,<br>Dihedral_W1_before: 0.25472,<br>Dihedral_W1_after: 0.36378,<br>Reaction_coordinate_W1_before: 0.15141,<br>Reaction_coordinate_W1_after: 0.38302,<br>loss_function: mean-squared error (MSE) between predicted CG forces and projected AA forces (force matching),<br>W1_TICA_after_active_learning: 0.77003 |
| **Application Domains** | Molecular Dynamics,<br>Protein conformational modeling,<br>Coarse-grained simulations for biomolecules,<br>ML-driven drug discovery / computational biophysics |

---


## [372. Guided multi-agent AI invents highly accurate, uncertainty-aware transcriptomic aging clocks]((https://doi.org/10.1101/2025.09.08.674588)), Preprint *(September 12, 2025)*

| Category | Items |
|----------|-------|
| **Models** | XGBoost,<br>LightGBM,<br>Support Vector Machine,<br>Linear Model,<br>Transformer |
| **Datasets** | ARCHS4,<br>ARCHS4 — blood subset,<br>ARCHS4 — colon subset,<br>ARCHS4 — lung subset,<br>ARCHS4 — ileum subset,<br>ARCHS4 — heart subset,<br>ARCHS4 — adipose subset,<br>ARCHS4 — retina subset |
| **Tasks** | Regression,<br>Feature Selection,<br>Feature Extraction,<br>Clustering |
| **Learning Methods** | Supervised Learning,<br>Ensemble Learning,<br>Imbalanced Learning |
| **Performance Highlights** | R2: 0.619,<br>R2: 0.604,<br>R2: 0.574,<br>R2_Ridge: 0.539,<br>R2_ElasticNet: 0.310,<br>R2: 0.957,<br>MAE_years: 3.7,<br>R2_all: 0.726,<br>MAE_all_years: 6.17,<br>R2_confidence_weighted: 0.854,<br>MAE_confidence_weighted_years: 4.26,<br>mean_calibration_error: 0.7%,<br>R2_per_window_range: ≈0.68–0.74,<br>lung_R2: 0.969,<br>blood_R2: 0.958,<br>ileum_R2: 0.958,<br>heart_R2: 0.910,<br>adipose_R2: 0.887,<br>retina_R2: 0.594 |
| **Application Domains** | aging biology / geroscience,<br>transcriptomics,<br>biomarker discovery,<br>computational biology / bioinformatics,<br>clinical biomarker development (biological age clocks),<br>AI-assisted scientific discovery (multi-agent workflows) |

---


## [326. Probing the limitations of multimodal language models for chemistry and materials research]((https://doi.org/10.1038/s43588-025-00836-3)), Nature Computational Science *(August 11, 2025)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Vision Transformer,<br>GPT,<br>BERT |
| **Datasets** | MaCBench (v1.0.0) |
| **Tasks** | Classification,<br>Multi-class Classification,<br>Regression,<br>Image Classification,<br>Feature Extraction,<br>Sequence-to-Sequence,<br>Text Generation,<br>Binary Classification |
| **Learning Methods** | Prompt Learning,<br>Fine-Tuning,<br>In-Context Learning,<br>Pre-training |
| **Performance Highlights** | equipment_identification_accuracy: 0.77,<br>table_composition_extraction_accuracy: 0.53,<br>hand_drawn_to_SMILES_accuracy: 0.8,<br>isomer_relationship_naming_accuracy: 0.24,<br>stereochemistry_assignment_accuracy: 0.24,<br>baseline_accuracy: 0.22,<br>crystal_system_assignment_accuracy: 0.55,<br>space_group_assignment_accuracy: 0.45,<br>atomic_species_counting_accuracy: 0.85,<br>capacity_values_interpretation_accuracy: 0.59,<br>Henry_constants_comparison_accuracy: 0.83,<br>XRD_amorphous_vs_crystalline_accuracy: 0.69,<br>AFM_interpretation_accuracy: 0.24,<br>MS_NMR_interpretation_accuracy: 0.35,<br>XRD_highest_peak_identification_accuracy: 0.74,<br>XRD_relative_intensity_ranking_accuracy: 0.28,<br>performance_dependency_on_internet_presence: positive_correlation (visualized in Fig. 5) |
| **Application Domains** | chemistry (organic chemistry, spectroscopy, NMR, mass spectrometry),<br>materials science (crystallography, MOF isotherms, electronic structure, AFM),<br>laboratory experiment understanding and safety assessment,<br>in silico experiments and materials characterization,<br>scientific literature information extraction and data curation |

---


## [300. AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model]((https://doi.org/10.1101/2025.06.25.661532)), Preprint *(July 11, 2025)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>U-Net,<br>Convolutional Neural Network,<br>Multi-Layer Perceptron,<br>Multi-Head Attention,<br>Self-Attention Network |
| **Datasets** | ENCODE,<br>GTEx (via RECOUNT3),<br>FANTOM5 (CAGE),<br>4D Nucleome (contact maps / Hi-C / Micro-C),<br>PolyA_DB / Polyadenylation annotations,<br>ClinVar,<br>MFASS (Multiplexed Functional Assay of Splicing using Sort-seq),<br>CAGI5 MPRA saturation mutagenesis challenge,<br>Open Targets (GWAS credible sets),<br>eQTL Catalog / SuSiE fine-mapped eQTLs,<br>ChromBPNet benchmarks (caQTL/dsQTL/bQTL),<br>ENCODE-rE2G (CRISPRi enhancer-gene validation),<br>gnomAD common variants (chr22 subset) |
| **Tasks** | Regression,<br>Binary Classification,<br>Sequence Labeling,<br>Structured Prediction,<br>Link Prediction,<br>Ranking,<br>Feature Extraction / Representation Learning |
| **Learning Methods** | Supervised Learning,<br>Pre-training,<br>Knowledge Distillation,<br>Ensemble Learning,<br>Multi-Task Learning,<br>Fine-Tuning,<br>Representation Learning,<br>Batch Learning,<br>Gradient Descent |
| **Performance Highlights** | genome_track_evaluations_outperform_count: AlphaGenome outperformed external models on 22 out of 24 genome track evaluations,<br>variant_effect_evaluations_outperform_count: AlphaGenome matched or outperformed external models on 24 out of 26 variant effect prediction evaluations,<br>gene_expression_LFC_rel_improvement_vs_Borzoi: +17.4% (relative improvement in cell type-specific gene-level expression LFC Pearson r vs Borzoi),<br>contact_maps_rel_improvement_vs_Orca_Pearson_r: +6.3% (Pearson r), +42.3% (cell type-specific differences),<br>ProCapNet_rel_improvement_total_counts_Pearson_r: +15% (vs ProCapNet),<br>ChromBPNet_rel_improvement_accessibility_Pearson_r: +8% ATAC, +19% DNase (total counts Pearson r),<br>splice_benchmarks_SOTA: AlphaGenome achieves SOTA on 6 out of 7 splicing VEP benchmarks,<br>ClinVar_deep_intronic_auPRC: 0.66 (AlphaGenome composite) vs 0.64 (Pangolin),<br>ClinVar_splice_region_auPRC: 0.57 (AlphaGenome) vs 0.55 (Pangolin),<br>ClinVar_missense_auPRC: 0.18 (AlphaGenome) vs 0.16 (DeltaSplice/Pangolin/DeltaSplice),<br>MFASS_auPRC: 0.54 (AlphaGenome) vs 0.51 (Pangolin); SpliceAI/DeltaSplice = 0.49,<br>Junctions_prediction_Pearson_r_examples: High correlations reported for junction counts across tissues (e.g., Pearson r ~0.75-0.76 in examples),<br>contact_map_Pearson_r_vs_Orca: +6.3% Pearson r improvement; cell type differential prediction improvement +42.3% (compared to Orca),<br>contact_map_examples_Pearson_r_values: Example intervals: AlphaGenome Pearson r ~0.79-0.86 vs ground truth maps (figure examples),<br>zero_shot_causality_auROC_comparable_to_Borzoi: AlphaGenome zero-shot causality comparable to Borzoi (mean auROC ~0.68),<br>supervised_RF_auROC: Random Forest using AlphaGenome multimodal features improved mean auROC from 0.68 (zero-shot) to 0.75, surpassing Borzoi supervised performance (mean auROC 0.71),<br>zero_shot_cell_type_matched_DNase_Pearson_r: 0.57 (AlphaGenome cell type-matched DNase predictions; comparable to ChromBPNet and Borzoi Ensemble),<br>LASSO_multi-celltype_DNase_Pearson_r: 0.63 (AlphaGenome with LASSO aggregation over all cell types),<br>LASSO_multimodal_Pearson_r: 0.65 (AlphaGenome integrating multiple modalities across cell types; SOTA on CAGI5 reported),<br>ENCODE-rE2G_zero_shot_auPRC: AlphaGenome outperformed Borzoi in identifying validated enhancer-gene links, particularly beyond 10 kb distance; zero-shot within 1% auPRC of ENCODE-rE2G-extended trained model,<br>supervised_integration_auPRC_improvement: Including AlphaGenome features into ENCODE-rE2G-extended model increased auPRC to new SOTA across distance bins (Fig.4j),<br>APA_Spearman_r: 0.894 (AlphaGenome) vs 0.790 (Borzoi) for APA prediction; reported as SOTA,<br>paQTL_auPRC_within_10kb: 0.629 (AlphaGenome) vs 0.621 (Borzoi),<br>paQTL_auPRC_proximal_50bp: 0.762 (AlphaGenome) vs 0.727 (Borzoi),<br>caQTL_African_coefficient_Pearson_r: 0.74 (AlphaGenome predicted vs observed effect sizes for causal caQTLs; DNase GM12878 track example),<br>SPI1_bQTL_coefficient_Pearson_r: 0.55 (AlphaGenome predicted vs observed SPI1 bQTLs),<br>caQTL_causality_AP_mean: AlphaGenome achieved higher Average Precision vs Borzoi and ChromBPNet across multiple ancestries and datasets (specific AP values shown in Supplementary/Extended Data; e.g., AP = 0.50-0.63 depending on dataset),<br>inference_speed: <1 second per variant on NVIDIA H100 (single student model), enabling fast large-scale scoring,<br>overall_variant_benchmarks_outperform_count: Matched or outperformed external SOTA on 24/26 variant effect prediction benchmarks (Fig.1e) |
| **Application Domains** | Regulatory genomics,<br>Variant effect prediction / clinical variant interpretation,<br>Splicing biology and splicing variant interpretation,<br>Gene expression regulation and eQTL interpretation,<br>Alternative polyadenylation (APA) and paQTLs,<br>Chromatin accessibility and TF binding QTL analysis,<br>3D genome architecture (contact map prediction),<br>Enhancer–gene linking and functional genomics perturbation interpretation,<br>Massively parallel reporter assay (MPRA) analysis,<br>GWAS interpretation and prioritization |

---


## [264. End-to-end data-driven weather prediction]((https://doi.org/10.1038/s41586-025-08897-0)), Nature *(May 2025)*

| Category | Items |
|----------|-------|
| **Models** | Vision Transformer,<br>U-Net,<br>Multi-Layer Perceptron,<br>Convolutional Neural Network,<br>Encoder-Decoder,<br>Multi-Head Attention,<br>Self-Attention Network |
| **Datasets** | ERA5 reanalysis,<br>HadISD (Hadley Centre integrated surface dataset),<br>ICOADS (International Comprehensive Ocean-Atmosphere Data Set),<br>IGRA (Integrated Global Radiosonde Archive),<br>ASCAT (Metop Advanced Scatterometer) Level 1B,<br>AMSU-A / AMSU-B / Microwave Humidity Sounder / HIRS,<br>IASI (Infrared Atmospheric Sounding Interferometer),<br>GridSat (Gridded Geostationary Brightness Temperature Data),<br>HRES (ECMWF Integrated Forecasting System high-resolution) forecasts,<br>GFS (NCEP Global Forecast System) forecasts,<br>NDFD (National Digital Forecast Database) |
| **Tasks** | Time Series Forecasting,<br>Regression,<br>Image-to-Image Translation,<br>Feature Extraction |
| **Learning Methods** | Supervised Learning,<br>Pre-training,<br>Fine-Tuning,<br>End-to-End Learning,<br>Transfer Learning,<br>Stochastic Gradient Descent,<br>Representation Learning |
| **Performance Highlights** | LW-RMSE: Aardvark achieved lower latitude-weighted RMSE than GFS across most lead times for many variables; approached HRES performance for most variables and lead times (held-out test year 2018, ERA5 ground truth),<br>LW-RMSE at t=0: initial-state estimation error reported and compared to HRES analysis; Aardvark has non-zero error at t=0 against ERA5 whereas HRES also non-zero,<br>MAE: Aardvark produced skilful station forecasts up to 10 days lead time; competitive with station-corrected HRES and matched NDFD over CONUS for 2-m temperature; for 10-m wind, mixed results (worse than station-corrected HRES over CONUS but outperformed NDFD).,<br>Fine-tuning improvement (MAE %): 2-m temperature: −6% MAE (Europe, West Africa, Pacific, Global), −3% MAE (CONUS). 10-m wind speed: 1–2% MAE improvements for most regions (except Pacific).,<br>Inference speed: Full forecast generation ~1 second on four NVIDIA A100 GPUs,<br>Computational cost comparison: HRES data assimilation and forecasting ~1,000 node hours (operational NWP) |
| **Application Domains** | Numerical weather forecasting / atmospheric sciences,<br>Local weather forecasting (station-level forecasts),<br>Transportation (weather impacts),<br>Agriculture (heatwaves, cold waves forecasting),<br>Energy and renewable energy (wind forecasts),<br>Public safety and emergency services (extreme weather warnings, tropical cyclones),<br>Marine forecasting (ocean/ship observations),<br>Insurance and finance (weather risk modelling),<br>Environmental monitoring (potential extension to atmospheric chemistry and air quality),<br>Operational meteorology (replacement/augmentation of NWP pipelines) |

---


## [205. Crystal structure generation with autoregressive large language modeling]((https://doi.org/10.1038/s41467-024-54639-7)), Nature Communications *(December 06, 2024)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Graph Neural Network,<br>Variational Autoencoder,<br>Denoising Diffusion Probabilistic Model,<br>U-Net,<br>Transformer |
| **Datasets** | CrystaLLM training set (2.3M unique cell composition-space group pairs, 2,047,889 training CIF files),<br>Challenge set (70 structures: 58 from recent literature unseen in training, 12 from training),<br>Held-out test set (subset of the curated dataset),<br>Perov-5 benchmark,<br>Carbon-24 benchmark,<br>MP-20 benchmark,<br>MPTS-52 benchmark |
| **Tasks** | Synthetic Data Generation,<br>Data Generation,<br>Regression |
| **Learning Methods** | Self-Supervised Learning,<br>Pre-training,<br>Supervised Learning,<br>Fine-Tuning,<br>Reinforcement Learning |
| **Performance Highlights** | held-out_test_validity_no_space_group_%: 93.8,<br>held-out_test_validity_with_space_group_%: 94.0,<br>space_group_consistent_no_space_group_%: 98.8,<br>space_group_consistent_with_space_group_%: 99.1,<br>atom_site_multiplicity_consistent_%: 99.4,<br>bond_length_reasonableness_score_mean: 0.988,<br>bond_lengths_reasonable_%: 94.6,<br>average_valid_generated_length_tokens_no_sg: 331.9 ± 42.6,<br>average_valid_generated_length_tokens_with_sg: 339.0 ± 41.4,<br>match_with_test_structure_with_space_group_within_3_attempts_%: 88.1,<br>Perov-5_match_rate_n=20_%_CrystaLLM_a: 98.26,<br>Perov-5_RMSE_CrystaLLM_a_n=20: 0.0236,<br>Carbon-24_match_rate_n=20_%_CrystaLLM_a: 83.60,<br>Carbon-24_RMSE_CrystaLLM_a_n=20: 0.1523,<br>MP-20_match_rate_n=20_%_CrystaLLM_a: 75.14,<br>MP-20_RMSE_CrystaLLM_a_n=20: 0.0395,<br>MPTS-52_match_rate_n=20_%_CrystaLLM_a: 32.98,<br>MPTS-52_RMSE_CrystaLLM_a_n=20: 0.1197,<br>Perov-5_match_rate_n=20_%_CrystaLLM_b: 97.60,<br>Perov-5_RMSE_CrystaLLM_b_n=20: 0.0249,<br>Carbon-24_match_rate_n=20_%_CrystaLLM_b: 85.17,<br>Carbon-24_RMSE_CrystaLLM_b_n=20: 0.1514,<br>MP-20_match_rate_n=20_%_CrystaLLM_b: 73.97,<br>MP-20_RMSE_CrystaLLM_b_n=20: 0.0349,<br>MPTS-52_match_rate_n=20_%_CrystaLLM_b: 33.75,<br>MPTS-52_RMSE_CrystaLLM_b_n=20: 0.1059,<br>MPTS-52_match_rate_n=1_CrystaLLM_c: 28.30,<br>MPTS-52_RMSE_n=1_CrystaLLM_c: 0.0850,<br>MPTS-52_match_rate_n=20_CrystaLLM_c: 47.45,<br>MPTS-52_RMSE_n=20_CrystaLLM_c: 0.0780,<br>unconditional_generation_attempts: 1000,<br>valid_generated_CIFs: 900,<br>unique_structures: 891,<br>novel_structures_vs_training_set: 102,<br>mean_Ehull_of_102_novel_structures_eV_per_atom: 0.40,<br>novel_structures_with_Ehull_<=_0.1_eV_per_atom: 20,<br>novel_structures_with_Ehull_exact_0.00_eV_per_atom: 3,<br>ALIGNN_used_as_predictor: formation energy per atom (used as reward in MCTS),<br>average_ALIGNN_energy_change_after_MCTS_meV_per_atom: -153 ± 15 (prediction decrease across 102 unconditional-generated compositions),<br>MCTS_validity_rate_improvement_no_space_group_%: 95.0,<br>MCTS_validity_rate_improvement_with_space_group_%: 60.0,<br>MCTS_minimum_Ef_improvement_no_space_group_%: 85.0,<br>MCTS_minimum_Ef_improvement_with_space_group_%: 65.0,<br>MCTS_mean_Ef_improvement_no_space_group_%: 70.0,<br>MCTS_mean_Ef_improvement_with_space_group_%: 65.0,<br>mean_Ehull_change_after_ALIGNN-guided_MCTS_meV_per_atom: -56 ± 15 (mean Ehull improved to 0.34 eV/atom across 102 structures); 22 structures within 0.1 eV/atom of hull,<br>successful_generation_rate_small_model_no_sg_%: 85.7,<br>successful_generation_rate_small_model_with_sg_%: 88.6,<br>successful_generation_rate_large_model_no_sg_%: 87.1,<br>successful_generation_rate_large_model_with_sg_%: 91.4,<br>match_rate_seen_small_model_%: 50.0,<br>match_rate_seen_large_model_%: 83.3,<br>match_rate_unseen_small_model_no_sg_%: 25.9,<br>match_rate_unseen_small_model_with_sg_%: 34.5,<br>match_rate_unseen_large_model_no_sg_%: 37.9,<br>match_rate_unseen_large_model_with_sg_%: 41.4,<br>pyrochlore_cell_parameter_R2: 0.62,<br>pyrochlore_cell_parameter_MAE_A: 0.08 Å |
| **Application Domains** | Materials science,<br>Computational materials discovery,<br>Inorganic crystal structure prediction,<br>Materials informatics,<br>Computational chemistry / solid-state physics,<br>High-throughput screening and DFT-accelerated materials design |

---


## [189. Transforming science labs into automated factories of discovery]((https://doi.org/10.1126/scirobotics.adm6991)), Science Robotics *(October 23, 2024)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Multi-Layer Perceptron |
| **Datasets** | historical and online data,<br>massive quantities of experimental data (generated by automated labs),<br>experimental runs / datasets produced by autonomous systems cited (e.g., AlphaFlow, mobile robotic chemist) |
| **Tasks** | Experimental Design,<br>Optimization,<br>Decision Making,<br>Policy Learning,<br>Control,<br>Planning,<br>Regression,<br>Experimental Design |
| **Learning Methods** | Reinforcement Learning,<br>Supervised Learning,<br>Prompt Learning,<br>Representation Learning |
| **Performance Highlights** | _None_ |
| **Application Domains** | chemistry,<br>biochemistry,<br>materials science,<br>energy,<br>catalysis,<br>biotechnology,<br>sustainability,<br>electronics,<br>drug design,<br>semiconductor materials,<br>batteries,<br>photocatalysis,<br>organic light-emitting devices (OLEDs) |

---


## [188. Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models]((https://doi.org/10.48550/arXiv.2410.12771)), Preprint *(October 16, 2024)*

| Category | Items |
|----------|-------|
| **Models** | Graph Neural Network,<br>Transformer |
| **Datasets** | OMat24 (Open Materials 2024),<br>MPtrj (Materials Project trajectories),<br>Alexandria,<br>sAlexandria (subset of Alexandria),<br>OC20 / OC22 (referenced),<br>WBM dataset (used by Matbench-Discovery),<br>Matbench-Discovery (benchmark) |
| **Tasks** | Regression,<br>Binary Classification,<br>Classification |
| **Learning Methods** | Pre-training,<br>Fine-Tuning,<br>Transfer Learning,<br>Supervised Learning,<br>Self-Supervised Learning |
| **Performance Highlights** | F1: 0.916,<br>MAE_energy: 0.020 eV/atom (20 meV/atom),<br>RMSE: 0.072 eV/atom (72 meV/atom),<br>Accuracy: 0.974,<br>Precision: 0.923,<br>Recall: 0.91,<br>R2: 0.848,<br>F1: 0.823,<br>MAE_energy: 0.035 eV/atom (35 meV/atom),<br>RMSE: 0.082 eV/atom (82 meV/atom),<br>Accuracy: 0.944,<br>Precision: 0.792,<br>Recall: 0.856,<br>R2: 0.802,<br>Energy_MAE_validation: 9.6 meV/atom,<br>Forces_MAE_validation: 43.1 meV/Å,<br>Stress_MAE_validation: 2.3 (units consistent with meV/Å^3),<br>Test_splits_energy_MAE_range: ≈9.7 - 14.6 meV/atom depending on test split (ID/OOD/WBM),<br>F1: 0.86,<br>MAE_energy: 0.029 eV/atom (29 meV/atom),<br>RMSE: 0.078 eV/atom (78 meV/atom),<br>Accuracy: 0.957,<br>Precision: 0.862,<br>Recall: 0.858,<br>R2: 0.823,<br>Validation_energy_MAE_on_MPtrj: 10.58 - 12.4 meV/atom depending on model variant; (Table 9: eqV2-L-DeNS energy 10.58 meV/atom; eqV2-S 12.4 meV/atom),<br>Validation_forces_MAE_on_MPtrj: ≈30 - 32 meV/Å |
| **Application Domains** | Materials discovery,<br>Inorganic materials / solid-state materials,<br>DFT surrogate modeling (predicting energies, forces, stress),<br>Computational screening for stable materials (thermodynamic stability / formation energy prediction),<br>Catalyst discovery and related atomistic simulations,<br>Molecular dynamics / non-equilibrium structure modeling (potential downstream application) |

---


## [178. Closed-loop transfer enables artificial intelligence to yield chemical knowledge]((https://doi.org/10.1038/s41586-024-07892-1)), Nature *(September 2024)*

| Category | Items |
|----------|-------|
| **Models** | Support Vector Machine,<br>Linear Model |
| **Datasets** | 2,200-molecule design space (donor–bridge–acceptor combinatorial space),<br>BO-synthesized experimental rounds (Phase I): 30 molecules,<br>Full experimental photostability dataset (CLT campaign),<br>Predicted photostabilities across 2,200 molecules (DFT+RDKit featurizations) |
| **Tasks** | Regression,<br>Optimization,<br>Feature Selection,<br>Feature Extraction,<br>Dimensionality Reduction |
| **Learning Methods** | Supervised Learning |
| **Performance Highlights** | LOOV_R2: 0.86,<br>Spearman_R2_on_validation_batches: 0.54,<br>Mann-Whitney_p_value: 0.026,<br>Top7_avg_photostability: 165,<br>Bottom7_avg_photostability: 97,<br>Most_predictive_models_R2_threshold: >0.70,<br>Top5_avg_photostability_improvement: >500%,<br>sampling_fraction: <1.5% of 2,200 space |
| **Application Domains** | molecular photostability / photodegradation for light-harvesting small molecules,<br>organic electronics (organic photovoltaics, organic light-emitting diodes),<br>dyed polymers and photo-active coatings,<br>solar fuels and photosynthetic system analogues,<br>organic laser emitters (mentioned as further application),<br>stereoselective aluminium complexes for ring-opening polymerization (mentioned as further application) |

---


## [172. The power and pitfalls of AlphaFold2 for structure prediction beyond rigid globular proteins]((https://doi.org/10.1038/s41589-024-01638-w)), Nature Chemical Biology *(August 2024)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Attention Mechanism |
| **Datasets** | AlphaFold Protein Structure Database,<br>Protein Data Bank (PDB),<br>Human proteome models (AF2 coverage),<br>McDonald et al. peptide benchmark (588 peptides),<br>Yin et al. heterodimer benchmark (152 heterodimeric complexes),<br>Bryant et al. heterodimer benchmark (dataset used in their study),<br>Terwilliger et al. molecular-replacement benchmark (215 structures),<br>Membrane protein benchmarks (various sets),<br>NMR ensemble datasets (general),<br>SAXS / SANS datasets and Small-Angle Scattering Biological Data Bank derived datasets |
| **Tasks** | Structured Prediction,<br>Sequence-to-Sequence,<br>Clustering |
| **Learning Methods** | Pre-training,<br>Fine-Tuning,<br>Ensemble Learning,<br>Representation Learning,<br>Transfer Learning,<br>Stochastic Learning |
| **Performance Highlights** | human_proteome_coverage: 98.5% modeled,<br>high_confidence_residue_fraction: ~50% of residues across all proteins predicted with high confidence (cited average),<br>pLDDT_thresholds: pLDDT > 70 interpreted as higher confidence; pLDDT > 90 as very high,<br>peptide_benchmark_size: 588 peptides (McDonald et al.),<br>peptide_prediction_note: AF2 predicts many α-helical and β-hairpin peptide structures with surprising accuracy (no single numeric accuracy given in paper excerpt),<br>heterodimer_success_Yin: 51% success rate (AF2 and AlphaFold2-Multimer on 152 heterodimeric complexes),<br>heterodimer_success_Bryant: 63% success rate (Bryant et al. study),<br>molecular_replacement_success: 187 of 215 structures solved using AlphaFold-guided molecular replacement (Terwilliger et al.),<br>alternative_conformation_sampling_note: modifications (reduced recycles, shallow MSAs, MSA clustering, enabling dropout) allow sampling of alternative conformations (no single numeric accuracy provided),<br>AlphaMissense_note: AlphaMissense provides probability of missense variant pathogenicity; AF2 itself 'has not been trained or validated for predicting the effect of mutations' (authors' caution) |
| **Application Domains** | Structural biology (protein 3D structure prediction and validation),<br>Proteomics (proteome-scale modeling; human proteome coverage),<br>Integrative structural methods (integration with SAXS, NMR, cryo-EM, X-ray diffraction),<br>Drug discovery / therapeutics (identifying therapeutic candidates, ligand/cofactor modeling),<br>Membrane protein biology (transmembrane protein modeling),<br>Intrinsically disordered proteins (IDPs/IDRs) and conformational ensembles,<br>Peptide biology and peptide–protein interactions,<br>De novo protein design |

---


## [171. OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization]((https://doi.org/10.1038/s41592-024-02272-z)), Nature Methods *(August 2024)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Attention Mechanism,<br>Self-Attention Network,<br>Multi-Head Attention |
| **Datasets** | OpenProteinSet (replication of AlphaFold2 training set),<br>Protein Data Bank (PDB),<br>Uniclust MSAs,<br>CAMEO validation set,<br>CASP15 domains,<br>CATH-derived domain splits (topologies/architectures/classes),<br>Subsampled training sets (ablation experiments),<br>Rosetta decoy ranking dataset (subset) |
| **Tasks** | Regression,<br>Sequence Labeling,<br>Binary Classification,<br>Ranking |
| **Learning Methods** | Supervised Learning,<br>Knowledge Distillation,<br>Fine-Tuning,<br>Pre-training,<br>Self-Supervised Learning,<br>Distributed Learning,<br>Ensemble Learning |
| **Performance Highlights** | OpenFold (mean lDDT-Cα on CAMEO, main comparison): 0.911,<br>AlphaFold2 (mean lDDT-Cα on CAMEO, main comparison): 0.913,<br>OpenFold final replication (after clamping change): 0.902 lDDT-Cα (on CAMEO validation set),<br>Full data model peak (early reported value): 0.83 lDDT-Cα (after 20,000 steps),<br>10,000-sample subsample after 7,000 steps: exceeded 0.81 lDDT-Cα,<br>1,000-chain ablation (short run, ~7,000 steps): 0.64 lDDT-Cα,<br>Inference speedup (overall OpenFold vs AlphaFold2): up to 3-4x faster (single A100 GPU),<br>FlashAttention effect on short sequences (<1000 residues): up to 15% additional speedup in OpenFold when applicable,<br>Sequence length robustness: OpenFold runs successfully on sequences and complexes exceeding 4,000 residues; AlphaFold2 crashes beyond ~2,500 residues on single GPU,<br>Secondary structure learning order (qualitative): α-helices learned first, then β-sheets, then less common SSEs (measured by F1 over DSSP categories); final high F1 scores for SSEs,<br>Contact F1 (for fragments / SSEs): improves earlier for shorter helices and narrower sheets; specific numbers are plotted in Fig. 5b and Extended Data Fig. 8 (no single consolidated numeric value in text),<br>Number of models in final ensemble: 10 distinct models (seven snapshots from main run + additional models from branch),<br>Effect: Ensemble used at prediction time to generate alternate structural hypotheses; explicit ensemble metric improvement not numerically summarized in a single value in main text |
| **Application Domains** | Protein structural biology,<br>Biomolecular modeling (protein complexes, peptide–protein interactions),<br>Evolutionary sequence analysis / MSA-based modeling,<br>RNA structure prediction (discussed as potential application),<br>Spatial reasoning over polymers and arbitrary molecules (structure module / invariant point attention) |

---


## [157. Closed-Loop Multi-Objective Optimization for Cu–Sb–S Photo-Electrocatalytic Materials’ Discovery]((https://doi.org/10.1002/adma.202304269)), Advanced Materials *(June 04, 2024)*

| Category | Items |
|----------|-------|
| **Models** | Gaussian Process |
| **Datasets** | Cu–Sb–S HTE experimental dataset (this work),<br>Initial sampling prior (Latin hypercube sampling runs),<br>Materials Project phase diagram (Cu–Sb–S subspaces) |
| **Tasks** | Regression,<br>Optimization,<br>Feature Selection,<br>Multi-objective Classification |
| **Learning Methods** | Active Learning,<br>Batch Learning |
| **Performance Highlights** | RMSE_Y: 0.05,<br>RMSE_bandgap: 0.19,<br>RMSE_Cu1+/Cu_ratio: 0.17,<br>GPRU_RMSE: high (not numerical; model uncertainty remained high for uniformity),<br>R2_improvements: R2 scores improved over iterations (not all numerical values specified; improvements noted from iterations 2–4 and 6–8),<br>photocurrent_optimum: -186 μA cm^-2 at 0 V vs RHE,<br>photocurrent_baseline: -86 μA cm^-2 (batch 1),<br>relative_improvement: 2.3x |
| **Application Domains** | photo-electrochemical water splitting (photo-electrocatalysis / photocathode discovery),<br>materials discovery,<br>high-throughput experimentation (automated synthesis and characterization),<br>closed-loop autonomous experimentation |

---


## [127. A dynamic knowledge graph approach to distributed self-driving laboratories]((https://doi.org/10.1038/s41467-023-44599-9)), Nature Communications *(January 23, 2024)*

| Category | Items |
|----------|-------|
| **Models** | Gaussian Process |
| **Datasets** | Aldol condensation closed-loop optimisation dataset (Cambridge & Singapore),<br>Knowledge graph provenance triples (experiment provenance) |
| **Tasks** | Optimization,<br>Experimental Design,<br>Data Generation |
| **Learning Methods** | Evolutionary Learning,<br>Online Learning |
| **Performance Highlights** | highest_yield_%: 93,<br>number_of_runs: 65,<br>best_environment_factor: 26.17,<br>best_space-time_yield_g_per_L_per_h: 258.175 |
| **Application Domains** | chemical reaction optimisation,<br>flow chemistry,<br>laboratory automation / self-driving laboratories (SDLs),<br>digital twin / knowledge graph representations for scientific labs,<br>experiment provenance and FAIR data in chemical sciences |

---


## [85. The rise of self-driving labs in chemical and materials sciences]((https://doi.org/10.1038/s44160-022-00231-0)), Nature Synthesis *(June 2023)*

| Category | Items |
|----------|-------|
| **Models** | Multi-Layer Perceptron,<br>Feedforward Neural Network,<br>Graph Neural Network |
| **Datasets** | Open Reaction Database (ORD),<br>Chiral metal halide perovskite nanoparticle experiments,<br>Photocatalyst formulation campaign (hydrogen evolution),<br>Quantum dot / semiconductor nanoparticle synthesis datasets,<br>3D-printed geometry experiments for mechanical optimization,<br>General datasets generated by self-driving labs (SDLs) |
| **Tasks** | Optimization,<br>Experimental Design,<br>Regression,<br>Image Classification,<br>Clustering,<br>Hyperparameter Optimization |
| **Learning Methods** | Active Learning,<br>Evolutionary Learning,<br>Supervised Learning,<br>Online Learning,<br>Transfer Learning,<br>Representation Learning |
| **Performance Highlights** | discovery_speedup: >1,000× faster (referenced for autonomous synthesis–property mapping and on-demand synthesis of semiconductor and metal nanoparticles),<br>notes: specific performance numbers vary by study,<br>photocatalyst_activity: 6× more active than prior art,<br>experiments: 688 experiments in 8-day continuous unattended operation,<br>experiment_count_reduction: 60× fewer experiments than conventional grid search (three-dimensional-printed geometry case),<br>general_benefit: reduced total cost of computation and experimentation when leveraging prior data/models (qualitative),<br>example_reference: transfer learning used in designing lattices for impact protection (ref. 82) |
| **Application Domains** | Chemical synthesis (organic synthesis, retrosynthesis),<br>Materials science (nanomaterials, thin films, perovskites),<br>Clean energy technologies (photocatalysts, solar materials),<br>Pharmaceuticals / active pharmaceutical ingredients (APIs),<br>Additive manufacturing / mechanical design (3D-printed geometries),<br>Catalysis,<br>Device manufacturing and co-design (materials + device integration) |

---


## [78. Generative Models as an Emerging Paradigm in the Chemical Sciences]((https://doi.org/10.1021/jacs.2c13467)), Journal of the American Chemical Society *(April 26, 2023)*

| Category | Items |
|----------|-------|
| **Models** | Variational Autoencoder,<br>Generative Adversarial Network,<br>Normalizing Flow,<br>Diffusion Model,<br>Graph Neural Network,<br>Recurrent Neural Network,<br>Gaussian Process |
| **Datasets** | GuacaMol,<br>MOSES (Molecular Sets),<br>Polymer Genome,<br>ANI-2x (ANI2x) |
| **Tasks** | Data Generation,<br>Graph Generation,<br>Sequence-to-Sequence,<br>Optimization,<br>Regression,<br>Language Modeling |
| **Learning Methods** | Reinforcement Learning,<br>Policy Gradient,<br>Actor-Critic,<br>Deterministic Policy Gradient,<br>Temporal Difference Learning,<br>Adversarial Training,<br>Representation Learning,<br>Active Learning,<br>Supervised Learning |
| **Performance Highlights** | penalized_logP_benchmark: GraphAF outperformed other common generative models at the time in its ability to generate high penalized logP values (no numeric value provided in text),<br>benchmarking_tasks: MolGAN enabled better predictions on a number of benchmarking tasks (no numeric values provided in text) |
| **Application Domains** | Chemical sciences,<br>Molecular discovery / drug discovery,<br>Materials science (including organic crystals and functional materials),<br>Polymeric/macromolecular design,<br>Automated/self-driving laboratories / autonomous experimentation,<br>Computational chemistry and molecular simulation (integration with ML interatomic potentials),<br>Synthetic chemistry / retrosynthetic planning |

---


## [52. Distributed representations of atoms and materials for machine learning]((https://doi.org/10.1038/s41524-022-00729-3)), npj Computational Materials *(March 18, 2022)*

| Category | Items |
|----------|-------|
| **Models** | Feedforward Neural Network,<br>Graph Neural Network,<br>Multi-Layer Perceptron |
| **Datasets** | Materials Project structures (used to train SkipAtom),<br>Elpasolite formation energy dataset,<br>OQMD (Open Quantum Materials Database) Formation Energy dataset,<br>Matbench test-suite datasets (benchmark tasks),<br>Mat2Vec corpus (materials science literature),<br>Atom2Vec embeddings / dataset (co-occurrence matrix),<br>Processed data & scripts for this study (repository) |
| **Tasks** | Regression,<br>Binary Classification,<br>Feature Extraction |
| **Learning Methods** | Unsupervised Learning,<br>Supervised Learning,<br>Maximum Likelihood Estimation,<br>Stochastic Gradient Descent,<br>Mini-Batch Learning,<br>Representation Learning,<br>Pre-training |
| **Performance Highlights** | MAE (eV/atom) - Elpasolite (SkipAtom 30 dim): 0.1183 ± 0.0050,<br>MAE (eV/atom) - Elpasolite (SkipAtom 86 dim): 0.1126 ± 0.0078,<br>MAE (eV/atom) - Elpasolite (SkipAtom 200 dim): 0.1089 ± 0.0061,<br>MAE (eV/atom) - OQMD Formation Energy (Bag-of-Atoms one-hot, sum pooled, 86 dim): 0.0388 ± 0.0002,<br>MAE (eV/atom) - OQMD Formation Energy (Atom2Vec 86, sum): 0.0396 ± 0.0004,<br>MAE (eV/atom) - OQMD Formation Energy (SkipAtom 86, sum): 0.0420 ± 0.0005,<br>MAE (eV/atom) - OQMD Formation Energy (Mat2Vec 200, sum): 0.0401 ± 0.0004,<br>Benchmark summary (qualitative): Pooled Mat2Vec achieved best results in 4 of 8 benchmark tasks; pooled SkipAtom best in 2 of 8. 200-dim representations generally outperform 86-dim. Sum- and mean-pooling outperform max-pooling.,<br>Qualitative improvement over existing benchmarks: Authors report outperforming existing benchmarks on tasks where only composition is available (Experimental Band Gap, Bulk Metallic Glass Formation, Experimental Metallicity) — see Fig. 5 for comparisons. |
| **Application Domains** | Materials science / materials informatics,<br>Computational materials / inorganic crystals,<br>DFT property prediction (formation energy, band gap),<br>High-throughput materials screening,<br>Chemical composition-based property prediction |

---


## [41. Accurate prediction of protein structures and interactions using a three-track neural network]((https://doi.org/10.1126/science.abj8754)), Science *(August 20, 2021)*

| Category | Items |
|----------|-------|
| **Models** | Transformer,<br>Attention Mechanism,<br>Self-Attention Network,<br>Cross-Attention,<br>Multi-Head Attention,<br>Ensemble Learning |
| **Datasets** | Protein Data Bank (PDB),<br>CASP14 targets,<br>CAMEO medium and hard targets,<br>Curated set of 693 human protein domains,<br>GPCR benchmark (human GPCRs of currently unknown structure and GPCR sequences with determined structures),<br>Escherichia coli protein complexes (known structures),<br>Cryo-EM map EMD-21645 (IL-12R–IL-12 complex) |
| **Tasks** | Sequence-to-Sequence,<br>Regression,<br>Sequence-to-Sequence |
| **Learning Methods** | End-to-End Learning,<br>Backpropagation,<br>Supervised Learning,<br>Cross-Attention,<br>Stochastic Learning,<br>Ensemble Learning |
| **Performance Highlights** | qualitative: structure predictions with accuracies approaching those of DeepMind (AlphaFold2) on CASP14 targets,<br>runtime_end_to_end: ~10 min on an RTX2080 GPU for proteins with fewer than 400 residues (after sequence and template search),<br>lDDT_fraction: >33% of 693 modeled human domains have predicted lDDT > 0.8,<br>lDDT_to_Ca-RMSD: predicted lDDT > 0.8 corresponded to an average Cα-RMSD of 2.6 Å on CASP14 targets,<br>TM-score_complexes: many cases with TM-score > 0.8 for two- and three-chain complexes,<br>Cα-RMSD_examples: p101 GBD predicted vs final refined structure: Cα-RMSD = 3.0 Å over the beta-sheets,<br>improved_accuracy: Ensembles and using multiple discontinuous crops generated higher-accuracy models (qualitative improvement reported),<br>CAMEO_benchmark: RoseTTAFold outperformed all other servers on 69 CAMEO medium and hard targets (TM-score values used for ranking),<br>molecular_replacement_success: RoseTTAFold models enabled successful molecular replacement for four challenging crystallographic datasets that had previously eluded MR with PDB models,<br>example_Ca-RMSD_SLP: 95 Cα atoms superimposed within 3 Å yielding a Cα-RMSD of 0.98 Å for SLP C-terminal domain |
| **Application Domains** | Structural biology,<br>Protein structure prediction,<br>X-ray crystallography (molecular replacement),<br>Cryo-electron microscopy model building/fitting,<br>Protein-protein complex modeling,<br>Functional annotation of proteins / interpretation of disease mutations,<br>Protein design and small-molecule / binder design (computational discovery) |

---


## [26. Coevolutionary search for optimal materials in the space of all possible compounds]((https://doi.org/10.1038/s41524-020-0322-9)), npj Computational Materials *(May 14, 2020)*

| Category | Items |
|----------|-------|
| **Models** | None of the standard ML architectures from the provided list (search uses custom evolutionary / coevolutionary algorithms and physics-based models) |
| **Datasets** | Chemical space of unary and binary compounds constructed from 74 elements (all elements excluding noble gases, rare earth elements, and elements heavier than Pu),<br>Sets of candidate crystal structures per composition (structures generated/optimized with USPEX/VASP) |
| **Tasks** | Optimization,<br>Ranking,<br>Clustering |
| **Learning Methods** | Evolutionary Learning,<br>Stochastic Learning |
| **Performance Highlights** | sampled_systems: 600 systems computed in 20 MendS generations (hardness/stability search),<br>search_space_total_binary_systems: 2775 possible binary systems (from 74 elements),<br>best_detected_hardness_diamond_Hv_GPa: 92.7,<br>lonsdaleite_Hv_GPa: 93.6,<br>SiC_Hv_GPa: 33.3,<br>BP_Hv_GPa: 37.2,<br>example_MoB2_Hv_GPa: 28.5,<br>example_MnB4_Hv_GPa (Pnnm ferromagnetic): 40.7,<br>sampled_systems: 450 binary systems over 15 MendS generations (magnetization search),<br>result_top_material: bcc-Fe identified as having the highest zero-temperature magnetization among all possible compounds |
| **Application Domains** | Computational materials discovery,<br>Theoretical crystallography / crystal structure prediction,<br>Materials design for mechanical properties (hardness, fracture toughness),<br>Magnetic materials discovery (magnetization at zero Kelvin),<br>High-throughput ab initio materials screening |

---


## [16. Active learning for accelerated design of layered materials]((https://doi.org/10.1038/s41524-018-0129-0)), npj Computational Materials *(December 10, 2018)*

| Category | Items |
|----------|-------|
| **Models** | Gaussian Process |
| **Datasets** | Three-layer TMDC hetero-structures (H3) — 126 unique structures,<br>Four-layer TMDC hetero-structures (partial set used in BO tests),<br>Adsorption energies dataset (reference dataset used for BO validation) |
| **Tasks** | Regression,<br>Optimization,<br>Feature Extraction,<br>Feature Selection |
| **Learning Methods** | Supervised Learning,<br>Active Learning,<br>Maximum Likelihood Estimation |
| **Performance Highlights** | training_split_threshold: training sets with fewer than 60% of structures did not produce reliable predictions; >60% showed no additional improvement,<br>evaluation_runs: 100 independent GPR models (randomly selected training sets) used to collect statistics and average out effects from initial training data selection,<br>band_gap_model_training_fraction_used_for_figure: 60% of structures randomly selected for training in shown example,<br>predicted_vs_ground_truth: Figures demonstrate predicted vs ground truth band gap, dispersion curves, and EFF curves with 95% confidence intervals (no single numeric MSE in main text),<br>BO_runs: 500 independent BO runs (different random initial training seeds),<br>max_band_gap_success_rate: 79% of BO runs correctly found the structure with the maximum band gap (1.7 eV); 15% found second-best (1.5 eV); 5% found third-best (1.3 eV),<br>desired_band_gap_1.1eV_success_rate: For searching band gap closest to 1.1 eV, MoSe2-WSe2-WSe2 (band gap 1.05 eV) was returned in 91% of 500 runs,<br>EFF_top_found_rate: In band gap (EFF) optimization, one of the top four (five) optimal structures is found within 30 BO iterations in over 95% of the 500 runs,<br>adsorption_dataset_result: On the adsorption energies dataset, after evaluating only 20% of the dataset, 82% of 500 independent BO runs successfully identified the pair with minimum adsorption energy |
| **Application Domains** | Materials design and discovery,<br>Two-dimensional materials (transition metal dichalcogenide heterostructures),<br>Optoelectronics (band gap engineering for solar cells; Shockley–Queisser limit relevance),<br>Thermoelectrics (electronic transport component and thermoelectric Electronic Fitness Function),<br>Catalysis / surface science (validation on adsorption energy dataset) |

---

