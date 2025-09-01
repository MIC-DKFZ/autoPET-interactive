# [autoPET Interactive] MICCAI 2025 Challenge: Team LesionLocator

🚀 This repository contains the code for our approach to the [AutoPET IV Challenge](https://autopet-iv.grand-challenge.org/) (*Team LesionLocator*), submitted to Task 1 (interactive lesion segmentation in PET/CT).  

Our method extends our [winning solution from AutoPET III](https://github.com/MIC-DKFZ/autopet-3-submission) with **promptable user interaction support**, enabling segmentation refinement from foreground/background clicks. The final submission is an **ensemble of two models** trained with distance-transform–based prompt encoding, simulated user clicks and additional PET/CT data.

📄 Please cite our paper when using this repository:  

**Towards Interactive Lesion Segmentation in PET/CT with Promptable Models**  

&nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2508.21680-b31b1b.svg)](http://arxiv.org/abs/2508.21680)


*Authors:* Maximilian Rokuss, Yannick Kirchhoff, Fabian Isensee, and Klaus H. Maier-Hein  

---

# Overview  

Our approach builds on [nnUNet ResEncL](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md) and the [winning solution from AutoPET III](https://github.com/MIC-DKFZ/autopet-3-submission). Key additions for AutoPET IV include:  

- **Interactive click prompts**: user clicks are encoded as additional input channels.  
- **Prompt encoding**: Gaussian kernels were tested, but Euclidean Distance Transform (EDT)–based encoding consistently outperformed them.  
- **Custom point simulation**: we extend the official challenge code to simulate more diverse prompts (used in 20% of training cases).  
- **Training data extension**: additional PET/CT datasets were used for robustness.

---

# Getting started  

### Installation  

```bash
git clone https://github.com/MIC-DKFZ/autoPET-interactive.git
cd autoPET-interactive
pip install -e .
```

### Preprocessing

1. Download the [autoPET/CT IV dataset](https://autopet-iv.grand-challenge.org/dataset/) (or any additional datasets you wish to include).
2. Perform organ extraction by following the instructions in the [autoPET III submission repository](https://github.com/MIC-DKFZ/autopet-3-submission) (using TotalSegmentator).
3. Run the standard **nnU-Net v2 preprocessing pipeline**, but specify a custom preprocessor (`autoPetPreprocessor`) that handles both lesion segmentation masks and organ segmentation masks.

   * Organ masks are expected to be stored in the folder `labelsTr_organs`.
   * You can adjust the number of processes (`-np` and `-npfp`) for faster execution.
   * The dataset identifier can be chosen freely and is referred to as `DATASET_ID`.

```bash
nnUNetv2_plan_and_preprocess \
    -d DATASET_ID \
    -preprocessor_name autoPetPreprocessor \
    -c 3d_fullres \
    -np 20 \
    -npfp 20
```


### Training

Training the model can then be simply achieved by [downloading the pretrained (not the final) checkpoint](https://zenodo.org/records/13753413) (Dataset619_nativemultistem) and running:

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2 -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights /path/to/pretrained/weights/fold_all/checkpoint_final.pth
```

We train a five fold cross-validation for our final submission. Note that we performed MultiTalent pretraining also on more data than in the checkpoint.


### Inference

After training your own model or [downloading our final checkpoint here](https://drive.google.com/file/d/1vn7yaD12CVlnNQMwNn6SG_ZoORLvs4rf/view?usp=sharing) you can use an adapted version of the nnUNet inference, for more information see [here](nnunetv2/inference/autopet_predictor.py) and [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md). A script to perform python based inference and docker building for Grand Challenge is provided in the [inference.py](inference.py). RESOURCE_PATH refers to the folder containing all 5 folds.

Happy coding! 🚀

# Citation


```
@article{rokuss2025interactivelesionsegmentationwholebody,
      title={Towards Interactive Lesion Segmentation in Whole-Body PET/CT with Promptable Models}, 
      author={Maximilian Rokuss and Yannick Kirchhoff and Fabian Isensee and Klaus H. Maier-Hein},
      year={2025},
      eprint={2508.21680},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21680}, 
}

@article{rokuss2024fdgpsmahitchhikersguide,
      title={From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging}, 
      author={Maximilian Rokuss and Balint Kovacs and Yannick Kirchhoff and Shuhan Xiao and Constantin Ulrich and Klaus H. Maier-Hein and Fabian Isensee},
      journal={ArXiv},
      year={2024},
      publisher={arXiv},
      eprint={2409.09478},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.09478}, 
}
```