##################################################
TopBrain 2025 MICCAI Challenge Data Release

Website: https://topbrain2025.grand-challenge.org
  _____           ____            _
 |_   _|__  _ __ | __ ) _ __ __ _(_)_ __
   | |/ _ \| '_ \|  _ \| '__/ _` | | '_ \
   | | (_) | |_) | |_) | | | (_| | | | | |
   |_|\___/| .__/|____/|_|  \__,_|_|_| |_|
           |_|
##################################################


1. Overview

This dataset is released for the "TopBrain Segmentation Challenge for Whole Brain Vessel Anatomy" to be held with the Medical Image Computing and Computer Assisted Intervention (MICCAI) conference in 2025.
The TopBrain challenge releases the first dataset with voxel annotations on over 40 landmark brain vessel anatomies covering both arteries and veins for both the magnetic resonance angiography (MRA) and computed tomography angiography (CTA) modalities.
The annotation protocol proposed by TopBrain aims to unify and standardize brain vessel anatomy segmentation that can handle real-world anatomical variations.
The segmentation model developed will have important research and clinical applications for all major vessels of the brain.


2. Data Usage License

Following the definitions of "https://opendata.swiss/en/terms-of-use" on open data use,
the following license is chosen:
    "Terms of data use:

    Open use. Must provide the source. Use for commercial purposes requires permission of the data owner.
      * You may use this dataset for non-commercial purposes.
      * You may use this dataset for commercial purposes, but you must seek prior permission from the data owner.
      * You must provide the source (author, title and link to the dataset)."

By downloading the data, you agree with the license terms.

For more information, please refer to the "License.txt" file in the same release folder.


3. Contents of Data

The training data consist of
- 50 (=25 pairs) CTA and MRA from the TopCoW image data as the batch-1 and batch-2 release (more data will be released in October 2025)
- voxel annotations for over 40 classes of landmark brain vessel anatomies, including the Circle of Willis (CoW) labels that are from the TopCoW data labels

The training (Tr) data folder has the following sub-folders:

* `imagesTr_topbrain_{ct|mr}`: Angiographic scans in nifti format, 16-bit signed. LPS+ orientation
  * The nifti filenames are saved with schema: `topcow_{modality}_{pat_id}_0000.nii.gz`
    * `modality`: `mr` for MRA, `ct` for CTA
    * `pat_id`: patient ID, `001`, `002`, ...
* `labelsTr_topbrain_{ct|mr}`: Multiclass segmentation mask nifti with TopBrain vessel anatomy labels
  * Labels 1-12 and 15 are retained from TopCoW annotations
    * TopCoW labels are a subset of TopBrain
  * CTA has 40 labels
  * MRA has 42 labels
    * CTA and MRA share 34 overlapping labels
  * Voxel values for different brain vessel segments are documented on the TopBrain challenge website's "Data" page;
    * and in the provided ITK-Snap labelmap files (see below)
* `itksnap_labelmap_txt`: Label-map files with values for each anatomy class that can be loaded to ITK-Snap to visualize the anatomy labels.


4. Citation

If you use the TopBrain challenge data in your work, please cite the TopBrain challenge website (for now) and the TopCoW challenge pre-print:

* TopBrain website: https://topbrain2025.grand-challenge.org
* TopCoW challenge pre-print: https://arxiv.org/abs/2312.17670


5. Contact

Kaiyuan Yang (kaiyuan.yang@uzh.ch) from University of Zurich
Pengcheng Shi (shipc1220@gmail.com) from Medical Image Insights

---
Updated August-14-2025
