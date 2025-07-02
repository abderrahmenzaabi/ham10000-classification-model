# 🩺 Skin Cancer Classification with Transfer Learning (HAM10000)

This notebook trains a deep learning model to classify skin lesions using transfer learning (DenseNet121) on the HAM10000 dataset.

## Main steps

-Data preparation:
   -used kagglehub to get ham10000 dataset and added image path to each sample in the csv file using find_image_path.
   -created an integer labeling for each class
   -split the data into train and test ds and used stratify to balance class distribution and used prefetch and autotune to speed up the data processing 
   -since the dataset ham10000 is highly imbalenced i used class weights to handle this imbalence
  
Building the model:
   -used DenseNet121 which works well for medical imaging because it preserves details
   -Transfer learning with frozen base with early stopping , checkpoint and tensorboard. ( i got the best accuracy here)
   -fine tuned the top 5 layers for few epochs which made it worse so i loaded back the weights of the model( before fine tuning it)

##  Results
-Validation accuracy: ~68% 
-Confusion matrix & classification metrics included

## Notes
- The dataset is highly imbalanced, so further improvements could include more data augmentation or testing other architectures.
- This is an educational project to demonstrate transfer learning on a real medical dataset.
- Logs and plots generated with TensorBoard.