# Tomato quality

This project trains `solanum_grader`, a small residual network named for *Solanum*. It looks at one fruit and calls it Damaged, Old, Ripe, or Unripe. The notes below are the experiment as it was run: where the images came from, how they were split, what the network looks like, and the test numbers from that finished training.

## What it classifies

A packing line cares most about Damaged fruit, and that class is also the smallest in the set. Ripe, Old, and Unripe are common enough that a test accuracy of 0.9379 reads as a strong score, while Damaged recall is 0.71. Macro F1 treats the four classes equally, and on this test set it is 0.92, so the weak class stays in view.

## Dataset

The photos are the Kaggle set [enalis/tomatoes-dataset](https://www.kaggle.com/datasets/enalis/tomatoes-dataset), folder `content/ieee-mbl-cls`.

The original `train/` folder is divided 85/15 with a stratified split and `SEED=42`. The larger part is what the network learns from. The smaller part is the validation set that EarlyStopping and the learning-rate schedule watch. The original `val/` folder is held out and scored once, as the test set, after training has finished.

| Split | Images | Damaged | Old | Ripe | Unripe |
|---|---:|---:|---:|---:|---:|
| Train | 5525 | 807 | 1693 | 1678 | 1347 |
| Val | 976 | 142 | 299 | 297 | 238 |
| Test | 725 | 106 | 222 | 220 | 177 |

Class names follow `sorted()`, which gives Damaged, Old, Ripe, Unripe.

## Model

`solanum_grader` is a residual CNN. A stem of 32 filters leads into three stages, at 64, 128, and 256 channels, with two residual blocks in each stage. Batch normalization follows the convolutions. Spatial dropout follows each pooling stage, then global average pooling and a softmax over the four classes. Every input is a 256×256 RGB image whose pixels have been divided by 255.

Horizontal flip, a rotation of 0.1, and a zoom of 0.1 are Keras layers inside the model. They perturb each training batch. Scoring a photo calls the model with `training=False`, so the resized image goes through the learned filters on their own.

## Test result

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Damaged | 0.95 | 0.71 | 0.81 | 106 |
| Old | 0.88 | 0.96 | 0.92 | 222 |
| Ripe | 0.97 | 0.98 | 0.98 | 220 |
| Unripe | 0.97 | 0.99 | 0.98 | 177 |

Across the 725 test images, accuracy is 0.9379. The macro average is 0.94 precision, 0.91 recall, and 0.92 F1. The weighted average is 0.94 for precision, recall, and F1. The evaluation cell did not print the test loss.

Of the 106 Damaged tomatoes, the model misses 31. Precision on that class is 0.95, so when it does say Damaged it is usually right. The misses are false negatives, and 24 of those 31 are called Old.

Adam started at 1e-3, with room for 80 epochs. EarlyStopping watched validation loss with a patience of 8 and `restore_best_weights`. The run stopped at epoch 74 and kept the weights from epoch 66, where the training accuracy was 0.9415, the training loss was 0.1624, the validation accuracy was 0.9457, and the validation loss was 0.1347, at a learning rate of 3.9063e-06. ReduceLROnPlateau halved the learning rate after 3 epochs without an improvement, and the rate reached its floor of 1e-6. Epoch 1 closed at a validation accuracy of 0.3053 and a validation loss of 3.0263. The test numbers above are from the restored epoch-66 weights.

## Reproduce

The notebook runs on Kaggle, where TensorFlow and this dataset are already on the machine. With that setup it reads `/kaggle/input/datasets/enalis/tomatoes-dataset/content/ieee-mbl-cls`.

On another computer, set `TOMATO_DATA_ROOT` to your copy of the `ieee-mbl-cls` folder, the directory that contains `train/` and `val/`.

## Inference

The trained weights are the GitHub Release asset `solanum_grader.keras`. The script loads that file, resizes the photo to 256×256, divides the pixels by 255, and calls `model(x, training=False)`. It then prints the predicted class and the four probabilities in the order Damaged, Old, Ripe, Unripe.

```
python src/infer.py --model solanum_grader.keras --image path/to/image.jpg
```

## Limits

This run trains the residual network from scratch and counts every image equally. Early stopping was on, with patience 8, and it stopped the run at epoch 74. On a T4, the first epoch took 408 seconds and the later ones about 366 seconds. The epoch log in `reports/metrics.csv` is the record of this run.

## Next

A later pass can give Damaged a class weight so it counts for more during training. The same split can also train a MobileNet or an EfficientNet, which would put a transfer-learning baseline next to `solanum_grader`. Saving the Damaged photos the model missed would turn the confusion matrix into an image-by-image review.
