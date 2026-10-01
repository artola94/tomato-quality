# Tomato quality

This project trains `solanum_grader`, a small residual network named for *Solanum*, the genus of the tomato. It looks at one fruit and calls it Damaged, Old, Ripe, or Unripe. The notes below are the experiment as it was run: where the images came from, how they were split, what the network looks like, and the test numbers from that finished training.

## What it classifies

A packing line cares most about Damaged fruit, and that class is also the smallest in the set. Ripe, Old, and Unripe are common enough that a test accuracy of 0.9241 reads as a strong score, while Damaged recall is 0.65. Macro F1 treats the four classes equally, and on this test set it is 0.90, so the weak class stays in view.

## Dataset

The photos are the Kaggle set [enalis/tomatoes-dataset](https://www.kaggle.com/datasets/enalis/tomatoes-dataset), folder `content/ieee-mbl-cls`.

The original `train/` folder is divided 85/15 with a stratified split and `SEED=42`. The larger part is what the network learns from. The smaller part is the validation set that EarlyStopping and the learning-rate schedule watch during the 80 epochs. The original `val/` folder is held out and scored once, as the test set, after training has finished.

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
| Damaged | 0.90 | 0.65 | 0.75 | 106 |
| Old | 0.86 | 0.94 | 0.90 | 222 |
| Ripe | 0.97 | 0.98 | 0.98 | 220 |
| Unripe | 0.96 | 0.99 | 0.98 | 177 |

Across the 725 test images, accuracy is 0.9241 and loss is 0.1826. The macro average is 0.92 precision, 0.89 recall, and 0.90 F1. The weighted average is 0.92 for precision, recall, and F1.

Of the 106 Damaged tomatoes, the model misses about 37. Precision on that class is 0.90, so when it does say Damaged it is usually right. The misses are false negatives, and they tend to be called Old.

Adam trained the network at 1e-3 for 80 epochs. EarlyStopping watched validation loss with a patience of 8 and `restore_best_weights`, and the run went through every epoch. ReduceLROnPlateau halved the learning rate after 3 epochs without an improvement in validation loss. Epoch 1 closed at a validation accuracy of 0.3053 and a validation loss of 3.05. The last epoch closed at a training accuracy of 0.9300 and a training loss of 0.1949, with validation accuracy 0.9375, validation loss 0.1567, and a learning rate of 1.95e-6.

## Reproduce

The notebook runs on Kaggle, where TensorFlow and this dataset are already on the machine. With that setup it reads `/kaggle/input/datasets/enalis/tomatoes-dataset/content/ieee-mbl-cls`.

On another computer, set `TOMATO_DATA_ROOT` to your copy of the `ieee-mbl-cls` folder, the directory that contains `train/` and `val/`.

## Inference

The trained weights are the GitHub Release asset `solanum_grader.keras`. The script loads that file, resizes the photo to 256×256, divides the pixels by 255, and calls `model(x, training=False)`. It then prints the predicted class and the four probabilities in the order Damaged, Old, Ripe, Unripe.

```
python src/infer.py --model solanum_grader.keras --image path/to/image.jpg
```

## Limits

This run trains the residual network from scratch and counts every image equally. Early stopping was on, with patience 8, and training still completed all 80 epochs. On a T4, an epoch takes about 364 seconds. The curves in the notebook are the record of that single run.

## Next

A later pass can give Damaged a class weight so it counts for more during training. The same split can also train a MobileNet or an EfficientNet, which would put a transfer-learning baseline next to `solanum_grader`. Saving the Damaged photos the model missed would turn the confusion matrix into an image-by-image review.
