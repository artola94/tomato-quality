"""Classify one tomato image with solanum_grader.
"""
import argparse
import sys

CLASS_NAMES = ("Damaged", "Old", "Ripe", "Unripe")
IMG_SIZE = (256, 256)

RELEASE_MSG = (
    "Pass the weights with --model. "
    "They are published as the GitHub Release asset solanum_grader.keras.\n"
    "  python src/infer.py --model solanum_grader.keras --image path/to/image.jpg"
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Classify one tomato image with solanum_grader. "
            "The classes, in order, are Damaged, Old, Ripe, and Unripe."
        )
    )
    parser.add_argument(
        "--model",
        help="Path to solanum_grader.keras from the GitHub Release.",
    )
    parser.add_argument("--image", help="Path to an RGB tomato image.")
    return parser.parse_args(argv)


def load_image(tf, path):
    # Same preparation as training: RGB, 256×256, pixels divided by 255.
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return tf.expand_dims(img, 0)


def main(argv=None):
    args = parse_args(argv)
    if not args.model:
        print(RELEASE_MSG, file=sys.stderr)
        return 1
    if not args.image:
        print("Pass the photo with --image.", file=sys.stderr)
        return 1

    # TensorFlow is imported here, once a weight file has been given.
    import tensorflow as tf

    model = tf.keras.models.load_model(args.model)
    row = model(load_image(tf, args.image), training=False)[0]
    probs = [float(v) for v in row]
    if len(probs) != len(CLASS_NAMES):
        print(
            f"The model returned {len(probs)} probabilities; "
            f"this network has {len(CLASS_NAMES)} classes.",
            file=sys.stderr,
        )
        return 1

    pred = max(range(len(probs)), key=probs.__getitem__)
    print(CLASS_NAMES[pred])
    for name, prob in zip(CLASS_NAMES, probs):
        print(f"{name} {prob:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
