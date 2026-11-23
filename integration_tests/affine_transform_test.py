"""Tests for `affine_transform` input validation."""

import numpy as np
from absl.testing import parameterized

from keras import ops
from keras.src import testing


class AffineTransformTest(testing.TestCase):
    # The other backends disagree on these shapes, so keras does not test them.
    @parameterized.named_parameters(
        ("one_image", (1, 7, 9, 2), (3, 8)),
        ("unbatched_image", (7, 9, 2), (3, 8)),
        ("fewer_transforms", (4, 7, 9, 2), (3, 8)),
    )
    def test_transform_batch_mismatch_raises(
        self, images_shape, transform_shape
    ):
        images = np.ones(images_shape, dtype="float32")
        transform = np.zeros(transform_shape, dtype="float32")
        with self.assertRaisesRegex(
            ValueError, "one transform or one per image"
        ):
            ops.image.affine_transform(images, transform)

    def test_batched_transform_with_unbatched_image(self):
        images = np.ones((7, 9, 2), dtype="float32")
        transform = np.array([[1, 0, 0, 0, 1, 0, 0, 0]], dtype="float32")
        self.assertAllClose(
            ops.image.affine_transform(images, transform), images
        )
