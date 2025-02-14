import cv2
from kani_algorytm import KaniAlgorythm, ImageShowKaniAlgorythmEnum

kaniAlgo = KaniAlgorythm(
    image_size=(500, 500),
    image_show_list=[
        ImageShowKaniAlgorythmEnum.FILTERED,
    ],
    kernel_size=5,
    deviation=1.5,
    threshold_dividers=(15, 8)
)

threshold_images = [
    (10, 10), (12, 8), (20, 5),
    (40, 4), (40, 30), (3, 1), (15, 10), (60, 1)
]

for item in threshold_images:
    kaniAlgo = KaniAlgorythm(
        image_size=(500, 500),
        image_show_list=[
        ],
        kernel_size=5,
        deviation=0.0001,
        threshold_dividers=item,
    )

    image, bounds = kaniAlgo.process_image_with_return('files/1.jpg')

    cv2.imshow(f"{round(bounds[0], 2)}-{round(bounds[1], 2)}", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
