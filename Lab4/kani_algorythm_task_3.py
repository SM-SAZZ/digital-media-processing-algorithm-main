from kani_algorytm import KaniAlgorythm, ImageShowKaniAlgorythmEnum

kaniAlgo = KaniAlgorythm(
    image_size=(500,500),
    image_show_list=[
        ImageShowKaniAlgorythmEnum.GRAYSCALE,
        ImageShowKaniAlgorythmEnum.GAUSSIAN,
        ImageShowKaniAlgorythmEnum.GRAD_LENGTH,
        ImageShowKaniAlgorythmEnum.GRAD_ANGLE,
        ImageShowKaniAlgorythmEnum.SUPPRESSED,
    ],
    kernel_size=5,
    deviation=1.5
)

kaniAlgo.process_image('files/1.jpg')
