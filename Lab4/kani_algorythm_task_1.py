from kani_algorytm import KaniAlgorythm, ImageShowKaniAlgorythmEnum

kaniAlgo = KaniAlgorythm(
    image_size=(500,500),
    image_show_list=[
        ImageShowKaniAlgorythmEnum.GRAYSCALE,
        ImageShowKaniAlgorythmEnum.GAUSSIAN,
    ],
    kernel_size=5,
    deviation=1.5
)

kaniAlgo.process_image('Lab4/files/1.jpg')
