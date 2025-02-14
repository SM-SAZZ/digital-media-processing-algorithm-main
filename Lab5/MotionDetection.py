import logging

import numpy as np
import cv2


class MotionDetect:
    _threshold: float = None
    _contour_area: float = None

    _kernel_size: int = None
    _deviation: float = None

    def __init__(
            self,
            threshold: float = 15.0,
            contour_area: float = 800.0,
            kernel_size: int = 5,
            deviation: float = 1,
    ):
        self._threshold = threshold
        self._contour_area = contour_area
        self._kernel_size = kernel_size
        self._deviation = deviation

    def __prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Подготовить фрейм к обработке,
        привести к ЧБ,
        применить фильтр Гаусса
        :param frame: Кадр
        :return: Обработанный кадр
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return cv2.GaussianBlur(
            gray_frame,
            (self._kernel_size, self._kernel_size),
            self._deviation
        )




    def process_video(
            self,
            input_path: str,
            output_path: str
    ):
        video_ifstream = cv2.VideoCapture(input_path)
        
        ret, frame = video_ifstream.read()

        if not ret:
            logging.error('Не удалось открыть видеофайл.')
            return

        # Читаем первый кадр в чб, применяем размытие Гаусса
        processed_frame = self.__prepare_frame(frame)

        # Подготовка файла для записи нового видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(video_ifstream.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_ifstream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_ofstream = cv2.VideoWriter(output_path, fourcc, 24, (frame_width, frame_height))

        logging.info(f"Видео будет сохранено по адресу: {output_path}")

        while True:
            previous_frame = processed_frame.copy()

            ret, frame = video_ifstream.read()

            if not ret:
                break

            # Преобразование текущего кадра в оттенки серого и размытие
            processed_frame = self.__prepare_frame(frame)

            # Вычисление разницы между текущим и предыдущим кадром
            frame_difference = cv2.absdiff(previous_frame, processed_frame)

            # Проводим операцию двоичного разделения:
            # проводим бинаризацию изображения по пороговому значению (оставляем либо 255, либо 0)
            _, thresholded_frame = cv2.threshold(frame_difference, self._threshold, 255, cv2.THRESH_BINARY)

            # Поиск контуров объектов
            contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:

                contour_area = cv2.contourArea(contour)

                if contour_area < self._contour_area: # Ищем контур больше заданного значения
                    continue

                # Запись исходного кадра, если найдены значимые изменения
                video_ofstream.write(frame)
                break

            # Прерывание по нажатию клавиши 'Esc'
            if cv2.waitKey(1) & 0xFF == 27:
                break

        video_ifstream.release()
        video_ofstream.release()
        logging.info("Видео успешно записано")
        cv2.destroyAllWindows()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    configs = [
        {
            "threshold": 40,
            "contour_area": 200,
            "deviation": 2,
        },
        {
            "threshold": 100,
            "contour_area": 20,
            "deviation": 2,
        },
        {
            "threshold": 10,
            "contour_area": 500,
            "deviation": 1,
        },
        {
            "threshold": 10,
            "contour_area": 5000,
            "deviation": 1,
        },
        {
            "threshold": 10,
            "contour_area": 3000,
            "deviation": 1,
        },
        {
            "threshold": 10,
            "contour_area": 2000,
            "deviation": 1,
        },
        {
            "threshold": 5,
            "contour_area": 2000,
            "deviation": 1,
        },
        {
            "threshold": 1,
            "contour_area": 2000,
            "deviation": 1,
        },
    ]

    for i, config in enumerate(configs):
        motionDetect = MotionDetect(
            threshold=config["threshold"],
            contour_area=config["contour_area"],
            deviation=config["deviation"],
        )
        motionDetect.process_video('Lab5/files/kotik.mp4', f'Lab5/files/2_proc_{i}.mp4')