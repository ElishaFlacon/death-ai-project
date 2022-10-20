import cv2
from deepface import DeepFace


# НУ МЕЙН ОН И В АФРИКЕ МЕЙН
def main():
    try:
        # РАСПОЗНАЕМ ТОЧКИ ЛИЦА
        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # ПОДКЛЮЧАЕМСЯ К ВЕБ КАМЕРЕ
        wc = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if wc.isOpened():
            webcam(wc, faceCascade)
        if not wc.isOpened():
            wc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if wc.isOpened():
                webcam(wc, faceCascade)
        if not wc.isOpened():
            print('ERROR, CHECK YOUR WEBCAM')
    except Exception as e:
        print('main' + e)


# ФУНКЦИЯ ЗАПУСКА КАМЕРЫ
def webcam(wc, faceCascade):
    try:
        while True:
            # ЗАПУСК КАМЕРЫ
            ret, frame = wc.read()
            # ОБРАБАТЫВАЕМ КАДРЫ
            predictions = DeepFace.analyze(
                frame, actions=['emotion'], enforce_detection=False)

            # МЕНЯЕМ ЦВЕТ КАРТИНКИ
            gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # РАСПОЗНАЕМ ТОЧКИ ЛИЦА
            faces = faceCascade.detectMultiScale(gray_face, 1.1, 4)

            # РИСУЕМ ПРЯМОУГОЛЬНИК ВОКРУГ ЛИЦА
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # ШРИФТ
            font = cv2.FONT_HERSHEY_SIMPLEX

            # РИСУЕМ ТЕКСТ НА КАДРАХ
            cv2.putText(frame, predictions['dominant_emotion'],
                        (5, 50), font, 1, (110, 0, 255), 2, cv2.LINE_4)

            # ВЫВОДИМ КАРТИНКУ
            cv2.imshow('Video', frame)

            # КНОПКА СТОП
            if cv2.waitKey(2) & 0xFF == ord('`') or cv2.waitKey(2) & 0xFF == ord('0'):
                close_wc(wc)
                break
    except Exception as e:
        print('webcam' + e)


# ЗАКРЫВАЕМ ОКНА
def close_wc(wc):
    try:
        wc.release()
        cv2.destroyAllWindows()
        print('!STOP!')
    except Exception as e:
        print('close' + e)


# ЗАПУСК
if __name__ == '__main__':
    try:
        print('!START!')
        main()
    except Exception as e:
        print(e)
