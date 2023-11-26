import cv2
import os
import glob


def __convert_images_to_video(directory: str,
                              fps: int,
                              output_filename: str):
    images = glob.glob(os.path.join(directory, '*.jpg'))

    images.sort()

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

    for img in images:
        frame = cv2.imread(img)
        out.write(frame)

    out.release()


def main():
    __convert_images_to_video('.', fps=5, output_filename='output_video.mp4')

if __name__ == '__main__':
    main()
