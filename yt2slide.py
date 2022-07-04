import cv2
import os
import argparse
from PIL import Image
from vidgear.gears import CamGear
from image_similarity_measures.quality_metrics import ssim

def main(args):
    images = []
    stream = CamGear(source=args.youtube_link, stream_mode=True, logging=True, STREAM_RESOLUTION=args.quality).start()
    current_frame = 0
    capture_rate = args.rate*30
    current_image = []
    last_frame = []
    is_first = False

    while True:
        frame = stream.read()
        if frame is None:
            break
        last_frame = frame

        if current_frame % capture_rate == 0:
            if not is_first:
                current_image = frame
                is_first = True
            else:
                ratio = ssim(current_image, frame)
                if ratio <= args.diff:
                    print(f"Add new slide at frame {current_frame}")
                    images.append(Image.fromarray(current_image, 'RGB'))
                    current_image = frame

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        current_frame += 1

    ratio = ssim(current_image, last_frame)
    if ratio <= args.diff:
        images.append(Image.fromarray(last_frame, 'RGB'))

    images[0].save(
        os.path.join(os.getcwd(), f"{args.output}.pdf"), "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
    )

    cv2.destroyAllWindows()
    stream.stop()
    stream.stop()


parser = argparse.ArgumentParser(description='Convert Youtube talk/conf to slide')
parser.add_argument('--rate', required=False, type=int, default=15, help='Capture screen every <rate> seconds. Default to %(default)s secs.')
parser.add_argument('youtube_link', type=str, help='Youtube link of talks/conf')
parser.add_argument('--quality', required=False, default='best', type=str, help='Youtube video quality: [144p, 240p, 360p, 480p, 720p, 1080p, best, worst]. Default to %(default)s.')
parser.add_argument('--diff', required=False, default=0.95, type=float, help='Ratio that, if smaller, will be treated as two different slides. Default: %(default)s.')
parser.add_argument('output', type=str, help='Name of PDF file to write to current directory.')

args = parser.parse_args()

if __name__ == "__main__":
    main(args)