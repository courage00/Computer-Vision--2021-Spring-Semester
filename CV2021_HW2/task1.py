import os
import cv2
import argparse
from PIL import Image
from hybrid_image import Hybrid
import matplotlib.pyplot as plt


def load_image(filename):
    """Load the image in 'PIL.Image.Image' format"""
    image = Image.open(filename)

    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', '--filename1', type=str,
                        help='specify the image1 file for high-pass', default='0_1.bmp')
    parser.add_argument('-f2', '--filename2', type=str,
                        help='specify the image2 file for low-pass', default='0_2.bmp')
    parser.add_argument('--dir', type=str, help='specify the directory for loading image files',
                        default='images')
    parser.add_argument('-c1', '--cutoff_h', type=int,
                        help='specify the cutoff frequency of high-pass filter', default=20)
    parser.add_argument('-c2', '--cutoff_l', type=int,
                        help='specify the cutoff frequency of low-pass filter', default=10)
    parser.add_argument('-s', '--fftshift', action='store_true',
                        help='whether use fftshift to center the transform', default=False)
    parser.add_argument('-m', '--multiple', action='store_true',
                        help='whether compute with multiple pairs of images or just one', default=False)
    args = parser.parse_args()

    d0_h, d0_l = args.cutoff_h, args.cutoff_l
    multiple, shift = args.multiple, args.fftshift
    img_path = args.dir
    # save_dir = f"{img_path.split('/')[0]}_result"
    save_dir = os.path.join(img_path, "task1")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if multiple:
        img_name = os.listdir(img_path)
        tmp_img = img_name.copy()
        for name in tmp_img:
            print(f'{name}')
            if not (name.endswith(".bmp") or name.endswith(".jpg")):
                img_name.remove(name)
                print(f'remove {name}')
        img_name = sorted(img_name)

        for i in range(len(img_name)):
            img1 = load_image(os.path.join(img_path, img_name[i]))
            j = i+1 if i%2==0 else i-1
            img2 = load_image(os.path.join(img_path, img_name[j]))
            
            task1 = Hybrid(img1, img2, shift)
            print(f"Computing {i}, {j}")
            hybrid_gaussian, hybrid_ideal = task1.DoHybrid(d0_h, d0_l)

            tmp_name = img_name[i].split('.')[0]
            filename = f"{tmp_name.split('_')[0]}{i%2}"
            s = '_s' if shift else ''
            save_path = os.path.join(
                save_dir, f"{filename}_gaussianhybrid_{d0_h}_{d0_l}{s}.bmp")
            cv2.imwrite(save_path, hybrid_gaussian)
            save_path = os.path.join(
                save_dir, f'{filename}_idealhybrid_{d0_h}_{d0_l}{s}.bmp')
            cv2.imwrite(save_path, hybrid_ideal)
    else:
        img1 = load_image(os.path.join(args.dir, args.filename1))
        img2 = load_image(os.path.join(args.dir, args.filename2))

        plt.title("High image")
        plt.imshow(img1)
        plt.show()
        plt.title("Low image")
        plt.imshow(img2)
        plt.show()

        task1 = Hybrid(img1, img2, shift)
        print("Computing. . .")
        hybrid_gaussian, hybrid_ideal = task1.DoHybrid(d0_h, d0_l)

        filename = f"{args.filename1.split('.')[0]}{args.filename2.split('.')[0]}"
        s = '_s' if shift else ''
        save_path = os.path.join(
            save_dir, f"{filename}_gaussianhybrid_{d0_h}_{d0_l}{s}.bmp")
        cv2.imwrite(save_path, hybrid_gaussian)
        save_path = os.path.join(
            save_dir, f'{filename}_idealhybrid_{d0_h}_{d0_l}{s}.bmp')
        cv2.imwrite(save_path, hybrid_ideal)

        # img = cv2.cvtColor(hybrid_gaussian.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # plt.imshow(img)
        # plt.show()

        # img = cv2.cvtColor(hybrid_ideal.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # plt.imshow(img)
        # plt.show()