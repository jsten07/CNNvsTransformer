import shutil
import os
import argparse

def subset_patches(im_dir, out_dir, patch_name):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    for path, subdirs, files in os.walk(im_dir):
        # dirname = path.split(os.path.sep)[-1]
        # print(subdirs)

        images = sorted(os.listdir(path))
        for i, image_name in enumerate(images):
            for p in patch_name:
                if p in image_name:
                    # print(os.path.join(path,image_name))
                    print(image_name)
                    shutil.copy2(os.path.join(path,image_name), out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='directory containing images to be patchified', required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save the patches to', required=True)
    parser.add_argument('--patch_name', '-p', action='append', default=None, help='substring name', required=True)
    opt = parser.parse_args()

    subset_patches(opt.in_dir, opt.out_dir, opt.patch_name)
    