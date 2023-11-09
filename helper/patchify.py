import argparse
import cv2
import os

def make_patches(im_dir, out_dir, patch_size = 512, step = 500):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    # image_dataset = []  
    for path, subdirs, files in os.walk(im_dir):
        subdirs.sort()
        dirname = path.split(os.path.sep)[-1]
        images = sorted(os.listdir(path))  #List of all image names in this subdirectory
        
        # TODO: create out folder if not exists
        for i, image_name in enumerate(images):
            if image_name.endswith(".tif") or image_name.endswith(".png") or image_name.endswith(".jpg"):   # Only read images, not xml files or other stuff...

                image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                
                for x in range(0, image.shape[1], step): 
                    for y in range(0, image.shape[0], step):
                        if (x+patch_size > image.shape[1]): # check if x is out of bounds
                            x = image.shape[1]-patch_size
                        if (y+patch_size > image.shape[0]): # check if y is out of bounds
                            y = image.shape[0]-patch_size
                        
                        im_name = image_name[:-4]+'_'+str(x)+'_'+str(y)+'.tif'
                        im_path = os.path.join(out_dir,im_name)
                        
                        if not os.path.isfile(im_path):
                            print(im_name)
                            tile = image[y:y+patch_size, x:x+patch_size]
                            cv2.imwrite(im_path, tile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='directory containing images to be patchified', required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save the patches to', required=True)
    parser.add_argument('--patch_size', type=int, default=512, help='size of the square patches to be created in pixels per side')
    parser.add_argument('--stride', type=int, default=500, help='number of pixels to move the window creating the patches')
    opt = parser.parse_args()

    make_patches(opt.in_dir, opt.out_dir, opt.patch_size, opt.stride)