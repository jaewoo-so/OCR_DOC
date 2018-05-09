import os, glob
import cv2

def main():
    #convert_pdf()
    img_path = '/Users/hyunmin/Downloads/project-handwriting/images'
    img_fn_pattern = img_path + '/*/*.png'

    imgfiles = glob.glob(img_fn_pattern)
    imgfiles.sort(key = lambda x: x.split('-')[2].split('.')[0])
  
    out_fn_path = '/Users/hyunmin/Downloads/project-handwriting/images_out'
    if not os.path.exists(out_fn_path):
        os.makedirs(out_fn_path)
 
    for c,img_file in enumerate(imgfiles):
        label = img_file.split('/')[-2]
        label_out_path = out_fn_path + '/' + str(label)
        if not os.path.exists(label_out_path):
            os.makedirs(label_out_path)
        out_img_fn = label_out_path + '/' + str(c) + '.png'

        img = cv2.imread(img_file)
        header_section = img[280:400, 700:1800] 
        cv2.imwrite(out_img_fn,header_section)

if __name__ == '__main__':
    main()
