# Test for only in Mac OS X
# Required 'convert' program
# Hyunmin Kim

import os
import argparse

def run(pdf, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cmd = 'convert -density 300 -trim -quality 100 %s[0-29] %s/sample.png'%(pdf,outdir)
    print (cmd)

    
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pdf", required = True,
        help = "input pdf file")
    ap.add_argument("-o", "--out_dir", required = False, default="./",
        help = "Path to the images to be scanned (Multiple file *.png)")

    args = vars(ap.parse_args())


    run(args["pdf"], args["out_dir"])
        

if __name__ == '__main__':
    main()
