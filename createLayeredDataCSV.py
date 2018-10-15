import csv
import os
import absl.app as app

def main(argv):
    with open('layeredTifData.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        # separate columns for signal and target
        writer.writerow(['path_tif_signal', 'path_tif_target'])
        # sort the list of files
        allFiles = sorted(os.listdir('/scratch/jagadish/pytorch_fnet/data/nikon_label_free'))
        # filename prefix 
        currPrefix = ''
        # loop over files to create a dictionary of the slices per file
        for filename in allFiles:
            if 'signal' in filename:
                writer.writerow(['/scratch/jagadish/pytorch_fnet/data/nikon_label_free/' + filename, 
                                 '/scratch/jagadish/pytorch_fnet/data/nikon_label_free/' + filename.replace('signal', 'target')])

if __name__ == "__main__":
    app.run(main)
