import os
import matplotlib.pyplot as plt
import json
import argparse

#//////////////////////// ARGUMENT PARSER \\\\\\\\\\\\\\\\\\\\\\\\
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="/mnt//Ahmad//Models//Segmentation//trained_model//our_final_SN_all",
                    help="Path to save model")

if __name__ == '__main__':
    args = parser.parse_args()
    f = open(os.path.join(args.model_dir, 'train_logs.txt'))
    lines_tr = f.readlines()
    f.close()

    f = open(os.path.join(args.model_dir, 'train_logs.txt'))
    lines_tr = f.readlines()
    f.close()
    lines_tr = [line for i,line in enumerate(lines_tr) if i > 0]

    f = open(os.path.join(args.model_dir, 'valid_logs.txt'))
    lines_val = f.readlines()
    f.close()
    lines_val = [line for i,line in enumerate(lines_val) if i > 0]
    scores_tr = []
    for line in lines_tr:
        try:
            scores_tr.extend([json.loads(line[4:].replace("\'", "\""))])
        except:
            scores_tr.extend([json.loads(line[5:].replace("\'", "\""))])

    scores_val = []
    for line in lines_val:
        try:
            scores_val.extend([json.loads(line[4:].replace("\'", "\""))])
        except:
            scores_val.extend([json.loads(line[5:].replace("\'", "\""))])


    iou_train = [float(dic['IOU Score']) for dic in scores_tr]
    iou_val = [float(dic['IOU Score']) for dic in scores_val]

    loss_train = [float(dic['LOSS']) for dic in scores_tr]
    loss_val = [float(dic['LOSS']) for dic in scores_val]

    plt.figure(figsize=(30,10))
    plt.subplot(1,2,1)
    plt.plot(iou_train)
    plt.plot(iou_val)
    plt.title("Model IOU Score over training")
    plt.xlabel('Epoch')
    plt.ylabel('IOU Score')
    plt.legend(['Train', 'Val'], loc = 'upper left')

    plt.subplot(1,2,2)
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.title("Model Loss over training")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc = 'upper left')
    plt.show()