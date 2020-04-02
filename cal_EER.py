import csv
import numpy as np
if __name__ == '__main__':
    EERs=[]
    for cl in range(50):
        print(cl)
        threshold = 0
        fn, fp = 0, 0
        real = 135.0
        fake = 270.0
        while fp/fake>=fn/real or (fp==0 and fn==0):
            fn, fp, tp, tn = 0, 0, 0, 0
            f = open('sim/similarity_%02d.csv'%cl, 'r', encoding='utf-8')
            reader = csv.reader(f)
            cnt=0
            for line in reader:
                cnt=cnt+1
                for i in range(len(line)-2):
                    if float(line[i+2])>threshold:
                        if line[0][0]=='f':
                            fp=fp+1
                    else:
                        if line[0][0]=='r':
                            fn=fn+1
            f.close()
            threshold=threshold+0.01
        EERs.append(fp/(2*cnt*(len(line)-2)/3))
        print("the number of subjects for test:%d"%(cnt/3))
        print("the len of sequence for test:%d"%(len(line)-2))
    print("EER:")
    for EER in EERs:
        print(EER)
    print("Average EER:")
    print(np.mean(EERs))
    
