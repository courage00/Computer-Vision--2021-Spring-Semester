import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import collections

train_path=os.path.join('hw5_data','train')
test_path=os.path.join('hw5_data','test')
class_names = os.listdir(train_path)

def get_feature(filepath):
    features = []
    for c in class_names:
        class_path = os.path.join(filepath, c)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            if image_name.endswith('.jpg'):
                img_path = os.path.join(class_path, image_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (16, 16))
                feature = np.reshape(img, (1, -1))
                # normalize
                feature = feature - feature.mean()
                feature /= np.linalg.norm(feature)
                features.append(feature[0])

    return np.asarray(features)

def knn(indices):
    indices = indices // 100
    k=indices.shape[1]
    acc=0
    for i in range(len(indices)):
        target_class=i//10
        predict_class=collections.Counter(indices[i]).most_common(1)[0][0]
        if target_class==predict_class:
            acc+=1

    return acc/len(indices)

if __name__=='__main__':

    features = get_feature(train_path)
    test_features = get_feature(test_path)
    distances = cdist(test_features, features, 'euclidean')
    # knn
    acc = 0

    indices=np.argsort(distances,axis=1)
    for k in range(1,10):
        acc=knn(indices[:,:k])
        print('k:',k,f'acc: {acc:.2f}')
    acc=knn(indices[:,:20])
    print('k:',20,f'acc: {acc:.2f}')
    acc=knn(indices[:,:100])
    print('k:',100,f'acc: {acc:.2f}')    