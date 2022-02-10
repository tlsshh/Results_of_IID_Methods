import numpy as np
import pickle
import sys
import json
import h5py

def compute_whdr(reflectance, judgements, delta=0.1):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    error_equal_sum = 0.0
    error_inequal_sum = 0.0

    weight_sum = 0.0
    weight_equal_sum = 0.0
    weight_inequal_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0.0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker == 'E':  
            if darker != alg_darker:
                error_equal_sum += weight

            weight_equal_sum += weight
        else:
            if darker != alg_darker:
                error_inequal_sum += weight

            weight_inequal_sum += weight

        if darker != alg_darker:
            error_sum += weight

        weight_sum += weight

    if weight_sum:
        return (error_sum / weight_sum), error_equal_sum/( weight_equal_sum + 1e-10), error_inequal_sum/(weight_inequal_sum + 1e-10)
    else:
        return None


file_name = "./test_img_batch.p"
pred_dir = './release_iiw/'
images_list = pickle.load( open( file_name, "rb" ) )

root = '/home/zl548/phoenix24/phoenix/S6/zl548/'

count = 0.0
whdr_sum =0.0

for i in range(0 , 3):
    img_list = images_list[i]

    for img_path in img_list:

        judgement_path = root + "/IIW/iiw-dataset/data/" + img_path.split('/')[-1][0:-6] + 'json'
        judgements = json.load(open(judgement_path))

        pred_path = pred_dir + img_path.split('/')[-1]

        print('pred_path', pred_path)

        hdf5_file_read = h5py.File(pred_path,'r')
        pred_R = hdf5_file_read.get('/prediction/R')
        pred_R = np.array(pred_R)

        pred_S = hdf5_file_read.get('/prediction/S')
        pred_S = np.array(pred_S)

        hdf5_file_read.close()

        # print(pred_R.shape)
        # sys.exit()

        whdr, whdr_eq, whdr_ineq = compute_whdr(pred_R, judgements)
        whdr_sum += whdr
        count+=1.0
        whdr_mean =whdr_sum/count

        print(whdr_mean)


whdr_mean = whdr_sum/count

print('whdr_mean %f', whdr_mean)

sys.exit()




