import numpy as np
from imageProcessor import ImagePreprocessor, ImgGenerator


def predict(model, gen, scaled_size, nChannel):
    y_predict = []
    y_predict_proba = []

    for batch_num in range(9999999):
        batch = next(gen, -1)
        if batch == -1:
            break
        test_X = np.array(batch).reshape((len(batch), scaled_size[0], scaled_size[1], nChannel))
        result = model.predict(test_X, verbose=0)
        y_predict.extend([np.argmax(x) for x in result])
        y_predict_proba.extend([np.max(x) for x in result])
        del batch
    return y_predict, y_predict_proba


def main_pipe(model, full_histones, params, scaled_size=(500, 500)):
    if isinstance(full_histones, dict):
        full_histones = [full_histones]

    batch_size = params['batch_size']
    nChannel = params['nChannel']
    amp = params['amp']
    hybrid_cutoff = params['hybrid_cutoff']
    immobile_cutoff = params['immobile_cutoff']

    for g_num, histones in enumerate(full_histones):
        key_list = list(histones.keys())
        ImagePreprocessor.make_channel(histones, immobile_cutoff=immobile_cutoff,
                                       hybrid_cutoff=hybrid_cutoff, nChannel=nChannel)
        gen = ImgGenerator.conversion(histones, key_list=key_list, scaled_size=scaled_size,
                                      batch_size=batch_size, amp=amp, eval=False)
        batch_y_predict, batch_y_predict_proba = predict(model, gen, scaled_size, nChannel)
        for index, histone in enumerate(key_list):
            histones[histone].set_predicted_label(batch_y_predict[index])
            histones[histone].set_predicted_proba(batch_y_predict_proba[index])
