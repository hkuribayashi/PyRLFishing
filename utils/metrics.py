import numpy as np

def getPrecision(tp, fp):
    return tp / (tp + fp)


def getRecall(tp, fn):
    return tp / (tp + fn)


def getAccuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def getFalsePositiveRate(fp, tn):
    return fp/(fp + tn)


def getFalseNegativeRate(fn, tp):
    return fn/(fn+tp)


def getF1Score(tp, fp, fn):
    precision = getPrecision(tp, fp)
    recall = getRecall(tp, fn)
    return 2 * (precision*recall/(precision+recall))


def processaResultados(resultados):
    fpr = []
    fnr = []
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    for resultado in resultados:
        fpr.append(resultado['fpr'])
        fnr.append(resultado['fnr'])
        precision.append(resultado['precision'])
        recall.append(resultado['recall'])
        f1_score.append(resultado['f1score'])
        accuracy.append(resultado['accuracy'])

    resultado_final = dict()
    resultado_final['fpr_media'] = np.average(fpr)
    resultado_final['fpr_desvio'] = np.std(fpr)
    resultado_final['fnr_media'] = np.average(fnr)
    resultado_final['fnr_desvio'] = np.std(fnr)
    resultado_final['precision_media'] = np.average(precision)
    resultado_final['precision_desvio'] = np.std(precision)
    resultado_final['recall_media'] = np.average(recall)
    resultado_final['recall_desvio'] = np.std(recall)
    resultado_final['f1score_media'] = np.average(f1_score)
    resultado_final['f1score_desvio'] = np.std(f1_score)
    resultado_final['accuracy_media'] = np.average(accuracy)
    resultado_final['accuracy_desvio'] = np.std(accuracy)

    return resultado_final


