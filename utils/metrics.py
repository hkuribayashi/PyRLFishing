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
