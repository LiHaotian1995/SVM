import numpy as np
import matplotlib.pyplot as plt

# a data structure
class optStruct:
    def __init__(self, dataMatIn, labelMatIn, C, tolerance):
        self.dataMat = dataMatIn
        self.labelMat = labelMatIn
        self.C = C
        self.tolerance = tolerance
        # set the shape[0](first dim) of dataMatIn
        self.m = np.shape(dataMatIn)[0]
        # set the alpha (initialize to m * 1, total = 0)
        self.alpha = np.mat(np.zeros((self.m,1)))
        # set b = 0 (initialize)
        self.b = 0
        # set a error cache(matrix size: m*2)
        self.errorCache = np.mat(np.zeros((self.m,2)))


def loadDataSet():
    dataMat = []; labelMat = []
    f = open('/Users/lihaotian/PycharmProjects/SVM/testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    # print("\n dataMat: \n" + str(dataMat))
    # print("\n labelMat: \n" + str(labelMat))
    return dataMat, labelMat


# calculate the error of Ek for every k
def calculateEk(oS, k):
    fXk = float(np.multiply(oS.alpha, oS.labelMat).transpose() * \
                (oS.dataMat * oS.dataMat[k,:].transpose())) + oS.b

    Ek = fXk - float(oS.labelMat[k])

    return Ek


# use random method choose alpha[j]
def selectAlphaJRandom(i, m):
    j = i
    while (j == i):
        # random select alpha[j]
        j = int(np.random.uniform(0,m))

    return j


# use Inspired method choose alpha[j] according to alpha[i]
def selectAlphaJ(i, oS, Ei):
    maxK = -1; Ej = 0; maxDeltaE = 0
    # according to the alpha[i], update the errorCache[i,:] to [1,Ei]
    oS.errorCache[i] = [1,Ei]
    # from all non-zero errorCache choose, and form the:  Non-zero ErrorCacheList,
    # here use A means transpose matrix to array;
    # it will return two array about (oS.errorCache[:,0].A)(it's an array) to validErrorCacheList,
    # first dim is the line number of nonzero(oS.errorCache[:,0].A), second is col of nonzero()
    # and we only use the first dim(use [0]) of nonzero(oS.errorCache[:,0].A)
    validErrorCacheList = np.nonzero(oS.errorCache[:,0].A)[0]
    # validErrorCacheList is an array: array([...0, 1, 2...]), the location number of nonzero-errorCache
    # choose alpha[j] from the Non-zero ErrorCacheList
    if (len(validErrorCacheList)) > 1:
        for k in validErrorCacheList:
            if k == i:
                continue
            Ek = calculateEk(oS, k)
            # the Inspired Method
            deltaE = np.abs(Ei - Ek)
            # select the max DeltaE
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectAlphaJRandom(i, oS.m)
        Ej = calculateEk(oS, j)

    return j, Ej


def updateEk(oS, k):
    Ek = calculateEk(oS, k)
    oS.errorCache[k] = [1, Ek]


def clipAlphaJ(alphaJ, L, H):
    if alphaJ > H:
        alphaJ = H
    elif alphaJ < L:
        alphaJ = L

    return alphaJ


def innerSMO(i, oS):
    Ei = calculateEk(oS, i)
    if ((oS.labelMat[i]*Ei < - oS.tolerance) and (oS.alpha[i] < oS.C)) or\
            ((oS.labelMat[i]*Ei > oS.tolerance) and (oS.alpha[i] > 0)):
        j, Ej = selectAlphaJ(i, oS, Ei)

        alphaOldI = oS.alpha[i].copy(); alphaOldJ = oS.alpha[j].copy()

        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, alphaOldJ - alphaOldI); H = min(oS.C, oS.C + alphaOldJ - alphaOldI)
        else:
            L = max(0, alphaOldI + alphaOldJ - oS.C); H = min(oS.C, alphaOldI + alphaOldJ)
        if (L == H):
            print("\n L == H = %f" % L)
            return 0


        eta = oS.dataMat[i,:] * oS.dataMat[i,:].transpose() + \
              oS.dataMat[j,:] * oS.dataMat[j,:].transpose() - \
              2.0 * oS.dataMat[i,:] * oS.dataMat[j,:].transpose()
        if (eta <= 0):
            print("\n eta <= 0")
            return 0

        oS.alpha[j] = oS.labelMat[j] * (Ei - Ej) / eta + alphaOldJ
        oS.alpha[j] = clipAlphaJ(oS.alpha[j], L, H)
        updateEk(oS, j)

        if (np.abs(oS.alpha[j] - alphaOldJ) <= 0.00001):
            print("\n alpha[j] is not changing")
            return 0

        oS.alpha[i] = alphaOldI + oS.labelMat[i] * oS.labelMat[j] * (alphaOldJ - oS.alpha[j])
        updateEk(oS, i)

        # calculate the new Bi and Bj, and then judge the alpha[i]
        newBi = oS.b - Ei - oS.labelMat[i] * (oS.alpha[i] - alphaOldI) * \
                oS.dataMat[i, :] * oS.dataMat[i, :].transpose() - \
                oS.labelMat[j] * (oS.alpha[j] - alphaOldJ) * oS.dataMat[i, :] * \
                oS.dataMat[j, :].transpose()

        newBj = oS.b - Ej - oS.labelMat[i] * (oS.alpha[i] - alphaOldI) * \
                oS.dataMat[i, :] * oS.dataMat[j, :].transpose() - \
                oS.labelMat[j] * (oS.alpha[j] - alphaOldJ) * oS.dataMat[j, :] * \
                oS.dataMat[j, :].transpose()
        # if alpha[i] satisfy, update and select b
        if (oS.alpha[i] > 0) and (oS.alpha[i] < oS.C):
            oS.b = newBi
        elif (oS.alpha[j] > 0) and (oS.alpha[j] < oS.C):
            oS.b = newBj
        else:
            oS.b = (newBi + newBj) / 2.0
        return 1

    else:
        return 0


def SMO(dataMatIn, labelMatIn, C, tolerance, maxIter, kernelFunction = ('line', 0)):
    # put each value into the structure
    oS = optStruct(np.mat(dataMatIn), np.mat(labelMatIn).transpose(), C, tolerance)
    # entireSet means if check the whole alpha set or not
    iter = 0; entireSet = True; alphaPairsChange = 0

    while (iter < maxIter) and ((alphaPairsChange > 0) or (entireSet)):
        alphaPairsChange = 0
        # check the whole alpha set(means entireSet = True)
        if entireSet:
            for i in range(oS.m):
                alphaPairsChange += innerSMO(i, oS)
                print("\nBrowse the whole set, iter: %d, i: %d, pairs change: %d" % \
                      (iter, i, alphaPairsChange))
                iter += 1
        # check the alpha set which is not in the bound
        else:
            # calculate all notInBoundAlpha[i] as a list
            notInBoundAlphaI = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0]
            for i in notInBoundAlphaI:
                alphaPairsChange += innerSMO(i, oS)
                print("\nNot in bound, iter: %d, i: %d, pairs change: %d" % (iter, i, alphaPairsChange))
                iter += 1

        if entireSet:
            entireSet = False
        elif (alphaPairsChange == 0):
            entireSet = True

        print("\nIteration count: %d" % iter)

    return oS.alpha, oS.b


def calculateWeights(dataMatIn, labelMatIn, alpha):
    dataMat = np.mat(dataMatIn); labelMat = np.mat(labelMatIn).transpose()
    m, n = np.shape(dataMat)
    weights = np.zeros((n,1))

    for i in range(m):
        weights += np.multiply(alpha[i] * labelMat[i], dataMat[i,:].transpose())

    return weights


# check out the picture
def plotDataSet(dataMat, labelMat, alpha, weights, b):
    dataMat1 = []; dataMatNo1 = []
    m, n = np.shape(np.mat(labelMat).transpose())

    for i in range(m):
        if labelMat[i] < 0:
            dataMatNo1.append(dataMat[i])
        else:
            dataMat1.append(dataMat[i])
    # print("\n dataMat1: \n" + str(dataMat1))
    # print("\n dataMatNo1: \n" + str(dataMatNo1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([data[0] for data in dataMat1], [data[1] for data in dataMat1])
    ax.scatter([data[0] for data in dataMatNo1], [data[1] for data in dataMatNo1], c='r')

    # plot the Support Vector
    for i in range(m):
        if alpha[i] > 0:
            x, y = dataMat[i]
            ax.scatter([x], [y], s = 150, color='', edgecolor='red')

    # plot the hyperPlane
    x = np.arange(0.0, 8.0)
    y = (- weights[0,0] * x - float(b)) / weights[1,0]
    ax.plot(x,y)

    plt.show()



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # print("\n dataMat: \n" + str(dataMat))
    # print np.mat(labelMat).transpose()

    alpha, b = SMO(dataMat, labelMat, 0.6, 0.001, 40)
    print("\n alpha: \n" + str(alpha))
    print("\n b = " + str(float(b)))

    weights = calculateWeights(dataMat, labelMat, alpha)
    # print("\n weights: \n" + str(weights))
    # print("\n shape: " + str(np.shape(weights)))
    # print("\n weights: \n" + str(weights[0,0]))
    # print("\n weights: \n" + str(weights[1,0]))

    plotDataSet(dataMat, labelMat,alpha, weights, b)