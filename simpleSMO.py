import matplotlib.pyplot as plt
import numpy as np


def loadDataSet():
    dataMat = []; labelMat = []
    f = open('/Users/lihaotian/PycharmProjects/SVM/testSet.txt')
    for line in f.readlines():
        lineArr = line.strip().split('\t')
        # print lineArr
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    print("\n dataMat: \n" + str(dataMat))
    print("\n labelMat: \n" + str(labelMat))
    return dataMat, labelMat


# check out the picture
def plotDataSet(dataMat, labelMat, alpha, b, supportVector, svCount):
    dataMat1 = []; dataMatNo1 = []
    m, n = np.shape(np.mat(labelMat).transpose())

    for i in range(m):
        if labelMat[i] < 0:
            dataMatNo1.append(dataMat[i])
        else:
            dataMat1.append(dataMat[i])
    print("\n dataMat1: \n" + str(dataMat1))
    print("\n dataMatNo1: \n" + str(dataMatNo1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([data[0] for data in dataMat1], [data[1] for data in dataMat1])
    ax.scatter([data[0] for data in dataMatNo1], [data[1] for data in dataMatNo1], c='r')

    # plot the Support Vector
    for i in range(svCount):
        ax.scatter(np.mat(supportVector)[i,0][0], np.mat(supportVector)[i,0][1], \
                   color='', marker='o', edgecolor='g', s=200)

    # plot the hyperPlane
    alphaIyI = np.mat(np.multiply(np.mat(labelMat).transpose(), alpha)).transpose()
    # print("\n alphaIyI: \n" + str(alphaIyI))
    # print np.shape(alphaIyI)
    w = np.mat(np.dot(alphaIyI, dataMat))
    # print w
    # print w[0,0]
    # print w[0,1]
    x = np.arange(0.0, 8.0)
    y = (- w[0,0] * x - b[0,0]) / w[0,1]
    ax.plot(x,y)

    plt.show()


def selectAlphaJ(i, m):
    j = i
    while (j == i):
        # random selected alpha_J
        j = int(np.random.uniform(0,m))

    return j


def clipAlpha(alphaJ, H, L):
    if alphaJ > H:
        alphaJ = H
    if alphaJ < L:
        alphaJ = L

    return alphaJ


def simpleSMO(dataMatIn, classLabels, C, toleranceRate, maxIter):
    dataMatrix = np.mat(dataMatIn); labelMatrix = np.mat(classLabels).transpose()
    # print("\n dataMatrix: \n" + str(dataMatrix))
    # print("\n labelMatrix: \n" + str(labelMatrix))
    # print labelMatrix.shape

    m, n = np.shape(dataMatrix); b = 0
    # print("\n m: " + str(m)); print("\n n: " + str(n))

    alpha = np.mat(np.zeros((m,1)))
    # print("\n alpha: \n" + str(alpha))

    iter = 0
    while (iter < maxIter):
        # count how many alpha changed in the end
        alphaPairsChange = 0

        for i in range(m):
            # get the value of fXi
            # multiply means Multiplies the corresponding element: 1 to 1, 2 to 2(location)
            # here is the simple version, so use Matrix Multiplication replace kernel function
            fXi = float(np.multiply(alpha, labelMatrix).transpose() * \
                        (dataMatrix * dataMatrix[i,:].transpose())) + b
            # calculate the errorValue between predictValue and reallyValue
            Ei = fXi - float(labelMatrix[i])

            # select the first alpha[i]
            if ((labelMatrix[i]*Ei < -toleranceRate) and (alpha[i] < C)) or \
                    (labelMatrix[i]*Ei > toleranceRate and (alpha[i] > 0)):
                # according to alpha_I get the alpha_J(random, so it's a simple SMO)
                j = selectAlphaJ(i,m)
                # calculate fXj & Ej, same as fXi & Ei
                fXj = float(np.multiply(alpha, labelMatrix).transpose() * \
                            (dataMatrix * dataMatrix[j,:].transpose())) + b

                Ej = fXj - float(labelMatrix[j])

                # copy the alpha[i & j] as alpha Old Value(i & j)
                alphaOldI = alpha[i].copy(); alphaOldJ = alpha[j].copy()

                # split the range of alpha[J]
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0, alphaOldJ - alphaOldI)
                    H = min(C, C + alphaOldJ - alphaOldI)
                else :
                    L = max(0, alphaOldJ + alphaOldI - C)
                    H = min(C, alphaOldJ + alphaOldI)

                # if L == H, print "L ==H" and break this cycle
                if L == H:
                    print("\nL == H = %f" % L)
                    continue

                # calculate Eta(n), still use Matrix Multiplication replace the kernel function
                eta = dataMatrix[i,:] * dataMatrix[i,:].transpose() + \
                      dataMatrix[j,:] * dataMatrix[j,:].transpose() - \
                      2.0 * dataMatrix[i,:] * dataMatrix[j,:].transpose()
                # if eta <= 0, it can not calculate, stop and begin next i cycle
                if eta <= 0:
                    print("\neta <= 0")
                    continue

                # calculate new alpha[j]
                alpha[j] = labelMatrix[j] * (Ei - Ej)/eta + alphaOldJ
                # and put this new alpha_J into the function: clipAlpha(),
                # to adjust the range of new alpha_J
                alpha[j] = clipAlpha(alpha[j], H, L)

                # calculate the absolute value between alpha[j] & alphaOldJ
                # if the absolute value is <= 0.00001, we can think the alpha[j] is no longer change
                # so stop and begin next i cycle
                if (np.abs(alpha[j] - alphaOldJ) <= 0.00001):
                    print("\nalpha[j] not changed enough")
                    continue

                # if alpha[j] is changed, so we can calculate the new alpha[i]
                alpha[i] = alphaOldI + labelMatrix[j] * labelMatrix[i] * (alphaOldJ - alpha[j])


                # calculate the new Bi and Bj, and then judge the alpha[i]
                newBi = b - Ei - labelMatrix[i] * (alpha[i] - alphaOldI) * \
                     dataMatrix[i,:] * dataMatrix[i,:].transpose() - \
                     labelMatrix[j] * (alpha[j] - alphaOldJ) * dataMatrix[i,:] * \
                     dataMatrix[j,:].transpose()

                newBj = b - Ej - labelMatrix[i] * (alpha[i] - alphaOldI) * \
                     dataMatrix[i,:] * dataMatrix[j,:].transpose() - \
                     labelMatrix[j] * (alpha[j] - alphaOldJ) * dataMatrix[j,:] * \
                     dataMatrix[j,:].transpose()


                if ( 0 < alpha[i]) and (C > alpha[i]):
                    b = newBi
                elif (0 < alpha[j]) and (C > alpha[j]):
                    b= newBj
                else:
                    b = (newBi + newBj)/2.0

                # calculate the number of how many alpha changed
                alphaPairsChange += 1
                # check and print the times
                print("\niter: %d, i: %d, alphaPairsChange: %d" % (iter, i, alphaPairsChange))

        # if there is no alpha changed, so add the number of iter(iter++)
        if (alphaPairsChange == 0):
            iter += 1
        else:
            iter = 0
        # print the iteration times
        print("\nIteration number: %d" % iter)

    return alpha, b, m


# show the support vector(SV)
def findSupportVector(alpha, m):
    supportVector = []
    for i in range(m):
        if alpha[i] > 0.0:
            # print '\n', dataMat[i], labelMat[i]
            supportVector.append((dataMat[i], labelMat[i]))
    svCount, n = np.shape(supportVector)

    return supportVector, svCount


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # print("\n shape: " + str(np.shape(dataMat)))
    print np.mat(labelMat).transpose()

    # # the parameters of simpleSMO: C = 0.6, and tolerance = 0.001, the maxIterTimes = 40
    # alpha, b, m = simpleSMO(dataMat, labelMat, 0.6, 0.001, 40)
    # print("\n alpha: \n" + str(alpha))
    # print("\n b: " + str(b))
    #
    # supportVector, svCount = findSupportVector(alpha, m)
    # print("\n supportVector: \n" + str(np.mat(supportVector)))
    # # print("\n supportVector[1,0]: \n" + str(np.mat(supportVector)[1,0][0]))
    # # print("\n supportVector[1,0]: \n" + str(np.mat(supportVector)[1,0][1]))
    #
    # plotDataSet(dataMat, labelMat, alpha, b, supportVector, svCount)
