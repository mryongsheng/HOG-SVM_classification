# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import sys
import os

DetectStride = 8
DetectScale = 1.2

#读取文件夹中图像
def loadImages(Dir,Nums):
    Amount = 0
    ImgList = []
    for root, dirs, files in os.walk(Dir):
        for file in files:
            ImgName = Dir+'\\'+file
            ImgList.append(cv2.imread(ImgName))
            cv2.imshow('ori',cv2.imread(ImgName))
            cv2.waitKey(10)
            Amount += 1
            if Amount>=Nums:
                return(Amount,ImgList)
    return(Amount,ImgList)

#从图像中随机截取部分作为负样本
def cutNegativeSamples(ImgList,SamNum,Width,Height,NegList):
    random.seed(1)
    for OriImg in ImgList:
        for Num in range(SamNum):
            y = int(random.random()*(len(OriImg)-Height))
            x = int(random.random()*(len(OriImg[0])-Width))
            NegList.append(OriImg[y:y+Height,x:x+Width])
            cv2.imshow('negi',OriImg[y:y+Height,x:x+Width])
            cv2.waitKey(10)
    return NegList

#计算HOG特征向量
def computeHOG(Img,Width,Height,HOGDis):
    ComImg = cv2.resize(Img,dsize=(Width,Height))
    return HOGDis.compute(ComImg)

#创建SVM分类器
def createSVM():
    SVM = cv2.ml.SVM_create()
    SVM.setCoef0(0)
    SVM.setDegree(3)
    Criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    SVM.setTermCriteria(Criteria)
    SVM.setGamma(0)
    SVM.setKernel(cv2.ml.SVM_LINEAR)
    SVM.setNu(0.5)
    SVM.setP(0.5)  # for EPSILON_SVR, epsilon in loss function?
    SVM.setC(10)  # From paper, soft classifier
    #注意样本label和SVM类型必须匹配，对EPS_SVR类型Label为+-1
    SVM.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    return SVM

#创建并训练SVM分类器
def trainSVM(SVM,Samples,Labels):
    SVM.train(np.array(Samples), cv2.ml.ROW_SAMPLE, np.array(Labels))
    return SVM

#强化训练
def hardTrain(SVMJudger,NegiImgs,HOGDis,Width,Height,Sams,Labels):
    SVMDetector = createSVMDetector(SVMJudger,HOGDis)
    HardNegi=[]
    #提取训练样本
    for FullNegi in NegiImgs:
        Img = FullNegi
        rects, wei = SVMDetector.detectMultiScale(FullNegi, winStride=(DetectStride, DetectStride),padding=(0, 0), scale=1.05)
        for (x,y,w,h) in rects:
            HardNegi.append(FullNegi[y:y+h, x:x+w])
            Img = cv2.rectangle(Img,(x,y),(x+w,y+h),(255,0,0),2,1)
        cv2.imshow('hard_train',Img)
        cv2.waitKey(1)
    for HardNegiImg in HardNegi:
        Sams.append(computeHOG(NegiImg,Width,Height,HOGDis))
        Labels.append(-1)
    SVM = trainSVM(SVMJudger,Sams,Labels)
    return SVM

#用SVM分类器创建SVM检测器
def createSVMDetector(NetSVM,HOGDis):
    SVMDetector = NetSVM.getSupportVectors()
    Rho, _, _ = NetSVM.getDecisionFunction(0)
    SVMDetector = np.transpose(SVMDetector)
    HOGDis.setSVMDetector(np.append(SVMDetector, [[-Rho]], 0))
    return HOGDis

#测试目标检测器
def testSVMDetector(Dir,SVMDetector):
    for root, dirs, files in os.walk(Dir):
        for file in files:
            ImgName = Dir+'\\'+file
            Img = cv2.imread(ImgName)
            Img = cv2.resize(Img,(0,0),fx=0.5,fy=0.5)
            rects, wei = SVMDetector.detectMultiScale(Img, winStride=(DetectStride, DetectStride),padding=(0, 0), scale=DetectScale)
            for (x,y,w,h) in rects:
                Img = cv2.rectangle(Img,(x,y),(x+w,y+h),(255,0,0),2,1)
            cv2.imshow('result',Img)
            cv2.waitKey(1)
    return None
#
#
# #开始流程
# Width = 64
# Height = 64
# SamPerImg = 10
# WinSize = (Width,Height)
# BlockSize = (16,16)
# BlockStride = (8,8)
# CellSize = (4,4)
# Nbins = 4
# HOGDis = cv2.HOGDescriptor(WinSize,BlockSize,BlockStride,CellSize,Nbins)
# #准备正负样本
# PosiNum,PosiImgs = loadImages(r'I:\SD\Programs\Python\HOG_SVM_Detector\Positive',99999)
# NegiNum,NegiImgs = loadImages(r'I:\SD\Programs\Python\HOG_SVM_Detector\Negitive',30)
# NegiList = []
# NegiList = cutNegativeSamples(NegiImgs,SamPerImg,Width,Height,NegiList)
# if PosiNum == 0:
#     print('No positive image found')
#     sys.exit()
# if NegiNum == 0:
#     print('No negitive image found')
#     sys.exit()
# print('完成图像样本生成')
# #准备正样本
# Sams = []
# Labels = []
# for PosiImg in  PosiImgs:
#     Sams.append(computeHOG(PosiImg,Width,Height,HOGDis))
#     Labels.append(1)
# for NegiImg in NegiList:
#     Sams.append(computeHOG(NegiImg,Width,Height,HOGDis))
#     Labels.append(-1)
# print('完成HOG数据生成')
# #第一轮训练
# SVMJudger = createSVM()
# trainSVM(SVMJudger,Sams,Labels)
# print('第一轮训练完成')
# #强化训练
# for ind in range(1,5):
#     SVMJudger = hardTrain(SVMJudger,NegiImgs,HOGDis,Width,Height,Sams,Labels)
#     print('训练完成')
# #保存训练结果
# SVMJudger.save(r'I:\SD\Programs\Python\HOG_SVM_Detector\SVMJudger.html')
# SVMDetector = createSVMDetector(SVMJudger,HOGDis)
# SVMDetector.save(r'I:\SD\Programs\Python\HOG_SVM_Detector\SVMDetector.bin')
# #测试
TestDet = cv2.HOGDescriptor()
TestDet.load('/home/fansne/Desktop/hog/HOG_SVM_Detector/SVMDetector.bin')
testSVMDetector('/home/fansne/Desktop/hog/HOG_SVM_Detector/Test',TestDet)
