# -*- coding: utf-8 -*-
"""
Fonctions pour le calcul des descripteurs d'images
"""

import os
import cv2
import numpy as np
import time
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import (
    hog, 
    greycomatrix,
    greycoprops,
    local_binary_pattern
)
from PyQt5.QtWidgets import QMessageBox

def showDialog():
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText("Merci de sélectionner un descripteur via le menu ci-dessus")
    msgBox.setWindowTitle("Pas de Descripteur sélectionné")
    msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    returnValue = msgBox.exec()

def process_hierarchical_dataset(base_dir, descriptor_name, descriptor_function, progressBar):
    """
    Traite une base de données hiérarchique et calcule les descripteurs
    en créant un seul dossier contenant tous les descripteurs.
    
    Args:
        base_dir: Chemin vers le dossier racine (MIR_DATASETS_B)
        descriptor_name: Nom du descripteur (BGR, HSV, SIFT, etc.)
        descriptor_function: Fonction qui calcule le descripteur pour une image
        progressBar: Barre de progression de l'interface
    """
    print(f"Démarrage de l'indexation {descriptor_name}...")
    
    # Créer le dossier principal pour ce descripteur s'il n'existe pas
    if not os.path.isdir(descriptor_name):
        os.mkdir(descriptor_name)
    
    start_time = time.time()
    
    # Compter le nombre total d'images pour la barre de progression
    total_images = 0
    for animal_dir in os.listdir(base_dir):
        animal_path = os.path.join(base_dir, animal_dir)
        if os.path.isdir(animal_path):
            for breed_dir in os.listdir(animal_path):
                breed_path = os.path.join(animal_path, breed_dir)
                if os.path.isdir(breed_path):
                    total_images += len([f for f in os.listdir(breed_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Traiter chaque animal
    processed_images = 0
    for animal_dir in os.listdir(base_dir):
        animal_path = os.path.join(base_dir, animal_dir)
        if os.path.isdir(animal_path):
            # Traiter chaque race
            for breed_dir in os.listdir(animal_path):
                breed_path = os.path.join(animal_path, breed_dir)
                if os.path.isdir(breed_path):
                    # Traiter chaque image
                    for img_file in os.listdir(breed_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(breed_path, img_file)
                            try:
                                # Calculer le descripteur
                                feature = descriptor_function(img_path)
                                
                                # Sauvegarder le descripteur avec un nom qui inclut l'animal et la race
                                img_name, _ = os.path.splitext(img_file)
                                output_name = f"{animal_dir}_{breed_dir}_{img_name}.txt"
                                output_path = os.path.join(descriptor_name, output_name)
                                np.savetxt(output_path, feature)
                                
                                # Mettre à jour la barre de progression
                                processed_images += 1
                                progressBar.setValue(100 * (processed_images / total_images))
                                
                            except Exception as e:
                                print(f"Erreur lors du traitement de {img_path}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"Indexation {descriptor_name} terminée en {elapsed_time:.2f} secondes.")

# Fonctions de calcul de descripteurs pour chaque image individuelle
def compute_color_histogram(img_path):
    img = cv2.imread(img_path)
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])
    return np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)

def compute_hsv_histogram(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([img], [0], None, [256], [0, 256])
    histS = cv2.calcHist([img], [1], None, [256], [0, 256])
    histV = cv2.calcHist([img], [2], None, [256], [0, 256])
    return np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)

def compute_sift(img_path):
    img = cv2.imread(img_path)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)
    return des if des is not None else np.array([])

def compute_orb(img_path):
    img = cv2.imread(img_path)
    orb = cv2.ORB_create()
    key_point1, descrip1 = orb.detectAndCompute(img, None)
    return descrip1 if descrip1 is not None else np.array([])

def compute_glcm(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = img_as_ubyte(gray)
    
    distances = [1, -1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True)
    glcmProperties1 = greycoprops(glcmMatrix, 'contrast').ravel()
    glcmProperties2 = greycoprops(glcmMatrix, 'dissimilarity').ravel()
    glcmProperties3 = greycoprops(glcmMatrix, 'homogeneity').ravel()
    glcmProperties4 = greycoprops(glcmMatrix, 'energy').ravel()
    glcmProperties5 = greycoprops(glcmMatrix, 'correlation').ravel()
    glcmProperties6 = greycoprops(glcmMatrix, 'ASM').ravel()
    
    return np.array([
        glcmProperties1, glcmProperties2, glcmProperties3,
        glcmProperties4, glcmProperties5, glcmProperties6
    ]).ravel()

def compute_lbp(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (350, 350))
    
    points = 8
    radius = 1
    method = 'default'
    subSize = (70, 70)
    
    fullLBPmatrix = local_binary_pattern(img, points, radius, method)
    histograms = []
    
    for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
        for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
            subVector = fullLBPmatrix[
                k*subSize[0]:(k+1)*subSize[0],
                j*subSize[1]:(j+1)*subSize[1]
            ].ravel()
            subHist, edges = np.histogram(
                subVector,
                bins=int(2**points),
                range=(0, 2**points)
            )
            histograms = np.concatenate((histograms, subHist), axis=None)
    
    return histograms

def compute_hog(img_path):
    """Calcule le descripteur HOG pour une image"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))  # Redimensionner pour HOG
    
    # Calculer le HOG
    fd, hog_image = hog(
        img, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        feature_vector=True
    )
    
    return fd

# Fonctions principales appelées par l'interface
def generateHistogramme_Color(filenames, progressBar):
    process_hierarchical_dataset(filenames, "BGR", compute_color_histogram, progressBar)

def generateHistogramme_HSV(filenames, progressBar):
    process_hierarchical_dataset(filenames, "HSV", compute_hsv_histogram, progressBar)

def generateSIFT(filenames, progressBar):
    process_hierarchical_dataset(filenames, "SIFT", compute_sift, progressBar)

def generateORB(filenames, progressBar):
    process_hierarchical_dataset(filenames, "ORB", compute_orb, progressBar)

def generateGLCM(filenames, progressBar):
    process_hierarchical_dataset(filenames, "GLCM", compute_glcm, progressBar)

def generateLBP(filenames, progressBar):
    process_hierarchical_dataset(filenames, "LBP", compute_lbp, progressBar)

def generateHOG(filenames, progressBar):
    """Génère les descripteurs HOG pour toutes les images dans la base de données"""
    process_hierarchical_dataset(filenames, "HOG", compute_hog, progressBar) 

def extractReqFeatures(fileName, algo_choice):
    print(f"Extraction des caractéristiques avec l'algorithme {algo_choice}")
    
    if algo_choice == 1:  # BGR
        vect_features = compute_color_histogram(fileName)
        
    elif algo_choice == 2:  # HSV
        vect_features = compute_hsv_histogram(fileName)
        
    elif algo_choice == 3:  # SIFT
        vect_features = compute_sift(fileName)
        
    elif algo_choice == 4:  # ORB
        vect_features = compute_orb(fileName)
        
    elif algo_choice == 5:  # GLCM
        vect_features = compute_glcm(fileName)
        
    elif algo_choice == 6:  # LBP
        vect_features = compute_lbp(fileName)
    
    elif algo_choice == 7:  # HOG
        vect_features = compute_hog(fileName)
    
    else:
        raise ValueError(f"Algorithme non reconnu: {algo_choice}")

    # Sauvegarder les caractéristiques
    np.savetxt("Methode_"+str(algo_choice)+"_requete.txt", vect_features)
    print("Caractéristiques sauvegardées")
    return vect_features