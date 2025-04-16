import numpy as np
import math
import cv2
#from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from skimage import feature
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
import operator
import collections 
from collections import Counter

def euclidean(l1, l2):
    # Convertir en numpy arrays et s'assurer qu'ils sont en 1D pour les histogrammes
    l1 = np.array(l1).ravel()
    l2 = np.array(l2).ravel()
    
    # Pour ORB et SIFT, on doit gérer des descripteurs de tailles différentes
    if len(l1.shape) == 2 and len(l2.shape) == 2:
        # Calculer la distance minimale entre les descripteurs
        distances = []
        for desc1 in l1:
            min_dist = float('inf')
            for desc2 in l2:
                dist = np.sqrt(np.sum((desc1 - desc2) ** 2))
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return np.mean(distances)
    else:
        # Pour les histogrammes (BGR, HSV, GLCM, LBP), utiliser la distance euclidienne classique
        return np.sqrt(np.sum((l1 - l2) ** 2))

def chiSquareDistance(l1, l2):
    l1 = np.array(l1).ravel()
    l2 = np.array(l2).ravel()
    s = 0.0
    for i, j in zip(l1, l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j + 1e-10)  # Ajout d'une petite valeur pour éviter la division par zéro
    return s

def bhatta(l1, l2):
    l1 = np.array(l1).ravel()
    l2 = np.array(l2).ravel()
    
    # Normaliser les vecteurs si ce n'est pas déjà fait
    l1 = l1 / (np.sum(l1) + 1e-10)
    l2 = l2 / (np.sum(l2) + 1e-10)
    
    num = np.sum(np.sqrt(np.multiply(l1, l2, dtype=np.float64)), dtype=np.float64)
    den = np.sqrt(np.sum(l1, dtype=np.float64) * np.sum(l2, dtype=np.float64))
    
    if den == 0:
        return 1.0
    
    return math.sqrt(1 - num / (den + 1e-10))


def flann(a,b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)

def distance_f(l1, l2, distanceName):
    # Convertir en numpy arrays
    l1 = np.array(l1)
    l2 = np.array(l2)
    
    if distanceName == "Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName in ["Correlation", "Chi carre", "Intersection", "Bhattacharyya"]:
        # Pour les histogrammes et descripteurs 1D (BGR, HSV, GLCM, LBP)
        if len(l1.shape) == 1 and len(l2.shape) == 1:
            if distanceName == "Correlation":
                # Normaliser les vecteurs
                l1_norm = (l1 - np.mean(l1)) / (np.std(l1) + 1e-10)
                l2_norm = (l2 - np.mean(l2)) / (np.std(l2) + 1e-10)
                distance = -np.mean(l1_norm * l2_norm)  # Négatif car on veut minimiser
            elif distanceName == "Chi carre":
                distance = chiSquareDistance(l1, l2)
            elif distanceName == "Intersection":
                # Normaliser les vecteurs
                l1_norm = l1 / (np.sum(l1) + 1e-10)
                l2_norm = l2 / (np.sum(l2) + 1e-10)
                distance = -np.sum(np.minimum(l1_norm, l2_norm))  # Négatif car on veut minimiser
            elif distanceName == "Bhattacharyya":
                distance = bhatta(l1, l2)
        else:
            # Pour ORB et SIFT, utiliser la distance euclidienne
            distance = euclidean(l1, l2)
    elif distanceName == "Brute force":
        distance = bruteForceMatching(l1, l2)
    elif distanceName == "Flann":
        distance = flann(l1, l2)
    return distance

def getkVoisins(lfeatures, req, k, distanceName): 
    ldistances = [] 
    for i in range(len(lfeatures)): 
        dist = distance_f(req, lfeatures[i][1], distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist)) 
    if distanceName in ["Correlation", "Intersection"]:
        ordre = True
    else:
        ordre = False
    ldistances.sort(key=operator.itemgetter(2), reverse=ordre) 

    lvoisins = [] 
    for i in range(min(k, len(ldistances))): 
        lvoisins.append(ldistances[i]) 
    return lvoisins

def getkVoisins_deep(features_dict, query_name, k):
    query_feature = features_dict[query_name]
    distances = []
    for name, feature_vector in features_dict.items():
        dist = euclidean(query_feature, feature_vector)  # Calcul de la distance
        distances.append((name, dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]