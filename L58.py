import os
import time
import random
import string
import gc
import timeit
import pickle
import numpy as np
import cirq
import tensorflow as tf
from subprocess import Popen
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import keras_tuner as kt
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import uuid 
from scipy.linalg import hadamard
# Configuration des chemins et paramètres
MINER_PATH = "C:/Users/marde/Documents/script/NBminer_Win"
MINER_EXECUTABLE = "nbminer.exe"
POOL_URL = "stratum+tcp://stratum.ravenminer.com:3838"
USER = "RAuXHPknR9prY2Guo9GHxqGyoxTLg6GAVh.77"
PASSWORD = "x"

INTERCEPT_CONSTANT = float(os.getenv('INTERCEPT_CONSTANT', 7))
BATCH_SIZE = 7  # Taille des lots pour écrire sur disque (ajustée pour des performances optimales)

MODEL_FILE_PATH = "model.keras"
CIRCUIT_FILE_PATH = "quantum_circuit.pkl"

# Fréquence initiale pour la simulation quantique
quantum_circuit_repetitions = 5000
initial_qubits = 3

# Initialiser les variables globales
process = None
X_data = None
y_data = None
batch_X = None
batch_y = None
rmse_data = []
intercept_data = []
r2_data = []
def define_data(size=100):
    """Fonction pour définir la variable Data."""
    # Générer un tableau d'entiers aléatoires
    Data = np.random.randint(1, 1000, size=size)
    
    # Créer un masque pour identifier les nombres premiers
    prime_mask = np.vectorize(is_prime)(Data)
    
    return Data, prime_mask
def appliquer_superposition(matrice, type_superposition='plus'):
    """
    Adapte une matrice pour représenter une superposition quantique.

    Args:
        matrice (np.ndarray): La matrice d'entrée à adapter.
        type_superposition (str): Type de superposition ('plus' ou 'moins').

    Returns:
        np.ndarray: La matrice adaptée à la superposition choisie.
    """
    # Définitions des vecteurs de superposition
    vecteur_plus = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=np.float64)
    vecteur_moins = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=np.float64)

    # Sélection du vecteur de superposition
    if type_superposition == 'plus':
        vecteur_superposition = vecteur_plus
    elif type_superposition == 'moins':
        vecteur_superposition = vecteur_moins
    else:
        raise ValueError("type_superposition doit être 'plus' ou 'moins'.")

    # Adapter la matrice avec le vecteur de superposition choisi
    matrice_superposition = np.outer(vecteur_superposition, vecteur_superposition.T)
    matrice_adaptee = matrice * matrice_superposition

    return matrice_adaptee
def est_premier(n):
    """Vérifie si un nombre est premier."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
def display_prime_matrices(X_data, y_data, data):
    # Vérifiez que X_data et y_data ne sont pas None
    if X_data is None or y_data is None:
        print("Erreur : X_data ou y_data est None.")
        return
    
    # Vérifiez que X_data et y_data sont des tableaux NumPy
    if not isinstance(X_data, np.ndarray) or not isinstance(y_data, np.ndarray):
        print("Erreur : X_data ou y_data n'est pas un tableau NumPy.")
        return
    
    # Assurez-vous que les dimensions sont compatibles pour la concaténation
    if X_data.ndim < 2 or y_data.ndim < 2:
        print("Erreur : Les données doivent avoir au moins 2 dimensions.")
        return

    # Combine X_data and y_data into a single array
    try:
        Data = np.concatenate([X_data.reshape(-1, X_data.shape[-1]), y_data.reshape(-1, y_data.shape[-1])], axis=1)
    except Exception as e:
        print(f"Erreur lors de la combinaison des données : {e}")
        return
def generateurs_premiers(n):
    """Génère les premiers n nombres premiers."""
    premiers = []
    compteur = 7
    while len(premiers) < n:
        if est_premier(compteur):
            premiers.append(compteur)
        compteur += 1
    return premiers
def creer_matrices_premieres(nombre_matrices, taille, valeur_initiale=0+0j, valeur_initiale_0 = 1 + 0j, valeur_initiale_1 = 0 + 1j, valeur_initiale_plus = (1/np.sqrt(2)) + (1/np.sqrt(2))*1j, valeur_initiale_moins = (1/np.sqrt(2)) - (1/np.sqrt(2))*1j):
    """
    Crée un certain nombre de matrices en utilisant des nombres premiers et des valeurs complexes.
matrice_0 = np.full(taille_matrice, valeur_initiale_0, dtype=np.complex128)
matrice_1 = np.full(taille_matrice, valeur_initiale_1, dtype=np.complex128)
matrice_plus = np.full(taille_matrice, valeur_initiale_plus, dtype=np.complex128)
matrice_moins = np.full(taille_matrice, valeur_initiale_moins, dtype=np.complex128)
    Args:
        nombre_matrices (int): Nombre de matrices à créer.
        taille (tuple): Taille de chaque matrice (par exemple, (1000, 1000) pour une matrice 1000x1000).
        valeur_initiale (complex): Valeur initiale (complexe) pour remplir les matrices.

    Returns:
        list: Liste contenant les matrices créées.
    """
    matrices = []
    premiers = generateurs_premiers(nombre_matrices)
    matrice_0 = np.full(taille_matrice, valeur_initiale_0, dtype=np.complex128)
    matrice_1 = np.full(taille_matrice, valeur_initiale_1, dtype=np.complex128)
    matrice_plus = np.full(taille_matrice, valeur_initiale_plus, dtype=np.complex128)
    matrice_moins = np.full(taille_matrice, valeur_initiale_moins, dtype=np.complex128)
    for i in range(nombre_matrices):
        matrice = np.full(taille, valeur_initiale * premiers[i], dtype=np.complex128)
        matrices.append(matrice)
    return matrices
def normalize_matrix(matrix):
    """
    Normalise une matrice pour garantir la stabilité des transformations.
    
    Args:
        matrix (numpy.ndarray): Matrice à normaliser.
        
    Returns:
        numpy.ndarray: Matrice normalisée.
    """
    norm = np.linalg.norm(matrix, 'fro')
    if norm == 0:
        return matrix
    return matrix / norm
def combiner_matrices(matrices, operation='addition'):
    """
    Combine les matrices en fonction de l'opération spécifiée.

    Args:
        matrices (list): Liste de matrices à combiner.
        operation (str): L'opération à appliquer (addition, multiplication, conjugaison).

    Returns:
        np.ndarray: La matrice résultante après combinaison.
    """
    if len(matrices) == 0:
        raise ValueError("La liste des matrices est vide.")
    
    resultat = matrices[0]
    
    if operation == 'addition':
        for matrice in matrices[1:]:
            resultat = np.add(resultat, matrice)
    
    elif operation == 'multiplication':
        for matrice in matrices[1:]:
            resultat = np.dot(resultat, matrice)
    
    elif operation == 'conjugaison':
        for i in range(len(matrices)):
            matrices[i] = np.conjugate(matrices[i])
        resultat = matrices[0]
        for matrice in matrices[1:]:
            resultat = np.add(resultat, matrice)
    
    else:
        raise ValueError("Opération non supportée. Choisissez 'addition', 'multiplication' ou 'conjugaison'.")
    
    return resultat
def configure_gpu_memory(log_file_path):
    """
    Configure la mémoire GPU pour utiliser la croissance dynamique de la mémoire.
    
    Args:
        log_file_path (str): Chemin du fichier journal pour les logs.
    """
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            log_data("GPU configuré pour utiliser la croissance de mémoire dynamique.", log_file_path)
        else:
            log_data("Aucun GPU trouvé.", log_file_path)
    except Exception as e:
        log_data(f"Erreur lors de la configuration du GPU: {e}", log_file_path)
def generate_dynamic_values(size, seed=None):
    """
    Génère des valeurs dynamiques pour les matrices en utilisant une fonction aléatoire ou déterministe.

    Args:
        size (int): Le nombre total de valeurs à générer.
        seed (int, optional): Une valeur pour initialiser le générateur aléatoire (optionnelle).

    Returns:
        numpy.ndarray: Un tableau de valeurs dynamiques.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Exemple de génération de valeurs dynamiques : valeurs aléatoires dans une plage donnée
    values = np.random.uniform(low=-1.0, high=1.0, size=size)
    
    return values
def trigonometric_adjustment(matrix, iteration, amplitude=7, frequency=0.01):
    """
    Ajuste les valeurs d'une matrice en utilisant des fonctions trigonométriques pour stabiliser et améliorer l'entraînement.

    Args:
        matrix (numpy.ndarray): La matrice à ajuster.
        iteration (int): Le nombre actuel d'itérations, utilisé pour la phase de la fonction trigonométrique.
        amplitude (float): Amplitude de l'ajustement trigonométrique.
        frequency (float): Fréquence de l'ajustement trigonométrique.

    Returns:
        numpy.ndarray: Matrice ajustée.
    """
    adjustment = amplitude * np.sin(frequency * iteration)
    return matrix + adjustment
    """
    Calcule les récompenses pour les mineurs en ajustant dynamiquement les matrices de poids à l'aide de trigonométrie.

    Args:
        shares (numpy.ndarray): Matrice des parts.
        weight_matrix (numpy.ndarray): Matrice des poids.
        num_iterations (int): Nombre d'itérations pour l'ajustement.
        initial_learning_rate (float): Taux d'apprentissage initial.

    Returns:
        numpy.ndarray: Récompenses pour chaque mineur.
    """
    # Assurez-vous que la taille de la matrice Hadamard correspond à la taille des données
    size = shares.shape[1]  # Utilisez le nombre de colonnes de 'shares' pour la taille
    hadamard_matrix = create_hadamard_matrix(size)
    
    shares = np.asarray(shares)
    weight_matrix = np.asarray(weight_matrix)

    for iteration in range(num_iterations):
        shares = apply_unitary(shares, hadamard_matrix)
        weighted_shares = np.dot(shares, weight_matrix)
        total_rewards = weighted_shares.sum(axis=1)
        
        shares = normalize_matrix(shares)
        weight_matrix = normalize_matrix(weight_matrix)

        gradients = weighted_shares - weight_matrix
        weight_matrix = update_weights_with_trigonometry(weight_matrix, gradients, iteration, initial_learning_rate)
    
    return total_rewards
def generate_matrices_for_training(size, num_iterations):
    """
    Génère des matrices aléatoires pour les données d'entraînement et calcule les récompenses en utilisant l'ajustement trigonométrique.

    Args:
        size (int): Taille des matrices.
        num_iterations (int): Nombre d'itérations pour le calcul.

    Returns:
        numpy.ndarray: Récompenses calculées.
    """
    shares = np.random.rand(size, size)
    weight_matrix = np.random.rand(size, size)
    return calculate_rewards_with_trigonometric_adjustment(shares, weight_matrix, num_iterations)
def configure_matrices_on_gpu(matrix):
    """
    Configure la matrice pour l'utiliser sur le GPU si disponible.
    
    Args:
        matrix (numpy.ndarray): Matrice d'entrée.
    
    Returns:
        cupy.ndarray or numpy.ndarray: Matrice configurée pour GPU ou CPU.
    """
    if gpu_available:
        return cp.asarray(matrix)
    else:
        return np.asarray(matrix)
def calculate_rewards(shares, weight_matrix):
    """
    Calcule les récompenses pour les mineurs basées sur leurs parts et la matrice de poids.
    Utilise un GPU si disponible, sinon bascule sur le CPU.

    Args:
        shares (numpy.ndarray or cupy.ndarray): Matrice des parts où chaque ligne est un mineur et chaque colonne est une part spécifique.
        weight_matrix (numpy.ndarray or cupy.ndarray): Matrice des poids.

    Returns:
        numpy.ndarray or cupy.ndarray: Récompenses pour chaque mineur, avec les matrices associées.
    """
    # Configurer les matrices pour l'utilisation GPU si possible
    shares = configure_matrices_on_gpu(shares)
    weight_matrix = configure_matrices_on_gpu(weight_matrix)
    
    # Calcul des parts pondérées
    weighted_shares = shares.dot(weight_matrix)
    
    # Calcul des récompenses totales
    total_rewards = weighted_shares.sum(axis=1)
    
    # Si sur GPU, convertir les résultats en numpy avant de les retourner
    if gpu_available:
        total_rewards = cp.asnumpy(total_rewards)
        shares = cp.asnumpy(shares)
        weighted_shares = cp.asnumpy(weighted_shares)
        weight_matrix = cp.asnumpy(weight_matrix)
    
    return total_rewards, shares, weighted_shares, weight_matrix
def simulate_quantum_circuit_optimized(num_qubits, depth):
    # Création du circuit quantique
    circuit = cirq.Circuit()
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    # Ajouter des portes Hadamard et CNOT
    for _ in range(depth):
        circuit.append(cirq.H.on_each(*qubits))
        for i in range(num_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    
    # Ajouter des mesures
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Afficher le circuit pour vérification
    print("Circuit quantique généré:")
    print(circuit)

    # Simulation du circuit quantique
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    
    return result
def generate_unique_filename():
    """Génère un nom de fichier unique basé sur l'horodatage et un identifiant unique."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())
    filename = f"log_{timestamp}_{unique_id}.txt"
    return os.path.join(os.getcwd(), filename)
def log_data(message, log_file_path):
    """Écrit un message dans le fichier de log spécifié."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + "\n")
def save_file(obj, file_path, log_file_path, mode='wb'):
    try:
        with open(file_path, mode) as file:
            pickle.dump(obj, file)
        log_data(f"Fichier sauvegardé à {file_path}", log_file_path)
    except Exception as e:
        log_data(f"Erreur lors de la sauvegarde du fichier: {e}", log_file_path)
def load_file(file_path, log_file_path, mode='rb'):
    try:
        if os.path.isfile(file_path):
            with open(file_path, mode) as file:
                obj = pickle.load(file)
            log_data(f"Fichier chargé depuis {file_path}", log_file_path)
            return obj, X_data, y_data, train_test_split, X_val, y_val
        else:
            log_data(f"Le fichier n'existe pas à {file_path}", log_file_path)
            return None
    except Exception as e:
        log_data(f"Erreur lors du chargement du fichier: {e}", log_file_path)
        return None, X_data, y_data
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, metrics, optimizer
def hyperparameter_tuning(X, y):
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, X.shape[10]),
        objective='val_loss',
        max_epochs=7,  # Réduit le nombre d'époques pour le tuning rapide
        directory='tuner',
        project_name='hyperparameter_tuning'
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    tuner.search(X_train, y_train, epochs=7, validation_data=(X_val, y_val))  # Réduit les époques pour le tuning rapide

    best_model = tuner.get_best_models(num_models=1)[0]
    best_params = tuner.get_best_hyperparameters(num_trials=1)[0].values

    return best_model, best_params
def prime_mask(data):
    # Analyse des données
    print("Analyse des données réelles :")
    print(f"Nombre total de valeurs : {data.size}")
    print(f"Nombre total de nombres premiers : {np.sum(prime_mask)}")
    print(f"Proportion de nombres premiers : {np.mean(prime_mask)}")
def apply_unitary(matrix, unitary):
    # Vérifier les dimensions des matrices
    matrix_shape = matrix.shape
    unitary_shape = unitary.shape
    print(f"Dimensions de la matrice : {matrix_shape}")
    print(f"Dimensions de l'unitaire : {unitary_shape}")

    # Assurez-vous que les dimensions sont compatibles
    if matrix_shape[1] != unitary_shape[0]:
        raise ValueError(f"Les dimensions des matrices ne sont pas compatibles pour la multiplication : "
                         f"{matrix_shape} et {unitary_shape}")

    # Appliquer la transformation unitaire
    return np.dot(matrix, unitary)
def adjust_matrix_for_unitary(matrix, unitary_matrix):
    """Ajuste une matrice pour qu'elle soit compatible avec la matrice unitaire."""
    rows, cols = matrix.shape
    unitary_size = unitary_matrix.shape[0]
    
    if rows != unitary_size:
        raise ValueError("La dimension des lignes de la matrice d'entrée ne correspond pas à celle de la matrice unitaire.")
    
    # Ajuste les colonnes
    if cols < unitary_size:
        # Ajoute des colonnes pour atteindre la taille nécessaire
        padding_cols = unitary_size - cols
        matrix = np.pad(matrix, ((0, 0), (0, padding_cols)), mode='constant', constant_values=0)
    elif cols > unitary_size:
        # Réduit les colonnes pour correspondre à la taille nécessaire
        matrix = matrix[:, :unitary_size]
    
    return matrix
def create_hadamard_matrix(size):
    """
    Crée une matrice Hadamard de taille spécifiée.

    Args:
        size (int): La taille de la matrice Hadamard. Doit être une puissance de 2.

    Returns:
        numpy.ndarray: La matrice Hadamard.
    """
    if (size & (size - 1)) != 0:
        # Trouver la plus grande puissance de 2 inférieure ou égale à 'size'
        power_of_2 = 1
        while power_of_2 < size:
            power_of_2 *= 2
        size = power_of_2

    if size < 2:
        raise ValueError("La taille minimale pour une matrice Hadamard est 2.")
    
    return hadamard(size)
def optimized_matrix_multiplication(unitary_matrix, matrix):
    """
    Function to perform optimized matrix multiplication.
    
    Parameters:
    unitary_matrix (np.ndarray): A unitary matrix.
    matrix (np.ndarray): Another matrix to be multiplied.
    
    Returns:
    np.ndarray: Result of the multiplication.
    """
    # Validate inputs
    if not isinstance(unitary_matrix, np.ndarray) or not isinstance(matrix, np.ndarray):
        raise ValueError("Both inputs must be numpy arrays.")
    
    if unitary_matrix.shape[1] != matrix.shape[0] or unitary_matrix.shape[0] != unitary_matrix.shape[1]:
        raise ValueError("Matrix dimensions are not aligned for multiplication or unitary_matrix is not square.")
    
    # Perform the optimized matrix multiplication
    result = unitary_matrix @ matrix @ unitary_matrix.T
    
    return result
def cross_val_score_with_cv(X, y, model=None, scoring=None, cv=5, pd=None):
    """
    Évalue un modèle en utilisant la validation croisée et une métrique de scoring personnalisée.
    
    Paramètres :
    - X : np.ndarray ou pd.DataFrame
        Matrice des caractéristiques d'entrée.
    - y : np.ndarray ou pd.Series
        Vecteur des valeurs cibles.
    - model : sklearn.base.BaseEstimator, optionnel (par défaut LinearRegression())
        Modèle à évaluer. Si aucun modèle n'est fourni, LinearRegression est utilisé.
    - scoring : fonction ou str, optionnel (par défaut mean_squared_error)
        Métrique de scoring à utiliser pour évaluer le modèle. Si aucune métrique n'est fournie, 
        l'erreur quadratique moyenne (MSE) est utilisée.
    - cv : int, optionnel (par défaut 5)
        Nombre de splits dans la validation croisée.
    
    Retourne :
    - float
        Erreur moyenne de validation croisée du modèle.
    """
    # Vérifier que X et y sont des tableaux numpy
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError("X doit être un tableau numpy ou un DataFrame pandas.")
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise TypeError("y doit être un tableau numpy ou une Série pandas.")
    
    # Utiliser un modèle par défaut si aucun modèle n'est fourni
    if model is None:
        model = LinearRegression()
    
    # Utiliser une métrique de scoring par défaut si aucune métrique n'est fournie
    if scoring is None:
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    # Vérifier que le modèle est un estimatrice sklearn
    if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
        raise TypeError("Le modèle doit être une instance d'un estimatrice sklearn.")
    
    # Exécuter la validation croisée
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return -np.mean(scores)  # Retourner la moyenne des scores (l'erreur quadratique moyenne)
    except Exception as e:
        raise RuntimeError(f"Une erreur est survenue lors de la validation croisée : {e}")
def calculate_rewards_with_trigonometric_adjustment(X_data, y_data, num_iterations, initial_learning_rate):
    # Exemple de données
    shares = np.random.rand(4, 2)  # Exemple de matrice
    hadamard_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Matrice Hadamard 2x2
    
    # Ajuster les dimensions
    num_rows, num_cols = shares.shape
    if num_cols % hadamard_matrix.shape[0] != 0:
        raise ValueError("Les dimensions des matrices ne sont pas compatibles pour la multiplication.")

    # Appliquer la transformation Hadamard par bloc
    transformed_shares = np.zeros_like(shares)
    block_size = hadamard_matrix.shape[0]
    for i in range(0, num_cols, block_size):
        end = i + block_size
        transformed_shares[:, i:end] = apply_unitary(shares[:, i:end], hadamard_matrix)
    
    # Calcul des récompenses
    rewards = np.random.rand(num_iterations)  # Remplacez par le calcul réel
    return rewards
def custom_objects():
    """
    Configure les objets personnalisés pour l'évaluation des performances et des calculs quantiques.
    
    Returns:
        dict: Dictionnaire d'objets personnalisés, incluant l'erreur quadratique moyenne et d'autres fonctions si nécessaire.
    """
    return {
        'mse': mean_squared_error,
        'quantum_rewards': lambda shares, weight_matrix: calculate_rewards_with_quantum(shares, weight_matrix)
    }
def apply_unitary(matrix, unitary):
    matrix_shape = matrix.shape
    unitary_shape = unitary.shape
    print(f"Dimensions de la matrice : {matrix_shape}")
    print(f"Dimensions de l'unitaire : {unitary_shape}")
    
    # Vérifiez les dimensions pour la multiplication
    if matrix_shape[1] != unitary_shape[0]:
        raise ValueError(f"Les dimensions des matrices ne sont pas compatibles pour la multiplication : "
                         f"{matrix_shape} et {unitary_shape}")
    
    # Appliquer la transformation unitaire
    return np.dot(matrix, unitary)
def perform_computations_and_save_results(output_file='results.csv', interval=3):
    """
    Génère des matrices, calcule la matrice finale, applique la discrimination gaussienne, 
    et enregistre les résultats dans un fichier CSV à intervalles réguliers.
    
    Args:
        output_file (str): Nom du fichier CSV de sortie.
        interval (int): Intervalle en secondes entre les itérations.
    """
    def matrix_to_string(matrix):
        """Convertit une matrice en une chaîne de caractères pour l'enregistrement."""
        return np.array2string(matrix, formatter={'all': lambda x: f"{x.real:.2f}+{x.imag:.2f}j"})
    
    while True:
        # Générer les matrices
        matrix_1 = np.array([
            [-0.26042861-0.19139968j,  0.57255816+0.33808403j, -0.06806726+0.13501693j, -0.65410259+0.05202425j],
            [ 0.0109332 -0.26345464j, -0.24181139+0.44573894j,  0.33055877+0.53709088j,  0.20631771+0.48271467j],
            [ 0.33179325+0.21515247j,  0.2475537 +0.44523057j, -0.51896598-0.29552076j,  0.27571349+0.38914447j],
            [ 0.36321908+0.73328697j,  0.14720238+0.13978109j,  0.27099991+0.38639668j, -0.12447468-0.22564916j]
        ])

        matrix_2 = np.array([
            [-0.30740029-0.26606278j,  0.72053982+0.37352234j,  0.16981591-0.33824394j, -0.02709281+0.17899345j],
            [-0.29878316-0.49480131j, -0.38324044-0.36303934j,  0.08986714-0.32515787j, -0.43794412+0.28571086j],
            [ 0.47005291-0.36476397j,  0.15718192+0.0203295j, -0.61247665+0.15811724j,  0.00047169+0.46983947j],
            [-0.37636653+0.07043928j,  0.13207999+0.14166178j, -0.37793211+0.44720069j, -0.6360875 -0.2616198j ]
        ])

        matrix_3 = np.array([
            [-0.3567544 +0.59435672j, -0.21265808-0.2120598j,  0.20810167+0.11015761j,  0.53926856+0.28813534j],
            [-0.00935938-0.65446943j, -0.40977868-0.07576778j,  0.44861097+0.22578212j,  0.36469645-0.11264979j],
            [ 0.09465945-0.09541468j, -0.0430492 +0.80668238j, -0.18388209+0.06583373j,  0.27860879+0.46214324j],
            [-0.2317807 +0.1387872j,  -0.23408207+0.16959879j, -0.36283266+0.72286609j, -0.11544083-0.41946174j]
        ])

        matrix_4 = np.array([
            [-0.32721066+0.55529143j, -0.40839288+0.03127304j,  0.56602308-0.17733346j, -0.00261867+0.25492297j],
            [-0.23853722+0.37637804j, -0.33651911-0.22507409j, -0.46701736+0.11431047j,  0.37260052-0.51723579j],
            [-0.39734793+0.41263777j,  0.73412955+0.32527934j, -0.04107806+0.11992573j, -0.03656744-0.0984128j ],
            [-0.14917065+0.1887246j,  -0.1432476 -0.05532701j, -0.61515654-0.15003824j, -0.57658345+0.43031695j]
        ])

        prime_matrix = np.array([
            [2, 3, 5, 7],
            [11, 13, 17, 19],
            [23, 29, 31, 37],
            [41, 43, 47, 53]
        ])

        # Calculer la matrice finale
        superposed_matrix = matrix_1 + matrix_2 + matrix_3 + matrix_4
        final_matrix = superposed_matrix + prime_matrix

        # Appliquer la discrimination gaussienne
        real_part = final_matrix.real
        mean = np.mean(real_part)
        std_dev = np.std(real_part)
        lower_bound = mean - 0.5 * std_dev
        upper_bound = mean + 0.5 * std_dev
        gaussian_results = (real_part > lower_bound) & (real_part < upper_bound)

        # Convertir les matrices en chaînes de caractères
        results = {
            'Matrix 1': matrix_to_string(matrix_1),
            'Matrix 2': matrix_to_string(matrix_2),
            'Matrix 3': matrix_to_string(matrix_3),
            'Matrix 4': matrix_to_string(matrix_4),
            'Prime Matrix': matrix_to_string(prime_matrix),
            'Final Matrix': matrix_to_string(final_matrix),
            'Gaussian Results': matrix_to_string(gaussian_results)
        }

        # Enregistrer les résultats dans un fichier CSV
        df = pd.DataFrame(results.items(), columns=['Matrix Name', 'Values'])
        df.to_csv(output_file, mode='a', header=False, index=False)
        
        # Attendre avant la prochaine itération
        time.sleep(interval)
def retry_on_failure(func, max_attempts=3, *args, **kwargs):
    attempts = 0
    while attempts < max_attempts:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempts += 1
            log_data(f"Tentative {attempts}/{max_attempts} échouée pour {func.__name__}: {e}", kwargs.get('log_file_path', ''))
            if attempts == max_attempts:
                log_data(f"Échec permanent de {func.__name__} après {max_attempts} tentatives.", kwargs.get('log_file_path', ''))
                raise e
            time.sleep(0.01)  # Réduire l'attente avant de réessayer
def create_dataset(X, y, batch_size=10):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
def train_tf_model(X, y, log_file_path, existing_model=None, intercept_adjustment=7, params=None):
    if params is None:
        params = {}

    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 1)

    print(f"epochs: {epochs}, batch_size: {batch_size}")

    if isinstance(existing_model, tuple):
        model = existing_model[0]
    else:
        model = existing_model

    if model is None:
        input_shape = X.shape[1]
        model = create_model(input_shape, intercept_adjustment)

    if not isinstance(model, tf.keras.Model):
        raise ValueError("L'objet passé comme modèle existant n'est pas un modèle Keras valide.")

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    predictions = model.predict(X)

    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()

    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Model trained with epochs={epochs}, batch_size={batch_size}\n")
        log_file.write(f"RMSE: {rmse}\n")
        log_file.write(f"R^2: {r2}\n")

    return model, rmse, r2
def clean_data(X, y):
    """Nettoie les données pour s'assurer qu'elles sont numériques."""
    try:
        X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = pd.Series(y).apply(pd.to_numeric, errors='coerce').fillna(0).values
        return X, y
    except Exception as e:
        raise ValueError(f"Erreur lors du nettoyage des données: {e}")
def fine_tune_model(model, X, y, log_file_path, intercept_adjustment=7):
    try:
        # Ajustement de l'intercept, si nécessaire
        if intercept_adjustment != 7.0:
            # Exemple d'ajustement de l'intercept
            model.add(Dense(64, use_bias=True))
            model.layers[-1].set_weights([model.layers[-1].get_weights()[0], np.array([intercept_adjustment])])
        
        model.fit(X, y, epochs=500)  # Ajustez les paramètres de formation selon vos besoins
        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        log_data(f"Modèle affiné avec RMSE: {rmse} et R^2: {r2}", log_file_path)
        return model, rmse, r2
    except Exception as e:
        log_data(f"Erreur lors de l'affinage du modèle: {e}", log_file_path)
        return None, None, None
def start_mining(log_file_path):
    global process
    miner_executable = os.path.join(MINER_PATH, MINER_EXECUTABLE)
    if not os.path.isfile(miner_executable):
        log_data(f"Erreur: Exécutable '{miner_executable}' non trouvé.", log_file_path)
        return
    command = [
        miner_executable,
        "-a", "kawpow",
        "-o", POOL_URL,
        "-u", USER,
        "-p", PASSWORD
    ]
    log_data(f"Exécution de la commande: {command}", log_file_path)
    try:
        process = Popen(command, cwd=MINER_PATH)
    except Exception as e:
        log_data(f"Erreur lors du démarrage du processus: {e}", log_file_path)
def stop_mining(log_file_path):
    global process
    if process:
        try:
            process.terminate()
            process.wait()
        except Exception as e:
            log_data(f"Erreur lors de l'arrêt du processus: {e}", log_file_path)
        finally:
            process = None
def create_model(input_shape, intercept_adjustment=7):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),  # Sortie unique pour la régression
        tf.keras.layers.Lambda(lambda x: x + intercept_adjustment, name='intercept_adjustment_layer')
    ])
    return model
def create_memmap_array(filename, shape, dtype, convert_value=None, replace_with=None):
    if not os.path.exists(filename):
        # Crée un fichier vide avec la forme et le type de données spécifiés
        memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    else:
        memmap_array = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)

    # Si convert_value est défini, remplacez cette valeur par replace_with
    if convert_value is not None and replace_with is not None:
        np.place(memmap_array, memmap_array == convert_value, [replace_with])

    return memmap_array
def collect_data_from_miner(log_file_path, X_data, y_data):
    try:
        with open(log_file_path, 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break
                # Process the data from the line and update X_data and y_data
                print(f"Data Line: {line.strip()}")
                # Vous pouvez ajouter des traitements ici
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{log_file_path}' n'existe pas.")
    except Exception as e:
        print(f"Erreur lors de la collecte des données du mineur : {e}")
def extract_min_hashrate(lines):
    """
    Extrait le taux de hachage le plus bas parmi les lignes fournies en utilisant des matrices NumPy.

    Args:
        lines (list): Liste de lignes contenant des informations sur le taux de hachage.

    Returns:
        float: Le taux de hachage le plus bas trouvé parmi les lignes. Retourne 0.0 si aucune ligne ne contient un taux de hachage valide.
    """
    # Convertir les lignes en une matrice de parties (tableau NumPy)
    matrix_lines = np.array([line.split() for line in lines])

    # Créer un tableau NumPy pour stocker les taux de hachage extraits
    hashrates = np.full(matrix_lines.shape[0], np.nan)  # Utiliser np.nan pour les valeurs non définies

    # Parcourir les lignes de la matrice pour extraire les taux de hachage
    for i, row in enumerate(matrix_lines):
        if "Hashrate:" in row:
            index = np.where(row == "Hashrate:")[0][0]
            if index + 1 < len(row) and 'H/s' in row[index + 1]:
                try:
                    # Extraire la valeur du hashrate et la convertir en float
                    hashrate_str = row[index + 1].replace('H/s', '')
                    hashrates[i] = float(hashrate_str)
                except ValueError:
                    # En cas d'erreur de conversion, laisser NaN
                    continue
    
    # Trouver le taux de hachage le plus bas parmi les valeurs valides
    min_hashrate = np.nanmin(hashrates) if not np.all(np.isnan(hashrates)) else 0.0
    
    return min_hashrate
def benchmark_extract_hashrate():
    for line in sample_lines:
        extract_hashrate(line)

         # Measure execution time
        execution_time = timeit.timeit(benchmark_extract_hashrate, number=9000000000)
        print(f"Temps d'exécution pour 900000000 itérations: {execution_time:.6f} secondes")
        intercept, gradient = calculate_gradient_and_intercept(X, y)
# Assurez-vous que les prédictions faites avec ces coefficients sont bien 1D
        predictions = X.dot(gradient) + intercept
        rmse = np.sqrt(mean_squared_error(y, predictions))
def calculate_gradient_and_intercept(X, y):
    if X.size == 0 or y.size == 0:
        return None, None
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None, None
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, None
    model = LinearRegression().fit(X, y)
    intercept = model.intercept_
    gradient = model.coef_
    return model.intercept_, model.coef_
def simulate_quantum_circuit(log_file_path, repetitions, qubits_count):
    try:
        qubits = cirq.LineQubit.range(qubits_count)
        circuit = cirq.Circuit([
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key='result')
        ])

        save_file(circuit, CIRCUIT_FILE_PATH, log_file_path, mode='wb')

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=repetitions)

        results_matrix = result.measurements['result'].reshape((repetitions, qubits_count))

        log_data(f"Résultats de la simulation quantique (répétitions={repetitions}, qubits={qubits_count}): {result}", log_file_path)
        log_data(f"Matrice des résultats: {results_matrix}", log_file_path)

    except Exception as e:
        log_data(f"Erreur lors de la simulation quantique: {e}", log_file_path)
def simulate_quantum_circuit_for_plot(repetitions, qubits_count):
    qubits = cirq.LineQubit.range(qubits_count)
    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key='result')
    ])

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=repetitions)
    return result.measurements['result'].flatten()
def apply_gaussian_discrimination(data):
    # Convertir 'data' en tableau NumPy si ce n'est pas déjà le cas
    data = np.array(data)
    
    # Vérifiez les dimensions de 'data' et ajuster si nécessaire
    if data.ndim == 1:
        # Convertir en tableau 2D avec une seule colonne
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise TypeError(f"L'entrée 'data' doit être un tableau NumPy 1D ou 2D. Recevu un tableau {data.ndim}D.")
    
    # Calculer la moyenne et l'écart type
    mean = np.mean(data)
    std_dev = np.std(data)
    
    # Appliquer la discrimination gaussienne
    lower_bound = mean - 0.5 * std_dev
    upper_bound = mean + 0.5 * std_dev
    
    # Appliquer les conditions de discrimination
    mask = np.logical_and(data > lower_bound, data < upper_bound)
    
    # Convertir le masque en 1 et 0
    result = mask.astype(int)
    
    # Si l'entrée était 1D, convertir la sortie pour correspondre à une dimension 1D
    if data.ndim == 1:
        result = result.ravel()
    
    return result
def update_graph(frame, log_file_path, fig, ax1, ax2, ax3, ax4, ax5, ax6):
    global X_data, y_data, rmse_data, intercept_data, r2_data, current_model, quantum_circuit_repetitions, initial_qubits

    # Collecte des données depuis le mineur
    collect_data_from_miner(log_file_path, X_data, y_data)

    # Calcul du nombre de qubits à utiliser pour cette itération
    qubits_count = initial_qubits + (frame // 10)
    
    # Simulation quantique
    simulate_quantum_circuit(log_file_path, quantum_circuit_repetitions, qubits_count)
    
    # Incrémentation des répétitions pour la simulation quantique
    quantum_circuit_repetitions += 10

    # Vérification de la validité des dimensions des données
    if X_data.shape[0] < 2 or y_data.shape[0] < 2 or X_data.size == 0 or y_data.size == 0 or X_data.shape[0] != y_data.shape[0]:
        log_data("Erreur: Dimensions des données inconsistantes ou échantillons insuffisants.", log_file_path)
        return

    # Entraînement du modèle si les données sont valides
    if X_data.shape[0] >= 1:
        model, rmse, r2 = retry_on_failure(train_tf_model, 3, X_data, y_data, log_file_path, existing_model=current_model)
        if model:
            current_model = model
            save_file(model, MODEL_FILE_PATH, log_file_path, mode='wb')
            rmse_data.append(rmse)
            r2_data.append(r2)
            intercept, gradient = calculate_gradient_and_intercept(X_data, y_data)
            if intercept is not None:
                intercept_data.append(intercept)

            # Mise à jour des graphiques
            ax1.clear()
            ax1.plot(rmse_data, label='RMSE')
            ax1.set_title('Erreur Quadratique Moyenne (RMSE)')
            ax1.set_xlabel('Itération')
            ax1.set_ylabel('RMSE')
            ax1.legend()

            ax2.clear()
            ax2.plot(intercept_data, label='Intercept')
            ax2.set_title('Intercept du Modèle')
            ax2.set_xlabel('Itération')
            ax2.set_ylabel('Intercept')
            ax2.legend()

            ax3.clear()
            ax3.plot(np.array(intercept_data) + INTERCEPT_CONSTANT, label='Intercept Ajusté')
            ax3.set_title('Intercept Ajusté')
            ax3.set_xlabel('Itération')
            ax3.set_ylabel('Intercept Ajusté')
            ax3.legend()

            # Simulation quantique et discrimination gaussienne
            qubit_results = simulate_quantum_circuit_for_plot(quantum_circuit_repetitions, qubits_count)
            gaussian_results = apply_gaussian_discrimination(qubit_results)
            
            # Création de l'histogramme équilibré de 0.0 à 1.0
            ax4.clear()
            ax4.hist(gaussian_results, bins=np.linspace(0.0, 1.0, 11), range=(0.0, 1.0), label='Résultats Discriminés de la Simulation Quantique')
            ax4.set_title('Histogramme des Résultats Discriminés de la Simulation Quantique')
            ax4.set_xlabel('Résultat')
            ax4.set_ylabel('Fréquence')
            ax4.legend()

            if X_data.shape[0] > 0:
                samples_idx = np.arange(len(y_data))
                ax5.clear()
                ax5.plot(samples_idx, y_data, 'b.', label='Valeur Réelle')
                ax5.set_title('Échantillons a/b vs Valeur Réelle')
                ax5.set_xlabel('Index d\'échantillon')
                ax5.set_ylabel('Valeur Réelle')
                ax5.legend()

            ax6.clear()
            ax6.plot(r2_data, label='R^2')
            ax6.set_title('Coefficient de Détermination (R^2)')
            ax6.set_xlabel('Itération')
            ax6.set_ylabel('R^2')
            ax6.legend()

            # Optimisation de l'affichage des graphiques
            plt.tight_layout()
            gc.collect()
def combined_data(is_prime, prime_mask, data):
    """Main function to generate and display combined data and prime mask."""
    # Generating some random data for demonstration
    data = np.random.randint(0, 100, size=(10, 10))

    # Create a binary matrix indicating prime numbers
    prime_mask = filter_primes(data, is_prime)

    # Plot the matrices
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title('Données combinées')
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Valeur')
    plt.xlabel('Colonnes')
    plt.ylabel('Lignes')

    plt.subplot(1, 2, 2)
    plt.title('Matrice des nombres premiers')
    plt.imshow(prime_mask, cmap='binary', aspect='auto')
    plt.colorbar(label='Nombre Premier')
    plt.xlabel('Colonnes')
    plt.ylabel('Lignes')

    plt.tight_layout()
    plt.show()
def main():
    global X_data, y_data, batch_X, batch_y, rmse_data, intercept_data, r2_data, current_model, log_file_path
    global process, quantum_circuit_repetitions, initial_qubits

    # Génération d'un chemin de fichier de log unique
    log_file_path = generate_unique_filename()
    print(f"Chemin du fichier de log généré : {log_file_path}")

    # Log de début d'exécution
    log_data(f"Début de l'exécution du script à {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file_path)

    # Configuration de la mémoire GPU
    configure_gpu_memory(log_file_path)

    # Initialisation des données et du modèle
    num_samples = 10
    num_features = 3
    BATCH_SIZE = 5
    
    X_data = create_memmap_array("X_data.dat", (num_samples, num_features), dtype='float32')
    y_data = create_memmap_array("y_data.dat", (num_samples,), dtype='float32')
    
    # Remplir les données avec des valeurs dynamiques
    X_data[:] = generate_dynamic_values(num_samples * num_features).reshape((num_samples, num_features))
    y_data[:] = generate_dynamic_values(num_samples)
    
    batch_X = np.zeros((BATCH_SIZE, num_features), dtype='float32')
    batch_y = np.zeros(BATCH_SIZE, dtype='float32')
    rmse_data = []
    intercept_data = []
    r2_data = []
    
    # Charger le modèle existant
    current_model = load_file("model_file.npy", log_file_path, mode='rb')

    # Démarrer le minage
    start_mining(log_file_path)

    # Boucle d'entraînement avec ajustement trigonométrique
    num_iterations = 20
    initial_learning_rate = 0.01

    # Utiliser les matrices dynamiques pour l'entraînement
    rewards = calculate_rewards_with_trigonometric_adjustment(X_data, y_data, num_iterations, initial_learning_rate)
    print("Récompenses calculées avec ajustement trigonométrique :", rewards)

    intercept_value = np.random.rand() * 10  # Valeur d'interception dynamique
    model, rmse, r2 = train_tf_model(X_data, y_data, log_file_path, existing_model=current_model, intercept_adjustment=intercept_value)

    # Configuration des graphiques
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(10, 10))
    ani = FuncAnimation(fig, update_graph, fargs=(log_file_path, fig, ax1, ax2, ax3, ax4, ax5, ax6), interval=1000, cache_frame_data=False) 

    # Affichage des graphiques
    try:
        plt.show()
    except KeyboardInterrupt:
        log_data("Interruption de l'utilisateur détectée.", log_file_path)
    finally:
        # Arrêter le minage
        stop_mining(log_file_path)
        log_data(f"Fin de l'exécution du script à {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file_path)
if __name__ == "__main__":
    num_qubits = 3
    depth = 5
    results = simulate_quantum_circuit_optimized(num_qubits, depth)
    print("Quantum Circuit Simulation Results:", results)
    main()
# Copyright 2024  Tomy Verreault
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
