import numpy as np
import math
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
from PIL import Image
import pickle
import os

# Définir le répertoire de sortie relatif à ce fichier
# Le script est dans src/, donc output est dans le répertoire parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "output")

# Créer le répertoire output s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Code pris de github.com/gabilodeau/INF8770/Codage Huffman.ipynb et modifié
def huffman(image_path):
    """
    Applique le codage de Huffman à une image PNG.
    
    Args:
        image_path: Chemin vers l'image PNG à compresser
    
    Returns:
        dict: Métriques de compression incluant taille originale, compressée et ratio de compression
    """
    
    img = Image.open(image_path)
    
    # Sauvegarder les métadonnées pour la reconstruction
    image_metadata = {
        'size': img.size,
        'mode': img.mode
    }
    
    # Convertir l'image en tableau numpy et obtenir les bytes originaux
    Message = np.array(img)
    original_image_bytes = img.tobytes()
    taille_originale = len(original_image_bytes)
    
    # On s'assure que l'array est 1D
    Message = Message.flatten()

    # On converti uint8 en int pour éviter un overflow lors d'une addition
    if np.issubdtype(Message.dtype, np.integer):
        Message = Message.astype(int)

    #Liste qui sera modifié jusqu'à ce qu'elle contienne seulement la racine de l'arbre
    # On utilise np.unique pour optimiser le dénombrement (O(N) au lieu de O(N^2))
    symbols_uniques, counts = np.unique(Message, return_counts=True)
    
    ArbreSymb = []
    #dictionnaire obtenu à partir de l'arbre.
    dictionnaire = []
    
    for symbol, count in zip(symbols_uniques, counts):
        ArbreSymb.append([symbol, count, Node(symbol)])
        dictionnaire.append([symbol, ''])
        
    nbsymboles = len(symbols_uniques)
    print("Nombre de symboles différents: {0}", nbsymboles)

    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

    OccSymb = ArbreSymb.copy()

    with open(os.path.join(OUTPUT_DIR, "occurences.txt"), "w") as f:
        f.write(str(OccSymb))



    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    # print(ArbreSymb)

    with open(os.path.join(OUTPUT_DIR, "occurences_triees.txt"), "w") as f:
        f.write(str(ArbreSymb))


    while len(ArbreSymb) > 1:
        #Fusion des noeuds de poids plus faibles
        symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0]
        #Création d'un nouveau noeud
        noeud = Node(symbfusionnes)
        temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]
        #Ajustement de l'arbre pour connecter le nouveau avec ses parents
        ArbreSymb[0][2].parent = noeud
        ArbreSymb[1][2].parent = noeud
        #Enlève les noeuds fusionnés de la liste de noeud à fusionner.
        del ArbreSymb[0:2]
        #Ajout du nouveau noeud à la liste et tri.
        ArbreSymb += [temp]

        #Pour affichage de l'arbre ou des sous-branches
        # print('\nArbre actuel:\n\n')
        # for i in range(len(ArbreSymb)):
            # if np.count_nonzero(ArbreSymb[i][0]) > 1:
                # print(RenderTree(ArbreSymb[i][2], style=AsciiStyle()).by_attr())

        ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])

        with open(os.path.join(OUTPUT_DIR, "occurences_triees_fusion.txt"), "w") as f:
            f.write(str(ArbreSymb))
        # print(ArbreSymb)



    ArbreCodes = Node('')
    noeud = ArbreCodes
    #print([node.name for node in PreOrderIter(ArbreSymb[0][2])])
    parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
    parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

    Prevdepth = 0 #pour suivre les mouvements en profondeur dans l'arbre
    for node in parcoursprefix:  #Liste des noeuds
        if Prevdepth < node.depth: #On va plus profond dans l'arbre, on met un 0
            temp = Node(noeud.name + '0')
            noeud.children = [temp]
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        elif Prevdepth == node.depth: #Même profondeur, autre feuille, on met un 1
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]  #Ajoute le deuxième enfant
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        else:
            for i in range(Prevdepth-node.depth): #On prend une autre branche, donc on met un 1
                noeud = noeud.parent #On remontre dans l'arbre pour prendre la prochaine branche non explorée.
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]
            if node.children:
                noeud = temp

        Prevdepth = node.depth

    with open(os.path.join(OUTPUT_DIR, "arbre_codes.txt"), "w") as f:
        f.write(str(RenderTree(ArbreCodes, style=AsciiStyle()).by_attr()))

    with open(os.path.join(OUTPUT_DIR, "arbre_symboles.txt"), "w") as f:
        f.write(str(RenderTree(ArbreSymb[0][2], style=AsciiStyle()).by_attr()))
    # print('\nArbre des codes:\n\n',RenderTree(ArbreCodes, style=AsciiStyle()).by_attr())
    # print('\nArbre des symboles:\n\n', RenderTree(ArbreSymb[0][2], style=AsciiStyle()).by_attr())



    ArbreSymbList = [node for node in PreOrderIter(ArbreSymb[0][2])]
    ArbreCodeList = [node for node in PreOrderIter(ArbreCodes)]

    for i in range(len(ArbreSymbList)):
        if ArbreSymbList[i].is_leaf: #Génère des codes pour les feuilles seulement
            temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
            if temp:
                indice = dictionnaire.index(temp[0])
                dictionnaire[indice][1] = ArbreCodeList[i].name


    with open(os.path.join(OUTPUT_DIR, "dictionnaire.txt"), "w") as f:
        f.write(str(dictionnaire))
    # print(dictionnaire)


    # On converti de list à dict pour O(1) un lookup
    dict_lookup = {entry[0]: entry[1] for entry in dictionnaire}

    MessageCode = []
    longueur = 0
    for symbol in Message:
        code = dict_lookup[symbol]
        MessageCode.append(code)
        longueur += len(code)

    with open(os.path.join(OUTPUT_DIR, "message_code.txt"), "w") as f:
        f.write(str(MessageCode))
    # print(MessageCode)

    # Calculer la taille compressée réelle en incluant l'arbre et les métadonnées
    compressed_data = {
        'arbre': ArbreSymb[0][2],
        'dictionnaire': dictionnaire,
        'message_code': MessageCode,
        'metadata': image_metadata if image_metadata else None
    }
    
    # CORRECTION: Ne pas utiliser pickle pour mesurer la vraie compression
    # Le pickle ajoute un overhead énorme pour les listes de strings
    # On calcule plutôt: bits encodés / 8 + overhead du dictionnaire
    
    # Taille du message encodé en bytes (arrondi au byte supérieur)
    message_encoded_bytes = (longueur + 7) // 8
    
    # Pour la décompression, on n'a besoin QUE du dictionnaire et des métadonnées
    # L'arbre n'est pas nécessaire pour décoder, seul le dictionnaire l'est
    decompression_data = {
        'dictionnaire': dictionnaire,
        'metadata': image_metadata
    }
    overhead_bytes = len(pickle.dumps(decompression_data))
    
    # Taille compressée totale = message encodé + overhead
    taille_compressee = message_encoded_bytes + overhead_bytes
    
    # Calculer les ratios de compression (raw et total)
    ratio_raw = longueurOriginale / longueur  # Au niveau des bits
    ratio_compression = taille_originale / taille_compressee if taille_compressee > 0 else 0
    pourcentage_reduction = (1 - taille_compressee / taille_originale) * 100 if taille_originale > 0 else 0

    print("")
    print("="*60)
    print("MÉTRIQUES DE COMPRESSION HUFFMAN")
    print("="*60)
    print(f"Taille originale:           {taille_originale:,} octets")
    print("")
    print("COMPRESSION (bits encodés):")
    print(f"  Bits originaux:           {int(longueurOriginale):,} bits")
    print(f"  Bits encodés:             {longueur:,} bits")
    print(f"  Octets encodés:           {message_encoded_bytes:,} octets")
    print(f"  Ratio (raw):              {ratio_raw:.2f}x")
    print(f"  Gain (raw):               {(1 - longueur/longueurOriginale)*100:.2f}%")
    print("")
    print("OVERHEAD (dictionnaire + métadonnées):")
    print(f"  Taille overhead:          {overhead_bytes:,} octets")
    print(f"  Overhead % original:      {overhead_bytes/taille_originale*100:.2f}%")
    print("")
    print("TOTAL (encodé + overhead):")
    print(f"  Taille compressée:        {taille_compressee:,} octets")
    print(f"  Ratio de compression:     {ratio_compression:.2f}x")
    print(f"  Réduction:                {pourcentage_reduction:.2f}%")
    print("="*60)
    print("")

    print('Espérance: ' + str(longueur/len(Message)))
    entropie = 0
    for i in range(nbsymboles):
        if OccSymb[i][1] > 0:
            entropie = entropie-(OccSymb[i][1]/len(Message))*math.log(OccSymb[i][1]/len(Message),2)
        else:
            raise ValueError('Trying to do log(0)')

    print('Entropie: ' + str(entropie))
    print("")
    
    # Sauvegarder les métriques dans un fichier
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    metrics_filepath = os.path.join(OUTPUT_DIR, f"{base_filename}_metrics.txt")
    
    with open(metrics_filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MÉTRIQUES DE COMPRESSION HUFFMAN\n")
        f.write("="*60 + "\n")
        f.write(f"Image: {os.path.basename(image_path)}\n\n")
        f.write(f"Taille originale:           {taille_originale:,} octets\n\n")
        f.write("COMPRESSION (bits encodés):\n")
        f.write(f"  Bits originaux:           {int(longueurOriginale):,} bits\n")
        f.write(f"  Bits encodés:             {longueur:,} bits\n")
        f.write(f"  Octets encodés:           {message_encoded_bytes:,} octets\n")
        f.write(f"  Ratio (raw):              {ratio_raw:.2f}x\n")
        f.write(f"  Gain (raw):               {(1 - longueur/longueurOriginale)*100:.2f}%\n\n")
        f.write("OVERHEAD (dictionnaire + métadonnées):\n")
        f.write(f"  Taille overhead:          {overhead_bytes:,} octets\n")
        f.write(f"  Overhead % original:      {overhead_bytes/taille_originale*100:.2f}%\n\n")
        f.write("TOTAL (encodé + overhead):\n")
        f.write(f"  Taille compressée:        {taille_compressee:,} octets\n")
        f.write(f"  Ratio de compression:     {ratio_compression:.2f}x\n")
        f.write(f"  Réduction:                {pourcentage_reduction:.2f}%\n\n")
        f.write(f"Espérance:                  {longueur/len(Message):.4f}\n")
        f.write(f"Entropie:                   {entropie:.4f}\n")
        f.write(f"Nombre de symboles:         {nbsymboles}\n")
        f.write("="*60 + "\n")
    
    print("="*50)
    print(f"Métriques sauvegardées: {metrics_filepath}")
    print("="*50)
    print("")
    
    # Retourner les métriques
    return {
        'taille_originale': taille_originale,
        'taille_compressee': taille_compressee,
        'ratio_compression': ratio_compression,
        'pourcentage_reduction': pourcentage_reduction,
        'longueur_bits': longueur,
        'entropie': entropie,
        'fichier_metrics': metrics_filepath
    }