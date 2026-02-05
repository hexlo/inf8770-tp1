import numpy as np
import math
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle

# Code pris de github.com/gabilodeau/INF8770/Codage Huffman.ipynb et modifié
def huffman(Message):

    if not isinstance(Message, np.ndarray):
        raise ValueError('Message doit être un tableau numpy')
    
    # On s'assure que l'array est 1D
    Message = Message.flatten()

    # On converti uint8 en int pour éviter un overflow
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

    with open("output/occurences.txt", "w") as f:
        f.write(str(OccSymb))



    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    # print(ArbreSymb)

    with open("output/occurences_triees.txt", "w") as f:
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

        with open("output/occurences_triees_fusion.txt", "w") as f:
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

    with open("output/arbre_codes.txt", "w") as f:
        f.write(str(RenderTree(ArbreCodes, style=AsciiStyle()).by_attr()))

    with open("output/arbre_symboles.txt", "w") as f:
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


    with open("output/dictionnaire.txt", "w") as f:
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

    with open("output/message_code.txt", "w") as f:
        f.write(str(MessageCode))
    # print(MessageCode)



    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))
    print("")
    print("Compression = {0}".format(longueurOriginale/longueur))
    print("")



    print('Espérance: ' + str(longueur/len(Message)))
    entropie =0
    for i in range(nbsymboles):
        if OccSymb[i][1] > 0:
            entropie = entropie-(OccSymb[i][1]/len(Message))*math.log(OccSymb[i][1]/len(Message),2)
        else:
            raise ValueError('Trying to do log(0)')

    print('Entropie: ' + str(entropie))