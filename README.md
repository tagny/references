# references
Notes de lectures d'articles et de rapports scientifiques

## Lus

### aria

#### retrofitting
* 2015 Retrofitting Word Vectors to Semantic Lexicons.pdf
* 2016 Counter-fitting Word Vectors to Linguistic Constraints.pdf

#### inter-annotator agreement
* Python Inter-Rater.pdf

#### intent detection
* 2019_Review_of_Intent_Detection_Methods_in_the_Human-Ma

## En cours de lecture

### aria

#### less annotations
* Active Learning and Why All Data Is Not Created Equal
* Data labeling in 2020

#### dataless intent recognition
**Problème** : absence de données annotées pour entrainer des algo de NLP

**Solutions**:
* **dataless text classification** : employer des connaissances (KB) à la place des données
  * exploite la similarité (lexicale, syntaxique ou sémantique) entre le texte à classifier et les catégories (pareil qu'en RI)
  * bon si données non annotées indisponibles
* **weak supervision** : générer automatiquement des données (par exemple avec un système de règles REGEX)
  * bon si textes brutes dispo et expertise de mots-clés dispo

###### 2019 - slide - NLP from scratch - Solving the cold start problem for NLP

###### 2019_ESWC_KB_Short_Text_Classification_Using_Entity_and_Category_Embedding**** en cours
Ce travail propose d'adresser l'absence de données annotées. La technique consiste à utiliser la similarité "sémantique" entre catégories et textes en s'appuyant sur des techniques de **graph embedding**.
**Définitions**
* **entité** : un lien hypertexte de Wikipedia indexé par un texte de lien dans le dico préfabriqué **Anchor-Text Dictionary** (e)
* **mention** : un terme dans le texte t qui peut se reférer à une entité e (m_e)
* **contexte** d'une entité e : ensemble d'autres mentions dans t exceptée celle de e (C_e)
* 

**Principe (application)**:
1. Détection des mentions d'**entités** du texte en entrée : n-grammes qui matchent une entrée d'un dico (**Anchor-Text Dictionary**)
2. Génération, pour chaque mention, d'un ensemble d'**entités** candidates à l'aide du dico préfabriqué (**Anchor-Text Dictionary**)  
  * PETITE IDEE: *indexer le dico comme la librairie [SML](http://www.semantic-measures-library.org) [Harispe et al. , 2013] indexe les mots dans des fichiers multiples: un chunkfile pour les n-grammes débutant par les mêmes 2 premiers chars particulier, indexer les noms de fichiers de  débuts de n-grammes dans un fichier chunk_index)*
3. Application la méthode probabiliste proposée (similaire à un classifieur bayésien P(c|t) ~= P(c,t) = P(c)P(t|c) pour trouver la catégorie la plus pertinente sémantiquement pour le texte
  * en utilisant les embeddings d'entités et de catégories appris à partir de Wikipedia

**Préparation (apprentissage)**
1. **Préfabriquation du Anchor-Text Dictionary** :
  * tous les **anchor text** (texte cliquable d'un lien hypertexte ; dit aussi *label de lien*, ou *texte de lien*) sont recupérés
  * le texte de lien est une mention utilisé comme **clé** dans le dico, et le lien qu'il référence est l'**entité**


**Question**
  * Le dictionnaire est-il construit sur le Wikipedia limité à la langue des textes (e.g. seulement l'anglais) ?

**Voir aussi**:
  * aria/dataless intent recognition/2019 Knowledge-Based Dataless Text Categorization
  * aria/dataless intent recognition/2019 Knowledge-Based Short Text Categorization Using Entityand Category Embedding

##### Weak supervision

###### 2019 A clinical text classification paradigm using weak supervision and deep representation
L'article présente une méthode simple pour adresser l'absence de données annotées pour une tâche de classification de textes de rapports cliniques. Il propose de construire une base d'entraînement à l'aide d'une méthode à base de REGEX. Le principe est de conclure en l'appartenance d'un document à une catégorie sir une phrase de ce doc comprend un mot-clé ou une combinaison de mots-clés prédéfinis pour cette catégorie (e.g. **uses tobacco** pour la classe **smoker**). Le jeu d'entraînement est donc potentiellement bruité et ne couvre que les motifs de mots-clés prédéfinis. Pour être robuste aux mots-clés inconnus, le modèle vectoriel emploi les embeddings de mots qui rapprochent les mots inconnus des connus. En effet, le vecteur d'un texte est la moyenne des embeddings des mots qu'il comprend (occurrence ou **type** (le doc est défini comme un ensemble de mots)?). Les expérimentations montrent de très bonnes performances en classification binaire avec le CNN (F1 à 0.92 & 0.97  pour 0.91 & 0.93 pour les règles) avec la quantité de données d'entraînement disponible (31861  et 22471) mais moins bonnes (0.77 pour 0.88 pour les règles) en multi-classe (5 classes) pour deux raisons : (1) faible quantité de données (389), (2) un important déséquilibre du jeu annoté (deux classes couvrant seuelement 5%).
* **Atout** : 
  * disponibilité de données non annotées
  * disponibilité de l'expertise pour prédéfinir les motifs de mots-clés pour les règles
* **Manque**:
  * quantité suffisante de textes à annoter pour le deep learning (CNN)
*  **Avantage**:
  * l'augmentation de données annotées améliorera les performances de classification

#### query analysis
* 2018_Chapter_UnderstandingInformationNeeds

## A lire en priorité



