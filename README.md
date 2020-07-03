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

###### 2019 - slide - NLP from scratch - Solving the cold start problem for NLP [EN COURS]

###### 2019_ESWC_KB_Short_Text_Classification_Using_Entity_and_Category_Embedding [EN COURS]
Ce travail propose d'adresser l'absence de données annotées. La technique consiste à utiliser la similarité "sémantique" entre catégories et textes en s'appuyant sur des techniques de **graph embedding**.
**Définitions et notation**
* t : texte à classifier
* **entité e** : un lien hypertexte de Wikipedia indexé par un texte de lien dans le dico préfabriqué **Anchor-Text Dictionary** (e)
* **mention m_e** : un terme dans le texte t qui peut se reférer à e 
* **contexte C_e** de e : ensemble d'autres mentions dans t exceptée celle de e 
* E_t : ensemble de toutes les entités possibles contenues dans t
* **Popularité P(e) de E** : Probabilité qu'un texte pris au hasard contienne e
* N : nombre d'entités dans le dico
* **Relation entité-catégorie P(c|e)** 
* **e est directement associée à la catégorie c, notée c_{a_e}** : e apparait dans un article de Wikipedia qui a la comme catégorie associée c_{a_e}
* C_{a_e} : ensemble des catégories directement associées à e
* sim(c,e) : similarité cosinus entre les vecteurs de c et e dans l'espace d'embedding
* A_{c_{a_e}} : ensemble des ancêtres de c_{a_e} dans la structure hiérarchicale des catégories dans la KB (Wikipedia)
* **Association mention-entité P(m_e|e)** : probabilité d'observer une mention m_e étant donnée l'entité e
* count(m_e, e) : nombre de lien utilisant m_e comme texte de lien pointant sur e comme destination
* M_e : ensemble de toutes les mentions qui peuvent se référer à e (qui pointent vers e dans les articles de Wikipédia)
* **Relation entité-contexte P(C_e|e)** : ...

**Principe (application)**:
1. Détection des mentions d'**entités** du texte en entrée : n-grammes qui matchent une entrée d'un dico (**Anchor-Text Dictionary**)
2. Génération, pour chaque mention, d'un ensemble d'**entités** candidates à l'aide du dico préfabriqué (**Anchor-Text Dictionary**)  
  * PETITE IDEE: *indexer le dico comme la librairie [SML](http://www.semantic-measures-library.org) [Harispe et al. , 2013] indexe les mots dans des fichiers multiples: un chunkfile pour les n-grammes débutant par les mêmes 2 premiers chars particulier, indexer les noms de fichiers de  débuts de n-grammes dans un fichier chunk_index)*
3. Application la **méthode probabiliste** proposée (similaire à un classifieur bayésien P(c|t) ~= P(c,t) = P(c)P(t|c) pour trouver la catégorie la plus pertinente sémantiquement pour le texte. **Comment calculer P(c,t) ?** :
  * en utilisant les embeddings d'entités et de catégories appris à partir de Wikipedia  
  * P(c,t) = \sum_{e in E_t} P(e)P(c|e)P(m_e|e)P(C_e|e)
  * P(e) = \frac{1}{N}
  * P(c|e) :  si c_{a_e} alors P(c|e) = P(c_{a_e}) = \frac{sim(c_{a_e}, e)}{\sum_{c'_{a_e} \in C_{a_e}} sim(c'_{a_e}, e)} sinon P(c|e) = \sum_{c_{a_e} \in C_{a_e}} P(c_{a_e}|e) P(c|c_{a_e})
  * P(c|c_{a_e}) = si c \in A_{c_{a_e}} alors \frac{1}{|A_{c_{a_e}}|} sinon 0
  * P(m_e|e) = \frac{count(m_e, e)}{\sum_{m'_e \in M_e} count(m'_e, e)}
  
  

**Préparation (apprentissage)**
1. **Préfabriquation du Anchor-Text Dictionary** :
  * tous les **anchor text** (texte cliquable d'un lien hypertexte ; dit aussi *label de lien*, ou *texte de lien*) sont recupérés
  * le texte de lien est une mention utilisé comme **clé** dans le dico, et le lien qu'il référence est l'**entité**


**Questions**
  * Le dictionnaire est-il construit sur le Wikipedia limité à la langue des textes (e.g. seulement l'anglais) ?
  * Pourquoi la popularité de e n'est pas estimée à partir de wikipédia (ou le dico) comme l'association mention-entité ?

**Voir aussi**:
  * aria/dataless intent recognition/2019 Knowledge-Based Dataless Text Categorization
  * aria/dataless intent recognition/2019 Knowledge-Based Short Text Categorization Using Entityand Category Embedding

##### Weak supervision

###### 2019 A clinical text classification paradigm using weak supervision and deep representation [FINI]
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



