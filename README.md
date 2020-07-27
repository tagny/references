# Revue de littérature
Notes de lectures d'articles et de rapports scientifiques


## aria/dataless intent recognition
**Problème** : absence de données annotées pour entrainer des algo de NLP

**Solutions**:
* **dataless text classification** : employer des connaissances (KB) à la place des données
  * exploite la similarité (lexicale, syntaxique ou sémantique) entre le texte à classifier et les catégories (pareil qu'en RI)
  * bon si données non annotées indisponibles
* **weak supervision** : générer automatiquement des données (par exemple avec un système de règles REGEX)
  * bon si textes brutes dispo et expertise de mots-clés dispo


### 2018 Doc2Cube - Allocating Documents to Text Cube without Labeled Data [En cours]
** Objectifs ** : Construire un modèle Doc2Cube de cubes de textes à partir d'un corpus textuel (D) automatiquement sans données annotées uniquement avec le label comme petit ensemble de termes semences
** Problème **: difficile d'annoter suffisamment de documents pour la classification
** Applications ** : 
* Faciliter les analyses textuelles multidimensionnelles
* révéler la similarité sémantique entre les labels, les termes et les documents
* BI : facilité d'exploration du corpus et de recherche des passages/articles désirés avec de simple requêtes
* biomedical: organisation de corpus en aspects maladies, gènes, protéine; 
* bioinfo : recherche plus facile d'article scientifique
** Définition ** 
* **modèle de cubes de textes (text cube)** : structure multidimensionnelle de données contenant des documents textuels, où les dimensions correspondent à plusieurs aspects (e.g. thème, temps, lieu) du corpus.
* **schéma prédéfini de cube (C)** : 

### 2018 DatalessTextClassificationATopicModelingApproachwith DocumentManifold [En cours]
**Problème**: classification sans donnée annotée

**Définitions**
* **Collecteur de documents** (*document manifold*): structure du voisinage local

**Existant**
* PRINCIPE : Entrainement basé sur les mots semences
* LIMITES : possibilité d'information supervisée **limitée** et **bruitée**

**Solution**
* HYPOTHESE: 
  * Les docs très similaires tendent à faire partie de la même catégorie
  * En préservant la structure de voisinage local, les documents d'entrainement peuvent être reliés de tel sorte que ça propage la supervision même si plusieurs d'entre eux ne contiennent pas de mots semences, et ça modifie simultanément l'info d'annotation bruitée uniquement pour les docs contenant des mots semences non pertinents.


### 2018 - [GOOD] web - How to Build Your Own Text Classification Model Without Any Training Data [FINI][MISE EN PRATIQUE INTERROMPUE]

**L'ARTICLE PROPOSE LE WEB SERVICE Custom Classifier SUR UNE API NOMMEE [ParallelDots API](https://www.paralleldots.com/text-analysis-apis) POUR DEVELOPPER UN CLASSIFIEUR A APPRENTISSAGE ZERO SUR DES CATEGORIES PERSONELLES**

**LE SERVICE Custom Classifier EST DEPRECIE ET EST EN COURS DE REDEVELOPPEMENT (juillet 2020) DONC IMPOSSIBLE D'ALLER PLUS LOIN POUR L'INSTANT**

**Définitions**
* **apprentissage zéro** (*zero-shot learning*) : c'est être capable de résoudre une tâche malgré n'avoir reçu aucun exemple de cette tâche.

**Avantage du type de problème**
* réduction du cout et du temps nécessaire pour construire un modèle de classification

**Solution**
* FEATURES : 
  * catégories retournées chacune associée de son score de probabilité 
  * le service web parvient à retrouver avec de bons scores les catégories relatives à la requête même avec une division fine (les championnats européens de foot)
* SETUP :
  * [Créer un compte gratuit](https://user.apis.paralleldots.com/signing-up?utm_source=website&utm_medium=footer&utm_campaign=signup) ParallelDots API et se connecter
  * le plan gratuit est mensuellement limité à **1000 requêtes / jour** à raison de **20 requêtes par minute**
  * le plan gratuit se renouvelle chaque mois
  * **LE SERVICE Custom Classifier EST DEPRECIE ET EST EN COURS DE REDEVELOPPEMENT (juillet 2020) DONC IMPOSSIBLE D'ALLER PLUS LOIN**


### 2019 Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings [FINI][A RESUMER]

**solution**
* 

### 2020 Early Forecasting of Text Classification Accuracyand F-Measure with Active Learning [EN COURS]

**Définitions**
* **Apprentissage actif**: un apprenant sélectionne les données qui seront annotées avec pour but d'optimiser l'efficience de l'apprentissage (plus de performance avec peu de données) en demandant des efforts d'annotation où il devrait être très utile.
* **TPC** (*Training Percent Cutoff*): proportion minimale suffisante de données d'entrainement
* **closest-to-hyperplane selection algorithm** : on sélectionne les *n* échantillons qui sont les plus proches de la frontière de décision (l'hyperplan)
* **modèle de régression de courbe d'apprentissage** : calcul la performance y (erreur, accuracy, f1, etc.) en fonction du pourcentage ou de la quantité x de données d'entrée, afin de prévoir/anticiper le x=TPC à partir duquel il ne sera plus nécessaire d'augmenter les données d'entrainement (plateau de performance):
  * linéaire : y = ax+b
  * logarithmique : y = a log(x) + b
  * exponential : y = ba^x
  * power law : y = bx^a

**Existant**
* **problème**: l'annotation de données d'entrainement, un goulot d'étranglement pour la classif de texte
* **solution**: minimiser le coût d'annotation avec l'apprentissage actif utilisant des méthodes d'arrêt (lorsque plus d'annotation n'est plus nécessaire)
  * **conditions d'utilité des mth d'arrêt** : prévoir efficacement la performance des modèles de classif de textes
  * **Comment**: utilisation de modèles logarithmiques régressés sur une portion des données pendant que l'apprentissage progresse
  * **Question**: QUELLE PORTION (QUANTITE) DE DONNEES EST NECESSAIRE POUR UNE PREVISION PRECISE ? i.e. PEU => *prévision tôt*  ou BEAUCOUP => *prévision précise* ?
  * cette question est encore plus importante en apprentissage actif 

**Question de recherche**
* Quelle est la différence dans la prévision du nombre de données d'entrainement nécessaire ? Algo ? Accurracy vs F1 ? 

**Conclusions**
* la F1 est plus difficile à prévoir
* prévision facile pour les arbres de décision, moyen pour les SVM, plus difficile pour le réseaux de neurones.
* le logarithme de la métrique de performance est le meilleur modèle de prévision

**Mes questions**
* Anticiper le nombre de données annotées est compréhensible; ça permet de savoir combien de données annotées sont nécessaire pour atteindre une certaine performance. Mais je ne vois pas en quoi c'est utile de prévoir le pourcentage de données ?

**Voir aussi**:
* 2018 ImpactofBatchSizeonStoppingActiveLearningforTextClassification
* 2018 SVMActiveLearningAlgorithmswithQuery-by-CommitteeVersusClosest-to-HyperplaneSelection

### 2015 slides - Text Classification without Supervision - Incorporating World Knowledge and Domain Adaptation [EN COURS]

**Existant**:
* les défis de la catégorisation de textes [**en production**]
  * Annotation par des expert du domaine pour des problèmes de grande taille
  * Domaines et tâches divers : thèmes, langages, etc.
  * des textes court et bruités: tweets, requêtes, etc.
* Approches traditionnelles : adaptation au domaine cible (i.e. sémi-supervision, transfer learning et zero-shot learning )
  *  mais difficile de déterminer quel est le domaine cible ? e.g. distinguer le *sport* du *divertissement*

**Solution proposée 2008 & 2014**:
* **apprentissage activé par les connaissances** au travers de millions d'entités et concepts, de milliards de relations
  * Wikipedia, freebase, Yago, ProBase, DBpedia
* Hypothèse : **les labels portent beaucoup d'information** (**ET NOUS AVONS EN PLUS DES DESCRIPTIONS**)
* Solution 1: 
  1. grâce aux connaissances du domaine, représenter les labels et documents dans le même espace
  2. calculer les similarités entre document et label
* choisir les labels

**Difficultés liées à l'utilisation des connaissances**
* **phase apprentissage** :  Monter en charge, adaptation au domaine, classes en domaine ouvert ==> **présenter quelques exemples intéressants
* **phase d'inférence** spécification des connaissances; désambiguïsation,  
  * Utilisation de la similarité cosinus, Average (Best toujurs), Max matching, Hungarian matching (plus on a de concept, mieux il est)
* **phase de représentation**: représentation des données différentes de celle des connaissances ==> **comparer différentes représentations**
  * polysémie et synonymie

**Voir aussi**:
* ./2014 On Dataless Hierarchical Text Classification
* ./2008 Importance of Semantic Representation - Dataless Classification
* ./2014 Transfer Understanding from Head Queries to Tail Queries [TO READ]
* ./2015 Open Domain Short Text Conceptualization [TO READ]

### 2014 Transfer Understanding from Head Queries to Tail Queries [EN COURS]

* En recherche d'info, le plus grand défi réside dans la gestion des **requêtes de "queue"**
* **requête de "queue"** : requête qui survient très rarement (**REFERENCE POTENTIELLE AUX MOTIFS RAREMENT SOLLICITES DANS NOTRE CAS => FAIBLEMENT REPRESENTEES DANS LES LOGS => DIFFICILE DE LES APPRENDRE PAR DES ALGORITHME D'ORDONNEMENT**) 
* Les **requêtes de "tête"** sont facile à gérer car leurs intentions sont mises en évidence par le grand nombre de "données clic" (i.e. **de sollicitation**)
* Le problème est de savoir *COMMENT MIEUX ESTIMER LES INTENTIONS D'UNE REQUETE*
* LITTERATURE : **la pertinence d'une url pour une requête q est estimée par la similarité moyenne entre elle et les anciennes requêtes q_i pondérée par le nombre de clics correspondants sur cette url lorsqu'elles ont été soumises**
  * PB : requete = **texte court** ==> insuffisance d'info contextuelle pour comparer la sémantique de 2 textes
  * LIMITE les ajustements avec la modélisation thématique ou le DNN extrait la sémantique latente ou hiérarchique des requêtes mais sont lentes à entrainer et à tester
* HYPOTHESE : **il est beaucoup plus utile de considérer ensemble la sémantique sur les anciennes requêtes et les clics d'utilisateurs, pour relier des requêtes différentes à la surface (lexique).**
* CONDITION & DEFI : **parvenir automatiquement** à correctement segmenter les requêtes en sous-expressions et identifier leurs concepts

### 2019 - slide - NLP from scratch - Solving the cold start problem for NLP [EN COURS]


### 2019_ESWC_KB_Short_Text_Classification_Using_Entity_and_Category_Embedding [FINI]

**CET ARTICLE VEND UNE APPROCHE QUI PERMET DE SE PRIVER D'ANNOTATION DE DONNEES D'ENTRAINEMENT GRACE A UNE KB, MAIS ELLE NE MARCHE QUE SI LES CATEGORIES DE LA TACHE CORRESPONDENT A DES CATEGORIES DE WIKIPEDIA ==> CE QUI REND DIFFICILE LA GENERALISABILITE DE LA METHODE**

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
* **e est directement associée à la catégorie c, notée c_{a_e}** : e apparait dans un article de Wikipedia qui a comme catégorie associée c_{a_e}
* C_{a_e} : ensemble des catégories directement associées à e
* sim(c,e) : similarité cosinus entre les vecteurs de c et e dans l'espace d'embedding
* A_{c_{a_e}} : ensemble des ancêtres de c_{a_e} dans la structure hiérarchicale des catégories dans la KB (Wikipedia)
* **Association mention-entité P(m_e|e)** : probabilité d'observer une mention m_e étant donnée l'entité e
* count(m_e, e) : nombre de lien utilisant m_e comme texte de lien pointant sur e comme destination
* M_e : ensemble de toutes les mentions qui peuvent se référer à e (qui pointent vers e dans les articles de Wikipédia)
* **Relation entité-contexte P(C_e|e)** : 
* e_c \in C_e : entité à laquelle se réfère une mention du contexte C_e de e 
* E_{C_e}} ensemble des entités qui peuvent être référencées par les mections de C_e.


**Principe (application)**:
1. Détection des mentions d'**entités** du texte en entrée : n-grammes qui matchent une entrée d'un dico (**Anchor-Text Dictionary**)
2. Génération, pour chaque mention, d'un ensemble d'**entités** candidates à l'aide du dico préfabriqué (**Anchor-Text Dictionary**)  
  * PETITE IDEE: *indexer le dico comme la librairie [SML](http://www.semantic-measures-library.org) [Harispe et al. , 2013] indexe les mots dans des fichiers multiples: un chunkfile pour les n-grammes débutant par les mêmes 2 premiers chars particulier, indexer les noms de fichiers de  débuts de n-grammes dans un fichier chunk_index)*
3. Application de la **méthode probabiliste** proposée (similaire à un classifieur bayésien P(c|t) ~= P(c,t) = P(c)P(t|c) pour trouver la catégorie la plus pertinente sémantiquement pour le texte. **Comment calculer P(c,t) ?** :
  * en utilisant les embeddings d'entités et de catégories appris à partir de Wikipedia  
  * P(c,t) = \sum_{e in E_t} P(e)P(c|e)P(m_e|e)P(C_e|e)
  * P(e) = \frac{1}{N} *entités équiprobables* ou encore *distribution uniforme*
  * P(c|e) :  si c est un c_{a_e} alors P(c|e) = P(c_{a_e}) = \frac{sim(c_{a_e}, e)}{\sum_{c'_{a_e} \in C_{a_e}} sim(c'_{a_e}, e)} sinon P(c|e) = \sum_{c_{a_e} \in C_{a_e}} P(c_{a_e}|e) P(c|c_{a_e})
  * P(c|c_{a_e}) = si c \in A_{c_{a_e}} alors \frac{1}{|A_{c_{a_e}}|} sinon 0
  * P(m_e|e) = \frac{count(m_e, e)}{\sum_{m'_e \in M_e} count(m'_e, e)}
  * P(C_e|e) = \sum_{e_c \in E_{C_e}} P(e_c|e)P(m_{e_c}|e_c)
  * P(e_c | e) = \frac{sim(e_c,e)}{\sum_{e' \in E} sim(e',e)}
  
**Embedding d'entités et catégories, pour calculer les similarités**
*  construction des réseaux de co-occurrence 
  *  entité-entité : poids = nombre de fois que 2 entités apparaissent dans le même article comme anchor-text (**graphe homogène**)
  * entité-categorie : poids = nombre de fois que l'entité pointe sur un lien dans un article classée dans la catégorie (bas de l'artcle) (**graphe hétérogène**)
*  Modèle d'embedding
  * objectif : **capturer la proximité de second-ordre** : calculé entre 2 sommets en considérant leur sommets partagés (voisins) : plus on partage de voisins plus on devrait être proche
  * C'est traduit par la proba conditionel P(v_j|v_i) = \frac{exp(-u_j^T * u_i)}{\sum_{v_k \in V} exp(-u_k^T * u_i)} ~= \frac{w_{ij}}{d_i}
  * V : ensemble des sommets connectés avec v_i
  * u_i : vecteur du sommet v_i
  * w_{ij} : poids de l'arête entre v_i et v_j
  * d_i : degré sortant de v_i
  * en se basant sur la divergence KL, il faut minimiser O_{homo} = - \sum_{(v_i, v_j) \in E} w_{ij} log(p(v_j | v_i)) => graphe entité-entité
  * Pour apprendre en même temps sur le graphe heterogène: O_{heter} = O_{ee} + O_{ec}


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
  * 2018 TECNE - Knowledge Based Text ClassificationUsing Network Embeddings
  * Pour la méthode de minimisation de O_{heter}, 2015 PTE-PredictiveTextEmbeddingthroughLarge-scaleHeterogeneousTextNetworks


### 2020 Description Based Text Classification with Reinforcement Learning [MIS DE COTE]
**L'article n'est pas dataless, il utilise la description pour retrouver le passage pertinent à classifier c'est tout. le modèle reste gourmand en données annotées**

**Nom de la méthode** : SQuAD-style machine reading comprehension task 

Ce papier décrit la classification de texte sans données annotées comme prenant *en entrée* la **description de la catégorie** et le **texte**, pour déterminer si le texte est de la catégorie.
Plusieurs thèmes apparaissent dans un document, aussi bien que plusieurs sentiments sur différents aspects. La phlosophie des auteurs est que **le modèle doit apprendre à associer le texte pertinent à l'aspect ciblé, et ensuite décider du sentiment**. L'association est formalisée et permet de dire **explicitement** au modèle ce qu'il doit classifier.


### 2018 A Pseudo Label based Dataless NB Bayes Algorithm for Text Classification with Seed Words [EN COURS]

Les auteurs soutiennent que la production de données annotées est trop exigente pour l'effort humain même en petite quantité. L'approche proposée se base sur une tâche qu'ils estiment plus facile, la proposition de **mots "semences"** représentatifs des labels. L'algorithme apprend directement à partir des documents non annotés et des mots "semences".

**Nom de la méthode** : PL-DNB (*Pseudo-Label based dataless Naive Bayes classifier*)

**Données dispo au départ**: 
* S^L : mots clés sélectionnés manuellement à partir des labels de catégories (1 seul en moyenne)
* S^D : mots clés sélectionnés manuellement par expertise de domaine à partir d'une liste produite automatiquement (non supervisée)
* D_U : ensemble de documents non annotés

**Application : Bayésien Naïf sémi-supervisé**
* **sémi-supervisé** : appris à la fois à partir de données annotées et non annotées
* y = \max_{c_i \in C}P(y = c_i|d)
* P(y = c_i|d) ~= P(y = c_i)\prod_{j=1}^{||W} P(w_j|y = c_i)^{N_{d,w_j}}
* N_{d,w_j}: nb occurrence de w_j dans d

**Entraînement**:
L'entraînement est une boucle de génération ou màj d'annotations et d'estimation des paramètres du classifieur bayésien sémi-supervisé.
1. **génération et màj d'annotations de données annotées D_L**
  * *Initialisation*
  
2. **Estimation des paramètres \theta={P(y), P(w_j|y)}) du Bayésien Naïf sémi-supervisé : Expectation-Maximisation (EM)**


### medium_com_ai_medecindirect_unsupervised_text_classification [WEB][FINI]

**Définitions**
*  **StackedEmbeddings** (vecteurs empilés) : concaténation des embeddings d'un mot obtenus à partir de différentes techniques (Glove, w2v, Elmo, BERT, fastText, etc.)

**Hypothèse** : apprendre d'une description est suffisant pour un homme et devrait l'être aussi pour la machine. On ne devrait pas forcément nécessiter une grande masse de données annotées pour qu'un système reconnaisse un concept.

**Tâche** : classer les feedbacks de clients de MédecinDirect dans 11 catégories prédéfinies (indiquant probablement la raison de l'insatisfaction ou du commentaire du client)

**Problème** : très faible quantité de données annotées, très déséquilibrées (une classe avec seulement 3 exemples sur 200 pour 11 catégories). Par conséquent, la configuration n'est pas bonne pour de la classificaton supervisée traditionnelle.

**Approche** : 

*  utiliser les embeddings de mots et construire le vecteur vecteurs empilés d'embeddings pour une phrase (requête : CommentVector) ou un document (description de catégorie faite de 3 phrases synthétiques exemples : CategoryVector) (documentEmbeddings).
*  calculer la similarité cosinus entre les vecteurs de requêtes et ceux de la catégorie
*  la catégorie ayant le score de similarité le plus élevé est retenu pour la requête, s'il n'y en a pas, alors la catégorie de la requête est définie comme "incertaine" (un label en plus).
*  Utiliser les exemples annotées pour déterminer les hyperparamètres : type de documentEmbeddings, le type de description à adopter pour les labels, et à quel point le model doit être, etc.



## aria/less annotation/Weak supervision


### 2019 A clinical text classification paradigm using weak supervision and deep representation [FINI]
L'article présente une méthode simple pour adresser l'absence de données annotées pour une tâche de classification de textes de rapports cliniques. Il propose de construire une base d'entraînement à l'aide d'une méthode à base de REGEX. Le principe est de conclure en l'appartenance d'un document à une catégorie sir une phrase de ce doc comprend un mot-clé ou une combinaison de mots-clés prédéfinis pour cette catégorie (e.g. **uses tobacco** pour la classe **smoker**). Le jeu d'entraînement est donc potentiellement bruité et ne couvre que les motifs de mots-clés prédéfinis. Pour être robuste aux mots-clés inconnus, le modèle vectoriel emploi les embeddings de mots qui rapprochent les mots inconnus des connus. En effet, le vecteur d'un texte est la moyenne des embeddings des mots qu'il comprend (occurrence ou **type** (le doc est défini comme un ensemble de mots)?). Les expérimentations montrent de très bonnes performances en classification binaire avec le CNN (F1 à 0.92 & 0.97  pour 0.91 & 0.93 pour les règles) avec la quantité de données d'entraînement disponible (31861  et 22471) mais moins bonnes (0.77 pour 0.88 pour les règles) en multi-classe (5 classes) pour deux raisons : (1) faible quantité de données (389), (2) un important déséquilibre du jeu annoté (deux classes couvrant seuelement 5%).
* **Atout** : 
  * disponibilité de données non annotées
  * disponibilité de l'expertise pour prédéfinir les motifs de mots-clés pour les règles
* **Manque**:
  * quantité suffisante de textes à annoter pour le deep learning (CNN)
*  **Avantage**:
  * l'augmentation de données annotées améliorera les performances de classification


## aria/knowledge graph

### 2017 Graph-based_Text_Representations_Tutorial_EMNLP_2017 [TURORIEL][FINI][A RESUMER]

**Objectifs**: Booster la fouille de textes, le TALN, et la RI avec les graphes

**Problèmes de la représentation par BoW**: hypothèse d'indépendance entre termes et pondération par fréquence de termes

**Intérêt de la représentation de texte par graphe**: capturer la **dépendance** entre les termes, de leur **ordre** et la **distance** entre eux.


## aria/query analysis

### 2018_Chapter_UnderstandingInformationNeeds [SUSPENDU]

en RI, la compréhension de l'information nécessitée par l'utilisateur passe par une bonne compréhension de sa requête. Pour cela, il existent des techniques comme la classification de la requête suivant des buts de niveau élevé (intention), segmentation en parties allant ensemble (e.g. noms composés), interpréter la structure de la requête, reconaître et désambiguïser les entités mentionnées, déterminer si un service spécifique ou un segment/domaine des contenus en ligne (*verticals*, e.g. shopping, voyage, recherche d'emploi, etc.) doit être invoqué.

#### Analyse sémantique de requête

**Classification de requête**

**Annotation de requête**

**Interprétation de requête**

## aria\short texts similarity\metric learning

### 2019 ATutorialonDistanceMetricLearning-MathematicalFoundationsAlgorithmsandExperiments [EN COURS]

*  Une distance standard peut ignorer des propriétés importantes dans le dataset = son utilisation par un apprentissage rendant ce dernier non optimal
*  L'objectif de l'apprentissage d'une distance, c'est de rapprocher autant que possible les objets similaires, tout en éloignant les différents, pour améliorer la qualité des applications 
*  Les bases de l'apprentissage de distance sont :
  *  **l'analyse convexe** : pour la présentation et la résolution de pbs d'obtimisation (estimation de paramètres)
  *  **l'analyse matricielle** : pour la compréhension de la discipline, la paramétrisation des algo, et l'obtimisation par les vecteurs propres
  *  **la théorie de l'information** : qui a motivé plusieurs des algorithmes
  
  
## aria/embeddings

### 2017 SIF-a_simple_but_tough_to_beat_baseline_for_sentence_embeddings [LU][A RESUMER]

### 2018-sent2vec [PRIORITAIRE]

### 2014 doc2vec [PRIORITAIRE]

### 2015 [GOOD PERF] AdaSent-SelfAdaptiveHierarchicalSentenceModel [PRIORITAIRE]

### 2020 SBERT-WK - A Sentence Embedding Method ByDissecting BERT-based Word Models [PRIORITAIRE]
