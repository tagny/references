# Revue de littérature
Notes de lectures d'articles et de rapports scientifiques


<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Revue de littérature](#revue-de-littrature)
	- [Evaluation](#evaluation)
		- [2020-DamirKrstinić-MultiLabelClassifPerfEvalWithConfusionMatrix.pdf](#2020-damirkrstini-multilabelclassifperfevalwithconfusionmatrixpdf)
	- [dataless intent recognition](#dataless-intent-recognition)
		- [2020-ExploitingClozeQuestionsForFewShotTextClassifAndNLI-Schick](#2020-ExploitingClozeQuestionsForFewShotTextClassifAndNLI-Schick)
		- [2019-WenpengYin-BenchmarkingZeroshotTextClassificationDatasetsEvaluationAndEntailmentApproach](#2019-wenpengyin-benchmarkingzeroshottextclassificationdatasetsevaluationandentailmentapproach)
		- [2020-YuMeng-TextClassifUsingLabelNamesOnlyALangModelSelfTrainingApproach](#2020-yumeng-textclassifusinglabelnamesonlyalangmodelselftrainingapproach)
		- [2018 Doc2Cube - Allocating Documents to Text Cube without Labeled Data [En cours]](#2018-doc2cube-allocating-documents-to-text-cube-without-labeled-data-en-cours)
		- [2018 DatalessTextClassificationATopicModelingApproachwith DocumentManifold [En cours]](#2018-datalesstextclassificationatopicmodelingapproachwith-documentmanifold-en-cours)
		- [2018 - [GOOD] web - How to Build Your Own Text Classification Model Without Any Training Data [FINI][MISE EN PRATIQUE INTERROMPUE]](#2018-good-web-how-to-build-your-own-text-classification-model-without-any-training-data-finimise-en-pratique-interrompue)
		- [2019 Towards Unsupervised Text Classification Leveraging Experts and Word Embeddings [FINI][A RESUMER]](#2019-towards-unsupervised-text-classification-leveraging-experts-and-word-embeddings-finia-resumer)
		- [2020 Early Forecasting of Text Classification Accuracyand F-Measure with Active Learning [EN COURS]](#2020-early-forecasting-of-text-classification-accuracyand-f-measure-with-active-learning-en-cours)
		- [2015 slides - Text Classification without Supervision - Incorporating World Knowledge and Domain Adaptation [EN COURS]](#2015-slides-text-classification-without-supervision-incorporating-world-knowledge-and-domain-adaptation-en-cours)
		- [2014 Transfer Understanding from Head Queries to Tail Queries [EN COURS]](#2014-transfer-understanding-from-head-queries-to-tail-queries-en-cours)
		- [2019 - slide - NLP from scratch - Solving the cold start problem for NLP [FINI]](#2019-slide-nlp-from-scratch-solving-the-cold-start-problem-for-nlp-fini)
		- [2019_ESWC_KB_Short_Text_Classification_Using_Entity_and_Category_Embedding [FINI]](#2019eswckbshorttextclassificationusingentityandcategoryembedding-fini)
		- [2020 Description Based Text Classification with Reinforcement Learning [MIS DE COTE]](#2020-description-based-text-classification-with-reinforcement-learning-mis-de-cote)
		- [2018 A Pseudo Label based Dataless NB Bayes Algorithm for Text Classification with Seed Words [EN COURS]](#2018-a-pseudo-label-based-dataless-nb-bayes-algorithm-for-text-classification-with-seed-words-en-cours)
		- [medium_com_ai_medecindirect_unsupervised_text_classification [WEB][FINI]](#mediumcomaimedecindirectunsupervisedtextclassification-webfini)
	- [dataless annotation/data augmentation](#dataless-annotationdata-augmentation)
		- [2020 When does data augmentation help generalization in NLP [EN COURS][PRIORITAIRE]](#2020-when-does-data-augmentation-help-generalization-in-nlp-en-coursprioritaire)
		- [2019 EDA - Easy Data Augment Tech for Boosting Perf on Text Classif [En cours][PRIORITAIRE]8](#2019-eda-easy-data-augment-tech-for-boosting-perf-on-text-classif-en-coursprioritaire8)
	- [dataless annotation/Weak supervision](#dataless-annotationweak-supervision)
		- [2019 A clinical text classification paradigm using weak supervision and deep representation [FINI]](#2019-a-clinical-text-classification-paradigm-using-weak-supervision-and-deep-representation-fini)
	- [embeddings](#embeddings)
		- [2014 GloVe - Global Vectors for Word Representation](#2014-glove-global-vectors-for-word-representation)
		- [2017 SIF-a_simple_but_tough_to_beat_baseline_for_sentence_embeddings [EN COURS]](#2017-sif-asimplebuttoughtobeatbaselineforsentenceembeddings-en-cours)
		- [2018-sent2vec](#2018-sent2vec)
		- [2018-ImprovingLanguageUnderstandingByGenerativePreTraining-RadfordNarasimhanSalimansSutskever-GPT](#2018-improvinglanguageunderstandingbygenerativepretraining-radfordnarasimhansalimanssutskever-gpt)
		- [2014 doc2vec](#2014-doc2vec)
		- [2017 fasttext - Enriching Word Vectors with Subword Information](#2017-fasttext-enriching-word-vectors-with-subword-information)
		- [2020 P-SIF - Document Embeddings Using Partition Averaging [PRIORITAIRE]](#2020-p-sif-document-embeddings-using-partition-averaging-prioritaire)
		- [2015 [GOOD PERF] AdaSent-SelfAdaptiveHierarchicalSentenceModel [PRIORITAIRE]](#2015-good-perf-adasent-selfadaptivehierarchicalsentencemodel-prioritaire)
		- [2019 Bert - Pre-Training of Deep Bidirectional Transformers for Language understanding](#2019-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding)
		- [2020 CamemBERT - a Tasty French Language Model](#2020-camembert-a-tasty-french-language-model)
		- [2020 SBERT-WK - A Sentence Embedding Method By Dissecting BERT-based Word Models [PRIORITAIRE]](#2020-sbert-wk-a-sentence-embedding-method-by-dissecting-bert-based-word-models-prioritaire)
		- [2019 Sentence-BERT [PRIORITAIRE]](#2019-sentence-bert-prioritaire)
		- [2020 Improving Sentence Representations via Component Focusing (CF-BERT)](#2020-improving-sentence-representations-via-component-focusing-cf-bert)
		- [2019 spherical-text-embedding](#2019-spherical-text-embedding)
	- [knowledge graph](#knowledge-graph)
		- [2017 Graph-based_Text_Representations_Tutorial_EMNLP_2017 [TURORIEL][FINI][A RESUMER]](#2017-graph-basedtextrepresentationstutorialemnlp2017-turorielfinia-resumer)
	- [query analysis](#query-analysis)
		- [2018_Chapter_UnderstandingInformationNeeds [SUSPENDU]](#2018chapterunderstandinginformationneeds-suspendu)
			- [Analyse sémantique de requête](#analyse-smantique-de-requte)
	- [short texts similarity](#short-texts-similarity)
		- [2019 Sentence Similarity Techniques for Short vs Variable Length Textusing Word Embeddings [FINI]*](#2019-sentence-similarity-techniques-for-short-vs-variable-length-textusing-word-embeddings-fini)
	- [short texts similarity / metric learning](#short-texts-similarity-metric-learning)
		- [2019 ATutorialonDistanceMetricLearning-MathematicalFoundationsAlgorithmsandExperiments [EN COURS]](#2019-atutorialondistancemetriclearning-mathematicalfoundationsalgorithmsandexperiments-en-cours)
		- [2019 Metric Learning for Dynamic Text Classification [EN COURS]](#2019-metric-learning-for-dynamic-text-classification-en-cours)
	- [misc-suggested](#misc-suggested)
		- [2020-LuyuGao-ModularizedTransfomerBasedRankingFramework](#2020-luyugao-modularizedtransfomerbasedrankingframework)
		- [2020-MujeenSung-BiomedicalEntityReprWithSynonymMarginalization](#2020-mujeensung-biomedicalentityreprwithsynonymmarginalization)
	- [speech recognition](#speech-recognition)
		- [2020-MLSALargeScaleMultilingualDatasetForSpeechResearch-VineelPratap](#2020-mlsalargescalemultilingualdatasetforspeechresearch-vineelpratap)
	- [TEMPLATE](#template)
		- [PDF FILE NAME](#pdf-file-name)

<!-- /TOC -->



## Evaluation

### 2020-DamirKrstinić-MultiLabelClassifPerfEvalWithConfusionMatrix.pdf
**Problème** : comment estimer et interpréter l'efficacité d'une approche de classification multi-label (MLC) avec une matrice de confusion ?

**Solution**: l'article propose une nouvelle approche pour calculer la matrice de confusion pour la classif multi-label:
* l'approche comble les lacunes des approches existantes
* Grâce à sa polyvalence (*abilité à s'adapté ou être adaptable à différentes fonctions ou activités*) elle peut donc être employée dans différents domaines

**contexte**
* Difficultés des algos MLC:
  * affecter plus d'un concept sémantique à chaque instance
  * existence d'une correlation entre différents labels
  * nombre inégal d'occurrences de labels dans les données (déséquilibre d'annotation de train)
	* chaque label n'a pas le même nombre d'instances : certains sont très sollicités/courants quand d'autres peuvent être rares en réalité et n'apparaitre que dans une très faible proportion des exemples
	* chaque instance n'a pas le même nombre de labels
  * différents labels sont souvent très similaires dans le contexte de certaines instances ; ce qui rend difficile leur annotation non ambigüe même par un humain
    * en plus de l'efficacité des algos, il faut **revéler les relations entre labels** et **indiquer clairement la faiblesse du classifieur (les ktqs du pb qui biaise ou handicape l'algo)** ==> **grâce à la matrice de confusion**
* La classif multi-label : entrainer une fonction pour prédire un vecteur binaire à dim defini par les labels (1 si le label est pertinent, 0 sinon
* Les **metriques couramment utilisées** distinguent l'éval basée sur les labels et celle basée sur les exemples pour un système H et un jeu de test D de taille n et un nb de labels candidats q:
  * eval basée sur les exemples :
    * Justesse_EB(H, D) = (1/n)(\sum_{i=1}^n |Yi \inter Z_i|/|Yi \union Z_i| : proportion des labels correctement prédits
	* Precision_EB(H, D) = (1/n)(\sum_{i=1}^n |Yi \inter Z_i|/|Z_i| : proportion de labels correctement prédits par le total de labels prédits
	* Rappel_EB(H, D) = (1/n)(\sum_{i=1}^n |Yi \inter Z_i|/|Yi| : proportion de labels correctement prédits par le total de labels attendus
	* F1_EB(H, D) = (2*Precision_EB*Rappel_EB)/(Precision_EB + Rappel_EB) =  (1/n)(\sum_{i=1}^n 2*|Yi \inter Z_i|/|Yi XOR Z_i|
	* HammingLoss_EB(H, D) = (1/n)(\sum_{i=1}^n |Yi \inter Z_i|/q : estime le nb moyen de fois où la pertinence d'un example pour un label est incorrectement prédit i.e. il y a prise en compte de l'erreur de prédiction (un label incorrect est prédit) et l'erreur de manquement (un label pertinent n'est pas prédit).
	* Justesse de sous-ensemble (**subset accuracy**) ou correspondance exacte (**exact match**) : (1/n) \sum_{i=1}^n I(Y_i=Z_i)
  * eval basée sur les labels : les labels sont considérés comme séparés, réduisant le MLC à un classifieur binaire pour chaque label avec des TP, TN, TN, FN
    * Justesse_j = (TP_j + TN_j) / (TP_j + TN_j + FP_j + FN_j)
	* Precision_j = (TP_j) / (TP_j + FP_j)
	* Rappel_j = (TP_j) / (TP_j + FN_j)
	* F1_j = (2*Precision_j*Rappel_j)/(Precision_j + Rappel_j) = 2*TP_j / (2TP_j + FP_j + FN_j)
	* Moyenne macro d'une metrique B: B_macro(H, D) = (1/q) \sum_{j=1}^q B_j
	Moyenne micro d'une metrique B: B_micro(H, D) = (1/q) B(\sum_{j=1}^q TP_j, \sum_{j=1}^q TN_j, \sum_{j=1}^q FP_j, \sum_{j=1}^q FN_j)
* La matrice de confusion est complémentaire à ces métriques car :
  * elle revèle que les instances d'un label sont souvent classés par erreur avec un autre label qu'on peut bien identifier ; ce qui **suggère au concepteur d'améliorer l'algo en cherchant plus de ktqs qui pourraient aider à mieux distinguer les labels confondus**.
  * son analyse pourrait :
    * fournir un aperçu des relations entre ktqs et objets de données différentes dfts
	* revéler la structure inhérente des données.
* la matrice de confusion compare la classe prédite définissant une colonne et la classe attendue définissant une ligne:
  * la cellule (Y, Z) est le nb de fois qu'un objet de la classe Y est affecter à la classe Z
  * la diagonale représente le nb de classif précise pour chaque classe quand les élts hors-diagonal sont les erreurs de classif.
* On peut aussi **normaliser** la matrice de confusion:
  * la **matrice de rappel** est la division de chaque cellule par la somme de tous les élts de la même ligne
    * en diagonal, on obtient le rappel de chaque classe
	* les autres éléments de la même ligne, représente la probabilité qu'un objet de la classe Y (ligne) sera classé par erreur dans Z (colonne)
  * la **matrice de précision** est la division de chaque cellule par la somme de tous les élts de la même colonne
    * en diagonal, on obtient la précision de chaque classe
	* les autres éléments de la même colonne, représente la probabilité qu'un objet classé dans Z (colonne ou prédit) est effectivement de la classe Y (ligne ou attendu)

**Algo proposé pour la construction de la matrice de confusion en MLC**:
* **condition d'application** : chaque instance a au moins un label i.e. |Y|>0 **et** |Z|>0 ;  *peut-être créer un label OUT-OF-DOMAIN en plus pour les cas sans label*
* 4 scénarios pour une instance x :
  * 𝑌 = Z : C(x) = diag(Y)
  * |𝑌\𝑍| = 0, |𝑍\𝑌| > 0 : 𝐶 = [𝑌 ⊗ (𝑍\ 𝑌) + |𝑌| ⋅ 𝑑𝑖𝑎𝑔(𝑌) ]/|𝑍| avec ⊗ = le outer product ??
  * |𝑌\𝑍| > 0, |𝑍\𝑌| = 0 : 𝐶 = [(𝑌\𝑍) ⊗ 𝑍]/|𝑍| + 𝑑𝑖𝑎𝑔(𝑍)
  * |𝑌\𝑍| > 0, |𝑍 \𝑌| > 0 : 𝐶 = [(𝑌\𝑍) ⊗ (𝑍\𝑌)]/|𝑍\𝑌| + 𝑑𝑖𝑎𝑔(𝑌 ∩ 𝑍)
  
  

## dataless intent recognition
**Problème** : absence de données annotées pour entrainer des algo de NLP

**Solutions**:
* **dataless text classification** : employer des connaissances (KB) à la place des données
  * exploite la similarité (lexicale, syntaxique ou sémantique) entre le texte à classifier et les catégories (pareil qu'en RI)
  * bon si données non annotées indisponibles
* **weak supervision** : générer automatiquement des données (par exemple avec un système de règles REGEX)
  * bon si textes brutes dispo et expertise de mots-clés dispo


### 2020-ExploitingClozeQuestionsForFewShotTextClassifAndNLI-Schick

**Question pour orienter la lecture**: 
  * What is a cloze question?
  * What do cloze questions have to do with text classification and natural language inference?
  * The paper deals with few shot text classification and NLI, how does it do that?
  * Is the approach pretrainabled on an other task dataset for the purpose of a zero-shot learning ?
  * why do we need few training data ?
  * Is the source code provided ?
  * Are all the resources provided (pretrained model, data, ...) ? For what languages ?
  * Is it simple to apply it?
  * How good is the approach compared to others?
  
**Pbs**: 
* few-shot learning == leverage some few examples to address the limit of unsupervised learning solutions based on language models and task descriptions
* comment combiner la description d'une tâche et l'apprentissage supervisé régulier sur un faible dataset annoté ?

**Approches précédentes**
* des solutions récentes sont proposées en zero-shot : voir (Radford
et al., 2019; Puri and Catanzaro, 2019)
* les approches précédentes utilisant des patrons le font pour définir le modèle de langue et non pour améliorer une tâche spécifique.

**proposition**: Entrainement par exploitation de patrons (motifs)
* reformulation de texte entrée en textes à trous suivant des patrons du NLP

**voir aussi**
* https://towardsdatascience.com/gpt-3-vs-pet-not-big-but-beautiful-7a73d17af981
* http://timoschick.com/explanatory%20notes/2020/10/23/pattern-exploiting-training.html [DETAILED][MORE EXAMPLES]
* https://inderjit.in/post/pet_for_nlp/
* https://www.aclweb.org/anthology/2020.coling-main.488.pdf
* https://www.pragmatic.ml/pet/ [PEDAGOGIC]


### 2019-WenpengYin-BenchmarkingZeroshotTextClassificationDatasetsEvaluationAndEntailmentApproach
**Pb**: 0shot-tc == affecter le label approprié à un morceau de texte indépendament du domaine et de l'aspect textuel décrit par le label.
* **aucune donnée annotée pour l'entrainement pour une partie ou tout l'ens des labels**

**Pb de l'Existant**
* modelisation restrainte à une seule tâche (catégorisation thématique)
* les labels sont encodés en indices sans tenir compte de leur sens
* difficile de comparer les différent travaux
  * datasets différents
  * métriqs d'éval différentes

**Proposition**: benchmark de datasets et d'éval, et une approche par implication (*entailment*) pour gérer différents aspects (thème, émotion, évts, ...) dans un paradigme unifié
1. déf plus général/large du pb: oshot-tc == apprendre une fct f: X -> Y où f(.) n'a jamais vu de données annotées des labels de Y lors de son dev
2 Dataset :
  * 3 aspects (thème, émotion, et situation == event).
  * données découpées :
    * train, dev, test
    * et séparation des classes vues et classes non vues
3. Evaluation: ?

**voir aussi**
* code source : https://github.com/CogComp/BenchmarkingZeroShot
* démo en ligne : https://cogcomp.seas.upenn.edu/page/demo_view/ZeroShotTextClassification
* Présentation à EMNLP : https://vimeo.com/409401114

### 2020-YuMeng-TextClassifUsingLabelNamesOnlyALangModelSelfTrainingApproach
**Pb**:
* les algos actuels ont besoin d'un bon jeu de données annotées par des humains
  * difficile et cher d'en obtenir
* l'humain n'a pourtant pas besoin de données d'entrainement
  * mais souvent d'un petit ens de termes décrivant les catégories

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
* **cube de texte (text cube)** : structure multidimensionnelle de données contenant des documents textuels, où les dimensions correspondent à plusieurs aspects (e.g. thème, temps, lieu) du corpus D.
* **schéma prédéfini de cube C = (L_1, L_2, ..., L_n) = (thème, temps, lieu)** : ensemble des n dimensions du cube.
* celulle de cube (l_{t_1}, l_{t_2}, ..., l_{t_n}) : localisation des documents ayant les labels l_{t_i} dans les dimensions L_i resp.
* **Tâche de construction de cube**: organiser un grand corpus de texte dans les classes d'un cube i.e. affecter n labels l_{t_1}, l_{t_2}, ..., l_{t_n} à chaque document d où l_{t_i} \in L_i représente la catégory de d dans la dimension L_i (L_i = Thème = {l_{t_1} = sport, l_{t_2} = business, ..., l_{t_n}=movies}

**Existant et limites**
* Génération des données d'entrainement limité par l'exigence d'effort d'ingénierie des caractéristiques
* dataless classification limitée par le risque de limite et non adaptation du corpus externe
* application de la classification de textes : données d'entrainement annotées nécessaires


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

### 2019 - slide - NLP from scratch - Solving the cold start problem for NLP [FINI]

**CETTE PRESENTATION RESTREINT LE PB DE COLD START A L'ANNOTATION FACILE ET RAPIDE D'UN JEU DE DONNEES PERTINENT POUR ENTRAINER UN MODELE DE MACHINE LEARNING LE PLUS EFFICACEMENT POSSIBLE. ELLE EXPLORE AINSI L'APPRENTISSAGE FAIBLEMENT SUPERVISE QUI PEUT ETRE AFFINER EN INCLUANT L'HUMAIN DANS LE PROCESSUS D'ANNOTATION PAR L'ACTIVE LEARNING. POUR CET AFFINEMENT, DEUX TECHNIQUES DE SELECTION DE DOCUMENTS PERTINENTS A ANNOTER SONT MISES EN AVANT : l'échantillonnage contradictoire ET l'échantillonnage de coude. LA PRESENTATION PROPOSE AUSSI LE TRANSFERT D'APPRENTISSAGE POUR BENEFICIER DE MODELES PREENTRAINES SUR D'AUTRES DONNEES.**

**Définition**
* Le **problème du cold start** consiste à trouver un moyen aussi efficace que possible pour apprendre à résoudre un problème d'apprentissage automatique lorsqu'on ne dispose pas de données annotées d'entrainement dès le départ.

**Approches existantes**
* **Supervision faible** : si on dispose de données non annotées, on peut annoter des cas faciles automatiquement et rapidement en définissant un algorithme à base de règles ou d'heuristiques. Malgré les quelques erreurs d'annotation résultantes, on peut utiliser ces données annotées pour entrainer un modèle de machine learning. Les règles sont faciles à écrire et apportent une haute précision mais sont rigides. par contre le machine learning est généralisable aux divers variantes et probabiliste (gestion des incertitudes et de l'aléatoire) mais nécessite des données annotées.
* **Apprentissage actif** : l'annotation des données d'entrainement peut-être plus précise en sélectionnant les éléments les plus pertinents (???) et en les passant à un oracle (humain de préférence) pour annotation manuelle ou validation d'annotation automatique. La sélection des documents pertinents peut se faire par l'**échantillonnage contradictoire**  ou par l'**échantillonnage de coude** (2x plus rapide)
* **Apprentissage par transfert** :
  * la technique simple consiste à utiliser des modèles de vecteurs de mots préentrainés (transfer 1.0)
  * il est possible d'affiner le modèle de langue sur des données du domaine cible (transfer 2.0)

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
  * dataless intent recognition/2019 Knowledge-Based Dataless Text Categorization
  * dataless intent recognition/2019 Knowledge-Based Short Text Categorization Using Entityand Category Embedding
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



## dataless annotation/data augmentation

### 2020 When does data augmentation help generalization in NLP [EN COURS][PRIORITAIRE]
**Problème**
* les réseaux de neurones apprennent des "faibles" features
* l'augmentation de données d'entrainement encourage les algos à préférer les features "fortes"
* Dans quelles conditions l'augmentation des données d'entrainement évite l'utilisation des features faibles?
  * Combien de contreexemples doivent être vus pour éviter une faible feature donnée?
  * Est-ce que ce nombre croit avec la taille originale des données d'entrainement ?
* Dans quelles conditions l'augmentation des données d'entrainement n'encourage pas l'utilisation des features plus forts?
  * Est-ce que la relative difficulté de représenter une feature impact son adoption par le modèle?
  * Comment l'effectivité de l'augmentation des données change dans les config qui contiennent plusieurs faibles features main une seule feature forte?
* ces problèmes sont relatives à la perturbation de la distribution de l'entrainement, la généralisation interdomaine, les données d'entrainement non-iid (La plupart des méthodes de conception de classificateurs supposent que les échantillons d'apprentissage sont tirés indépendamment et de manière identique d'une distribution de génération de données inconnue, bien que cette hypothèse soit violée dans plusieurs problèmes de la vie réelle.[Dundar et al., IJCAI07, Learning Classifiers When The Training Data Is Not IID])


### 2019 EDA - Easy Data Augment Tech for Boosting Perf on Text Classif [En cours][PRIORITAIRE]8
**Problème**: insufisance des données annotées pour l'entrainement d'un classifieur de textes

**Solution**: générer "facilement" de nouvelles données annotées à l'aide de 4 opérations:
* remplacement par synonyme : remplacer n mots (non stopword) choisis aléatoirement par leur synonyme
* insertion aléatoire : n fois, remplacer un mot (non stopword) aléatoirement choisi par un synonyme
* permutation aléatoire : n fois, permuter la position de 2 mots choisis aléatoirement
* suppression aléatoire : supprimer aléatoirement chaque mot dans la phrase

**Code**: https://github.com/jasonwei20/eda_nlp

**voir aussi**
* https://towardsdatascience.com/text-data-augmentation-makes-your-model-stronger-7232bd23704
* https://medium.com/opla/text-augmentation-for-machine-learning-tasks-how-to-grow-your-text-dataset-for-classification-38a9a207f88d
* https://towardsdatascience.com/data-augmentation-for-text-data-obtain-more-data-faster-525f7957acc9
* https://towardsdatascience.com/text-classification-with-extremely-small-datasets-333d322caee2?gi=16ddf2ae2aa6


## dataless annotation/Weak supervision

### 2019 A clinical text classification paradigm using weak supervision and deep representation [FINI]
L'article présente une méthode simple pour adresser l'absence de données annotées pour une tâche de classification de textes de rapports cliniques. Il propose de construire une base d'entraînement à l'aide d'une méthode à base de REGEX. Le principe est de conclure en l'appartenance d'un document à une catégorie sir une phrase de ce doc comprend un mot-clé ou une combinaison de mots-clés prédéfinis pour cette catégorie (e.g. **uses tobacco** pour la classe **smoker**). Le jeu d'entraînement est donc potentiellement bruité et ne couvre que les motifs de mots-clés prédéfinis. Pour être robuste aux mots-clés inconnus, le modèle vectoriel emploi les embeddings de mots qui rapprochent les mots inconnus des connus. En effet, le vecteur d'un texte est la moyenne des embeddings des mots qu'il comprend (occurrence ou **type** (le doc est défini comme un ensemble de mots)?). Les expérimentations montrent de très bonnes performances en classification binaire avec le CNN (F1 à 0.92 & 0.97  pour 0.91 & 0.93 pour les règles) avec la quantité de données d'entraînement disponible (31861  et 22471) mais moins bonnes (0.77 pour 0.88 pour les règles) en multi-classe (5 classes) pour deux raisons : (1) faible quantité de données (389), (2) un important déséquilibre du jeu annoté (deux classes couvrant seuelement 5%).
* **Atout** :
  * disponibilité de données non annotées
  * disponibilité de l'expertise pour prédéfinir les motifs de mots-clés pour les règles
* **Manque**:
  * quantité suffisante de textes à annoter pour le deep learning (CNN)
*  **Avantage**:
  * l'augmentation de données annotées améliorera les performances de classification


## embeddings

### 2014 GloVe - Global Vectors for Word Representation

**voir aussi**
* homepage https://nlp.stanford.edu/projects/glove/
* code original en C/C++ : https://github.com/stanfordnlp/GloVe
* *Possible implémentation fonctionnant sur windows:
    * les 2 derniers commentaires de golslan : https://groups.google.com/forum/#!topic/globalvectors/rqPmTBqFbCQ
    * code py2 : https://github.com/hans/glove.py
    * code py3 : https://github.com/maierhofert/glove.py
* http://www.foldl.me/2014/glove-python/
* code python 3 : https://github.com/maierhofert/glove.py.git
* explication : https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010

### 2017 SIF-a_simple_but_tough_to_beat_baseline_for_sentence_embeddings [EN COURS]

**Problème** : comment obtenir des vecteurs qui captent suffisamment la sémantique des phrases à partir de vecteurs de leurs mots?

**Existant**
* Approches non supervisées :
  * moyenne pondérée (tfidf-GloVe) ou pas (avg-Glove) de vecteur de mots
  * skip-thought vectors : basé sur le modèle encodeur-décodeur avec un encodeur pour l'entrée (phrase) et deux décodeurs pour prédire respectivement la phrase précédente et la suivante.
* Approches semi supervisées :
  * avg-PSL : vecteurs de mots PARAGRAMSL999 appris par supervision, puis moyenne simple
* Approches supervisées :
  * PP, PP-proj: moyenne des vecteurs de mots PARAGRAMSL999 avec ou sans projection linéaire.
  * DAN : réseau profond moyennant
  * RNN, iRNN : réseau de neurones récurrent sans ou avec l'activation comme identité (??)
  * LSTM, LSTM : avec (LSTM(o.g.)) ou sans (LSTM (no)) portes de sorties

**Hypothèses**
* modèle génératif à variable latente
* [Ancien modèle] processus dynamique de génération de corpus :
  * génération du t ème mot à l'étape t
  * processus guidé par une marche aléatoire de variables latentes: vecteur de discours c_t \in R^d, vecteurs des mots w (v_w) dans R^d
  * le produit scalaire de c_t et v_w capte les correlations entre le discours et le mot w
  * proba d'observer w à l'étape t : P(w émis à t | c_t) ~= exp(<c_t, v_w>)
  * marche aléatoire lente de c_t : c_t+1 obtenu de c_t en ajoutant un petit vecteur de déplacement
  * c_t peut être estimé (c_s) pour le discours (phrase) : MAP estimate = moyenne pondérée des vecteurs de mots dans la phrase
* [Amélioration de la marche aléatoire] **lissage de l'estimation de c_t**: meilleur terme de pondération avec 2 types de "terme de lissage" qui doivent tenir compte du fait que certains mots apparaissent en dehors du contexte, et que certains mots courant apparaissent indépendamment du discours ("the", "and", etc.):
  * introduction de \alpha*p(w), où p(w) est la proba d'un mot (unigram) dans tout le corpus, \alpha est un scalaire: pour permettre au mot d'apparaitre même si sont produit avec c_s est très faible (mot hors contexte)  
  * introduction d'un vecteur de discours commun c_0 \in R^d : terme de correction pour les mots les plus fréquents (souvent lié à la syntaxe)
* Ainsi P(w émis à t | c_t) ~= \alpha*p(w) + (1-\alpha)exp(<~c_s, v_w>)/Z_{~c_s} où ~c_s = \beta*c_0+(1-\beta)*c_s, c_0 et c_s étant orthogonaux, \alpha et \beta sont des hyperparamètres, Z_{~c_s} = \sum_{w \in V} exp(<~c_s, v_w>)
  * si \alpha = \beta = 0, on retombe sur le modèle précédent
  * si \alpha > 0, tout mot peut apparaitre dans la phrase (grâce à p(w))
  * si \beta > 0, les mots corrélés avec c_0 peuvent apparaitre

**Algorithme de la méthode SIF** (*smooth inverse frequency*)
* calcul de la moyenne pondérée des vecteurs de mots w dans la phrase ; le poids étant a/(a+p(w)) avec a=paramètre (usually set to 1e-3) et p(w) fréquence de w
* retrait des projections des vecteurs moyens sur leur 1er vecteur singulier (élimination du composant commun) [re-pondération pour éviter de très grandes différences inutiles d'échelles entre les composantes dûe à la trop grande fréquence de certains mots]

**Questions:**
* A quel point le SIF est sensible à la valeur de a, p(w), et aux word embedding?
* comment employer le coefficient de correlation de pearson pour estimer le bon a?
* comment employer le coefficient de correlation de pearson pour evaluer une tâche de similarité comme dans l'aricle ( Pearson’s r × 100 )?
* **Est-ce important d'éliminer les stop-words ?**
  * [Réponse sur BERT (de l'auteur)] l'élimination des stopwords fait améliore les performances de 10% https://github.com/huggingface/transformers/issues/876#issuecomment-523228498
* **Si oui quand doit-on éliminer les stopwords : avant l'entrainement de vecteurs de mots ou après i.e. au moment l'agrégation ?**
* **Est-ce qu'un entrainement préalable des vecteurs de mots sur des textes du domaine cible (retail, finance, public service, health) peut améliorer encore plus les résultats ?**

**Voir aussi**
  * https://www.offconvex.org/2018/06/17/textembeddings/
  * https://blog.dataiku.com/how-deep-does-your-sentence-embedding-model-need-to-be
  * une implémentation https://www.kaggle.com/procode/sif-embeddings-got-69-accuracy
  * critique des fondements théoriques: https://www.groundai.com/project/a-critique-of-the-smooth-inverse-frequency-sentence-embeddings/1

### 2018-sent2vec
* méthode non supervisée d'apprentissage de la représentation vectorielle des textes: sorte d'extension de la fonction objectif de C-BOW mais pour entrainer les vecteurs de phrases
* forme générale : min_{U,V} \sum_{S \in C} f_S(UV i_S)
  * U \in R^kxh, et V \in R^hx|Vocab| : matrices des paramètres
  * les colonnes de V collecte les vecteurs de mots de dimension h
  * le vecteur indicateur i_S \in {0, 1}^|vocab| est un vecteur binaire encodant S (S est la fenêtre de contexte)
  * k = |vocab|
* le principe est :
  * d'apprendre les embeddings source v_w et destination u_w pour chaque mot w.
  * l'embedding de la phrase est défini comme la moyenne des embeddings source de ces mots constituants
  * le modèle est augmenté en apprenant les embeddings source non seulement pour les unigrams mais aussi pour les n-grams présent dans chaque phrase :
  v_S = (1/|R(S)|) V i_{R(S)} = (1/|R(S)|) \sum_{w \in R(S)} v_w
  * R(S) étant la liste des n-grams présents dans la phrase S
  * afin de prédire le mot manquant dans le contexte, la fonction objectif modélise la sortie softmax approchée par échantillonnage négatif (améliore le temps d'apprentissage même pour un grand nombre de mots)
* la fonction objectif d'entrainement de sent2vec est : min_{U,V} \sum_{S \in C} \sum_{w_t \in S} (l(u_{w_t}.T v_{S\{w_t}}) + \sum_{w'\in N_{w_t}}l(-u_{w'}.T v_{S\{w_t}}))
  * S phrase courrante
  * l : x-> log(1+e^{-x})
  * N_{w_t} ensemble des mots négativement échantillonnés pour le mot w_t de S (en suivant une distribution multinomiale où chaque mot est associé à la proba q_n(w) = \sqrt{f_w} / (\sum_{w_i \in vocab } \sqrt{f_{w_i}} avec f_w la fréq normalisée de w dans le corpus  
  * pour sélectionner les possible unigrams destinations (positifs), on utilise les sous-échantillonnages, chaque mot étant écarté avec la proba 1-q_p(w) où q_p(w) = min{1, \sqrt{t/f_w} + t/f_w}, où t est l'hyper-paramètre de sous-échantillonnage.
  * le sous échantillonnage évite que les mots très fréquents n'aient trop d'influence au cours de l'apprentissage pour ne pas introduire des biais dans la tâche de prédiction
  * la fonction objectif devient : min_{U,V} \sum_{S \in C} \sum_{w_t \in S} q_p(w)(l(u_{w_t}.T v_{S\{w_t}}) + |N_{w_t}|\sum_{w'\in N_{w_t}}q_n(w')l(-u_{w'}.T v_{S\{w_t}}))

**voir aussi**
* https://rare-technologies.com/sent2vec-an-unsupervised-approach-towards-learning-sentence-embeddings/

### 2018-ImprovingLanguageUnderstandingByGenerativePreTraining-RadfordNarasimhanSalimansSutskever-GPT
**Contexte** : Compréhension du langage naturel (NLU), pré-trainement de modèle de langue, réseau de neurones profond, GPT, apprentissage semi-supervisé pour le NLP/NLU, implication textuelle, réponse aux questions, évaluation de la similitude sémantique, et classification de documents

**Problème** : rareté des données annotées pour le NLU

**Hypothèses** : Une amélioration considérable des solutions à ces tâches peut être obtenue par **pré-entrainement génératif** d'un modèle de langue sur un corpus diversifié de textes non-annotés, **suivi d'un affinement discriminatif** sur la tâche spécifique

**Existant**

* Difficile d'exploiter des informations au-delà du niveau des mots:
  1. non clareté du type d'optimisation de fonction objectif pour apprendre les représentations des textes util au transfer de façon optimale
  2. aucun consensu sur la manière la plus efficace de transférer ces représentations apprises vers la tâche cible

* apprentissage semi-supervisé de modèle de langue (LM): ajout d'une partie non supervisée à la fonction objectif d'une approche supervisée pour exploiter une grande qté de données non-annotées dans le but d'améliorer les performance grâce à une meilleur représentation des textes. OU BIEN pré-entrainement de la représentation des textes sur des données non-annotées avant de les utiliser dans un contexte supervisé
  * l'approche proposée combine à la fois une partie non-supervisée à la partie supervisée, les 2 étant entrainés pendant l'apprentissage supervisé  avec une **légère modification de la partie non-supervisée**
* apprentissage non-supervisé de LM: apprentissage sur des données non-annotées des poids de neurone du réseau supervisé pour une tâche spécifique ==> recherche non-supervisé des valeurs d'initialisation
  * l'approche proposée a une partie non supervisée pré-entraînée sur des données non-annotées
* apprentissage de LM auxiliaire: ajout de fonctions objectif auxiliaires (non supervisé par rapport à la tâche) par exemple des réseaux multi-tâches (LM, POS, NER, ...)

**Solution** : objectif == apprendre une representation universelle (Transformeurs : une mémoire plus structurée pour gérer les dépendances à long terme dans le texte (comparé au RNN)) qui se transfère avec une petite adaptation à un large panel de tâche (traitement des entrées comme une séq continue de tokens) = faible changement sur l'architecture du modèle pré-entrainé

**Question**
Comment utiliser cette approche sur de longs documents ? classifications de décisions de justice par exemple

**voir aussi**


### 2014 doc2vec

**Existant**
* BoW  (FAIBLESSE perte d'ordre entre les mots, pas de sémantique)
* BoNgram : (ATOUT ordre dans le contexte) (FAIBLESSE pas de sémantique)
* (Bengio et al., 2006) concaténation des vecteurs de mots d'un contexte pour entrainer un réseau de neuronnes à reconnaître le prochain mot
* moyenne pondérée de tous les mots du document (FAIBLESSE perte de l'ordre comme BoW)
* (Socher et al., 2011b) combinaison des mots dans un ordre donné par un arbre d'analyse de structure de phrase (parsing)  d'une phrase en utilisant des opérations matricielles (FAIBLESSE limitée aux phrases parce que le parsing est utilisé)

**Approche proposée : Praragraph Vector**
* ATOUT
  * applicable aux textes de toutes tailles,
  * ne nécessite pas d'affinement orienté tâche pour la fonction de pondération de mot,
  * ne dépend pas des arbres de structure de textes,
  * gain de 16% de taux d'erreur par rapport au SOTA pour la classification de sentiment
  * gain de 30 % par rapport au BoW pour la classification de texte

* PRINCIPE
  * MODELE : concaténation du vecteur du paragraphe avec ceux de plusieurs mots du paragraphe, pour prédire le mot suivant
  * APPRENTISSAGE DES VECTEURS : SGD + rétro-propagation,
  * PREDICTION : pour prédire le vecteur d'un nouveau paragraphe, l'intuition est que les paragraphes (leur vecteur) est unique, mais il partage les vecteurs de mots. le nouveau vecteur est inféré en fixant les vecteur de mots et en entraînant le nouveau vecteur de paragraphe jusqu'à la convergence

**voir aussi**
* CODE entraînement sur wikipedia : https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html
* https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5


### 2017 fasttext - Enriching Word Vectors with Subword Information

**voir aussi**
* https://medium.com/paper-club/bag-of-tricks-for-efficient-text-classification-818bc47e90f#:~:text=%20Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification,up%20of%20N%20ngram%20features%20in...%20More%20
* https://towardsdatascience.com/fasttext-ea9009dba0e8
* Utilisation pour la correction orthographique:
  * https://github.com/Bicky23/FastText-Spell-Checker
  * https://www.haptik.ai/tech/extract-spelling-mistakes-fasttext/

### 2020 P-SIF - Document Embeddings Using Partition Averaging [PRIORITAIRE]

### 2015 [GOOD PERF] AdaSent-SelfAdaptiveHierarchicalSentenceModel [PRIORITAIRE]
** existant**
* cBoW:
  * agrégation de vecteurs de mots : avg ou max
  * inconvénient: insensible à l'ordre entre les mots et aussi à la longueur de la phrase; par conséquent, il est possible pour 2 phrases de sens différents d'avoir la même représentation vectorielle

**Principe**
* représentation hiérarchique de phrase (mot-terme/expression-phrase) sous forme de flow simulant à la fois des recurrent nn et des recursive nn. chaque niveau est agrégé et la pyramide entière est réduite en une hiérarchie H; la hiérarchie est ensuite fournie à un gating network et à un classifier pour former un ensemble
* en recurrent nn, la dynamique consiste à transformer consécutivement les mots combiner au vecteur caché précédent; le dernier vecteur caché étant celui de la phrase (h_0 = 0  ; h_t = f(Wh_t^0+Hh_{t-1}+b). W est la matrice de connexion entrée-caché, H est la matrice de connexion récurrente caché-caché
* en recursive nn, l'idée est de composer suivant un arbre binaire prédéfini dont les mots (leur vecteur) sont les feuilles. Des transformations non-linéaires sont récursivement appliqués du bas vers le haut pour générer la repr caché d'un noeud parent à partir de la repr de ses 2 fils (h = f(W_L h_l + W_R h_r + b). W_L et W_R sont les matrices de connexion recursive cauche et droite, h_l et h_r sont les repr cachées des fils gauche et droit.

**Différence avec grConv (réseau de neuronne récursif à porte**
* AdaSent forme une hiérarchie d'abstractions de la phrase d'entrée
* AdaSent nourrit la hiérarchie comme un résumé dans le classifieur suivant
* combiné à un réseau de portes pour décider du poids de chaque niveau dans le consensus final.

**Structure**
* graphe acyclique orienté
* structure pyramidale de T niveaux (de 1 à T du bas vers le haut) pour une entrée (phrase) de longueur T
* la portée de chaque unité due niveau t=1 est le mot correspondant i.e. scope(h_j^1) = {x_j} \forall j \ in 1:T
* \forall t>=2 scope(h_j^t) = scope(h_j^{t-1}) \cup scope(h_{j+1}^{t-1}) = {x_{j:j+t-1}}
* le niveau t contient T-t+1 unités, et chaque unité a une porté de taille t*
* l'unité h_j^t peut être interprété comme le résumé de l'expression x_{j:j+t-1} dans la phrase originale.
* le niveau 1 comprend les vecteurs de mots
* le niveau T est le résumé global de la phrase entière
* **pretraitement** transformation linéaire des vecteurs de mots de R^d à R^D (D >= d): la représentation cachée au niveau 1 est h_{1:T}^1 = U'h_{1:T}^0 = U'Ux_{1:T}, où U' \in R^{Dxd} est la matrice de transformation linéaire dans AdaSent et U \in R^{dxV} est la matrice d'embeddings de mots entrainés sur un large corpus non-annoté: **cette factorisation aide à réduire le nombre de paramètre du model lorsque d << D**

**Composition locale et niveau de pooling (mise en commun)**
* la composition locale récursive : h_j^t = w_l h_j^{t-1} + w_r h_{j+1}^{t-1} + w_c h~_j^t, avec  h~_j^t = f(W_L h_j^{t-1} + W_R h_{j+1}^{t-1} + b_W) avec
  * j \in 1:T-t+1,
  * t \in 2:T,
  * W_L et W_R \in R^{DxD} sont les matrices de combinaison caché-caché, matrices récurrente doublées,
  * b_W \in R^D est le vecteur biais
  * w_l, w_r, et w_c sont les coefficients de porte s.c. w_l, w_r, w_c >= 0 et w_l + w_r + w_c = 1
* h_j^t, w_l, w_r, et w_c sont des fonctions paramétrées de h_j^{t-1} et h_{j+1}^{t-1} de telle sorte qu'ils peuvent décider soit de composer ces enfants par une transfo non-linéaire ou simplement transmettre leur représentation pour de futures compositions.
* une fois la pyramide construite, on applique une agrégation de moyenne ou de max sur le niveau t pour obtenir un résumé \bar{h}^t de toutes les expressions consécutives de taille t dans la phrase originale

**réseau de porte**
* l'approche peut-être étendu pour résoudre un pb de classification
  * soit g() un classifieur discriminatif qui prend \bar{h}^t \ in R^D en entrée et retourne les proba des différentes classes.
  * w() est le réseau de porte qui prend \bar{h}^t en entrée et retourne un score de confiance 0<=\gama_t<=1
  * intuitivement \gama_t dépeind la confiance qu'il y a dans le fait que le niveau t de résumé dans la hiérarchie, est adéquat pour être utilisé comme une repr adéquate pour l'instance d'entrée courante pour la tâche en main.
  * on exige que  \gama_t >= 0 et \sum_{t=1}^T \gama_t = 1
* soit C la variable aléatoire correspondant au label de la classe,
* le consensus du système entier est atteint en prenant un mélange des décisions faites par les niveaux de résumé de la hiérarchie:
  * p(C = c | x_{1:T}) = \sum_{t=1}^T p(C=c|H_x=t) p(H_x = t|x_{1:T}) = \sum_{t=1}^T h(\bar{h}^t) w(\bar{h}^t)

**retro-propagation à travers la structure (BPTS)**
* la BPTS est utilisé pour calculer les dérivées partielles (les matrices W_L, W_R, G_L, G_R) de la fonction objectif L() par rapport au paramètres du model
* dL/dW_L = \sum_{t=1}^T \sum_{j=1}^{T-t+1} dL/dh_j^t dh_j^t/dW_L
* dL/dW_R = \sum_{t=1}^T \sum_{j=1}^{T-t+1} dL/dh_j^t dh_j^t/dW_R
* grâce à la structure DAG : dL/dh_j^t = dL/dh_j^{t+1} dh_j^{t+1}/dh_j^t + dL/dh_{j-1}^{t+1} dh_{j-1}^{t+1}/dh_j^t
* les formulations locales BP: dh_{j-1}^{t+1}/dh_j^t = w_r I + w_c diag(f')W_R et dh_{j}^{t+1}/dh_j^t = w_l I + w_c diag(f')W_L
  * I matrice identité
  * diag(f') matrice diagonale couverte par le vecteur f' qui est la dérivée de f pr rapport à son entrée

**Voir aussi**
* Implémentation de l'auteur: https://hanzhaoml.github.io/papers/IJCAI2015/adasent.zip
* Implémentation : https://github.com/AllenCX/Adasent-pytorch [seems the best online]
* Implémentation : https://github.com/Mooonside/AdaSent [seems to have the best io]

### 2019 Bert - Pre-Training of Deep Bidirectional Transformers for Language understanding


### 2020 CamemBERT - a Tasty French Language Model
**voir aussi**
* https://blog.baamtu.com/word2vec-camembert-use-embedding-models/

### 2020 SBERT-WK - A Sentence Embedding Method By Dissecting BERT-based Word Models [PRIORITAIRE]

### 2019 Sentence-BERT [PRIORITAIRE]

### 2020 Improving Sentence Representations via Component Focusing (CF-BERT)
**existant**
* moyenne des vecteurs de mots : methode la plus facile et plus populaire
* LSTM : meilleur perf, mais complexe à mettre en oeuvre ;
* RNN: obtienne l'info global par recursion graduelle;
* CNN: n'obtient que l'info local; utilisation de filtres convolutionnels pour capter les dépendances locales + application de la couche d'agrégation pour extraire les features globales
* Transformers: obtiennent directement l'info global, sont plus rapide car exécution en parallèle
* BERT : utilise des transformer mais aucun embedding indépendant de phrase n'est calculé, (difficile de dériver un vecteur de phrase de BERT);
* SIF embedding : somme des embeddings de mots pondérés par l'idf et soustrit à un vecteur basé sur les composantes principale des vecteurs de phrases
* Sent2Vec apprend les features de  n-grams dans la phrase pour prédire le mot central à partir du contexte environnant
* Skip-thought : un encoder neural séquentiel de phrase, entraine une archi encodeur-décodeur qui peut prédire les phrases environnantes
* InferSent : un encoder neural séquentiel de phrase, utilise des données labélisées des datasets SNLI et Multi-Genre NLI pour entrainer un réseau BiLSTM siamois avec l'agrégation par max sur la sortie pour générer l'encodeur de phrase.
* USE universal Sentence Encoder : entraine un réseau de transformeur et augmente l'apprentissage non-supervisé initialement réalisé sur un corpus non annonté comme wikipédia puis continué sur SNLI
* Sentence-BERT (SBERT) : utilise une structure de réseau siamois pour affiner le réseau pré-entrainé BERT premièrement par NLI puis spécifiquement par les tâches spécifique de NLP pour déduire des repr sémantiquement significative de phrases.

**proposition CF-BERT**
* modification du réseau pré-entrainé BERT
* utilise une structure de réseau siamois pour dériver des vecteurs sémantiques significatifs par focalisation sur les composants
* divise la représentation d'une phrase en 2 parties:
  * **partie de base** correspond à la phrase complète (positon dominante)
  * la **partie améliorée par composant** qui tient compte de l'info pertinente et réduit l'impact des mots nuisibles sur le sens de la phrase (rôle de supplément)
* Pour la partie améliorée, un arbre de dépendance grammaticale est utilisé
* un facteur de poids est déterminé par grid search pour générer la représentation optimale de la phrase
* le vecteur final est obtenu par une stratégie d'agrégation
* CF-BERT est du SBERT si le facteur poids de la partie focalisée composant est nul (W_{cf}=0) : emb_S = emb_{S_{cf}} * W_{cf} + emb_{S_{basic}}
* la couche de sortie peut être définir en fonction de la tâche spécifique à résoudre

### 2019 spherical-text-embedding
**existant**
* word2vec : apprend les repr de mots en espace euclidian en modélisant les co-occurrences locales entre mots
* gap entre l'apprentissage en espace euclidian et l'usage en espace sphérique (normalisation préalable des vect tfidf, similarité cosinus)


**proposition**
* apprendre directement la repr de texte en espace sphérique ne imposant des contraintes de norme sur les repr i.e. définition d'un modèle génératif à 2 étapes à la surface d'une sphère unitaire
  * un mot est généré en premier conformement à la sémantique du paragraphe
  * ensuite, les mots environnant sont générés en consistance evec la sémantique du mot central
* l'apprentissage est défini par une procédure d'optimisation efficiente de Rieman
* Autre avantage: apprentissage joint de repr de mots et de repr de paragraphes; la repr des paragraphes peut être directement obtenue pendant l'entraînement grace à la modélisation explicite de la relation générative entre mots et leur paragraphe
  * meilleur repr de mots par exploitation jointe des stats de co-occurrence mot-mot et mot-paragraphe
* rapidité dû au remplacement de la couche conventionnelle softmax par la fontion sphérique de perte


**voir aussi**
* Code de l'auteur: https://github.com/yumeng5/Spherical-Text-Embedding
* slides: https://yumeng5.github.io/files/kdd20-tutorial/Part1.pdf


## knowledge graph

### 2017 Graph-based_Text_Representations_Tutorial_EMNLP_2017 [TURORIEL][FINI][A RESUMER]

**Objectifs**: Booster la fouille de textes, le TALN, et la RI avec les graphes

**Problèmes de la représentation par BoW**: hypothèse d'indépendance entre termes et pondération par fréquence de termes

**Intérêt de la représentation de texte par graphe**: capturer la **dépendance** entre les termes, de leur **ordre** et la **distance** entre eux.


## query analysis

### 2018_Chapter_UnderstandingInformationNeeds [SUSPENDU]

en RI, la compréhension de l'information nécessitée par l'utilisateur passe par une bonne compréhension de sa requête. Pour cela, il existent des techniques comme la classification de la requête suivant des buts de niveau élevé (intention), segmentation en parties allant ensemble (e.g. noms composés), interpréter la structure de la requête, reconaître et désambiguïser les entités mentionnées, déterminer si un service spécifique ou un segment/domaine des contenus en ligne (*verticals*, e.g. shopping, voyage, recherche d'emploi, etc.) doit être invoqué.

#### Analyse sémantique de requête

**Classification de requête**

**Annotation de requête**

**Interprétation de requête**


## short texts similarity

### 2019 Sentence Similarity Techniques for Short vs Variable Length Textusing Word Embeddings [FINI]*
**Problème**: Comment estimer la similarité sémantique entre une phrase S1 courte (1-3 mots e.g. des commandes courtes comme "*supprimer la commande*" ou "*montrer les éléments récents*") et une autre phrase S2 plus grande ?

**Contexte**: agent conversationnel (e.g. chatbot)

**Existant**:
* le cosinus de similarité des sac-de-mots ne tient pas compte de l'ordre des mots dans la phrase
* la somme ou moyenne (pondérée ou pas) des vecteurs de mots :
  * inconvénient: le cosinus est invariant entre la somme et la moyenne
  * inconvénient: la moyenne pondérée n'aide pas pour une longue phrase
  * avantage: la somme et  la moyenne sont moins chères à calculer
* problème dans le dev de chatbot:
  * nécessité d'un grand nombre de données annotées d'entrainement
  * gestion perpétuelle de plusieurs données à chaque prédiction : complexe et chèr en temps

**Proposition1: fenêtre coulissante avec moyenne pondérée des vecteurs de mots**
* taille_fenetre = nombre de mot de S1 (la phrase la plus courte)
* S2 est découpé en sous chaines {S2_j} de taille taille_fenetre
* le vecteur de chaque S2_i est la moyenne des vecteurs de ses mots
* sim(S1, S2) = max_j cos_sim(vecteur_S1, vecteur_S2_j)

**Proposition2: vecteurs pondérés de n-grams**
* N-gram = sous-chaines de N mots consécutifs (e.g. unigram=1-gram, bigram=2-gram, trigram=3-gram)
* les phrases Si (i \in 1:2) sont découpées en N_{i1} unigrams {Si_1-gram_j}, N_{i2} bigrams {Si_2-gram_j}, N_{i3} trigrams {Si_3-gram_j}, ..., Nmax-gram {{Si_Nmax-gram_j}}
* le vecteur de chaque N-gram est la moyenne des vecteurs de ses mots
* \forall k in 1:Nmax, score_k = 1/N_{1k} \sum_{n1 \in 1:N_{1k}} max_{n2 \in 1:N_{2k}} {cos_sim(vecteur_{S1_k-gram_j}, vecteur_{S2_k-gram_j})}
* sim(S1, S2) = \sum_{k \in 1:Nmax} w_k * score_k, avec w_k = k / {\sum_{k \in 1:Nmax}}

**discussion des résultats**
* LIMITE: **dataset conçu par les auteurs** : une commande courte de base (S1) et diverses façons (généralement un peu plus longue) d'exprimer cette commande (S2)
* le seuil minimal de similarité est fixé à 0.9
* proposition1 beaucoup moins bonne que la proposition 2 (F1-score de 0.3708 contre 0.9316, et 0.1451 pour le cos_sim Google's Universal Sentence Encoding)

## short texts similarity / metric learning

### 2019 ATutorialonDistanceMetricLearning-MathematicalFoundationsAlgorithmsandExperiments [EN COURS]

*  Une distance standard peut ignorer des propriétés importantes dans le dataset = son utilisation par un apprentissage rendant ce dernier non optimal
*  L'objectif de l'apprentissage d'une distance, c'est de rapprocher autant que possible les objets similaires, tout en éloignant les différents, pour améliorer la qualité des applications
*  Les bases de l'apprentissage de distance sont :
  *  **l'analyse convexe** : pour la présentation et la résolution de pbs d'optimisation (estimation de paramètres)
  *  **l'analyse matricielle** : pour la compréhension de la discipline, la paramétrisation des algo, et l'optimisation par les vecteurs propres
  *  **la théorie de l'information** : qui a motivé plusieurs des algorithmes

### 2019 Metric Learning for Dynamic Text Classification [EN COURS]
**problème** : en classif de textes, des labels peuvent parfois être ajoutés ou supprimés dans le temps => **classif dynamique**

**solution proposée** : remplacer la couche de sortie de classif des réseaux de neurones par un **espace métrique appris à moindre coût et sématiquement significatif** pour améliorer les performances du kNN

**context**:
* pb des classifs traditionnelles:
  * sortie de taille fixe == nb de labels i.e ajout / suppr de label ==> changer l'architecture du model
  * impossible de réutiliser les paramètres déjà appris ==> pb car les nvls classes n'ont souvent que très peu de données
  * pas d'exploitation de l'info entre label ==> affibli l'adaptation aux nouveaux labels

## misc-suggested

### 2020-LuyuGao-ModularizedTransfomerBasedRankingFramework

**voir aussi**:
* présentation vidéo : https://slideslive.com/38939359
* code TBA : https://github.com/luyug/MORES


### 2020-MujeenSung-BiomedicalEntityReprWithSynonymMarginalization


## speech recognition

### 2020-MLSALargeScaleMultilingualDatasetForSpeechResearch-VineelPratap

**voir aussi** :
* wav2vec : https://github.com/pytorch/fairseq/tree/master/examples/wav2vec
* ASR avec wav2vec : https://github.com/facebookresearch/flashlight/tree/master/flashlight/app/asr
* [Massively Multilingual ASR: 50 Languages, 1 Model, 1 Billion Parameters](https://arxiv.org/pdf/2007.03001.pdf)

## TEMPLATE
### PDF FILE NAME
**Contexte** :

**Problème** :

**Hypothèses** :

**Existant**

**Solution**

**Question**

**voir aussi**
