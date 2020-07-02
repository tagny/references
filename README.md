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
##### 2019_ESWC_KB_Short_Text_Classification_Using_Entity_and_Category_Embedding

##### 2019 A clinical text classification paradigm using weak supervision and deep representation
L'article présente une méthode simple pour adresser l'absence de données annotées pour une tâche de classification de textes de rapports cliniques. Il propose de construire une base d'entraînement à l'aide d'une méthode à base de REGEX. Le principe est de conclure en l'appartenance d'un document à une catégorie sir une phrase de ce doc comprend un mot-clé ou une combinaison de mots-clés prédéfinis pour cette catégorie (e.g. **uses tobacco** pour la classe **smoker**). Le jeu d'entraînement est donc potentiellement bruité et ne couvre que les motifs de mots-clés prédéfinis. Pour être robuste aux mots-clés inconnus, le modèle vectoriel emploi les embeddings de mots qui rapprochent les mots inconnus des connus. En effet, le vecteur d'un texte est la moyenne des embeddings des mots qu'il comprend (occurrence ou type?). Les expérimentations montrent de très bonnes performances en classification binaire avec le CNN (F1 à 0.92 & 0.97  pour 0.91 & 0.93 pour les règles) avec la quantité de données d'entraînement disponible (31861  et 22471) mais moins bonnes (0.77 pour 0.88 pour les règles) en multi-classe (5 classes) pour deux raisons : (1) faible quantité de données (389), (2) un important déséquilibre du jeu annoté (deux classes couvrant seuelement 5%).
* **Atout** : 
  * disponibilité de données non annotées
  * disponibilité de l'expertise pour prédéfinir les motifs de mots-clés pour les règles
* **Manque**:
  * quantité suffisante de textes à annoter pour le deep learning (CNN)

#### query analysis
* 2018_Chapter_UnderstandingInformationNeeds

## A lire en priorité



