Question 1 : Dans TP1/rapport.md, ajoutez immédiatement un court en-tête (quelques lignes) contenant : (i) votre nom/prénom, (ii) la commande d’installation/activation d’environnement utilisée, (iii) les versions (Python + bibliothèques principales).

(i) votre nom/prénom : Davoust Kilian 
(ii) la commande d’installation/activation d’environnement utilisée :

pip install -r requirements.txt

(iii) les versions (Python + bibliothèques principales) :

Python : 3.12.3
Torch : 2.9.1
TensorFlow : 2.20.0
Tiktoken : 0.12.0
Pandas : 2.3.3
Matplotlib : 3.10.8
JupyterLab : 4.5.1

Question 2 : What type is the object setting, and what is its structure (e.g. if it is a list, its length; if a dictionary, its keys, etc.)?

Type : Dictionnaire (<class 'dict'>).

Structure : Il contient 5 clés correspondant aux hyperparamètres architecturaux :
- n_vocab : 50257 (Taille du vocabulaire BPE)
- n_ctx : 1024 (Fenêtre de contexte maximale)
- n_embd : 768 (Dimension des vecteurs d'embedding)
- n_head : 12 (Nombre de têtes d'attention)
- n_layer : 12 (Nombre de blocs Transformer)

```python
# Analyse `settings`
# TODO: your code here.
print(f"Type: {type(settings)}")
print(f"Clés: {settings.keys()}")
print(f"Contenu: {settings}")

# Analyse `params`
# TODO: your code here.
print(f"Type: {type(params)}")
print(f"Clés principales: {params.keys()}")
```

Question 3 : What type is the object params, and what is its structure?

Type : Dictionnaire (<class 'dict'>).

Structure : Il contient les poids (matrices numpy) du modèle.

Clés de niveau supérieur :

- wpe : Poids des embeddings de position.
- wte : Poids des embeddings de tokens.
- blocks : Une liste contenant les paramètres des 12 couches cachées (Attention + MLP).
- g, b : Paramètres (gain et biais) de la normalisation finale (LayerNorm).

Question 4: Analyse the __init__ method, and check what is the required structure for the cfg parameter. Is the settings variable we have obtained in the right format? If not, perform the mapping to convert the variable setting into a variable model_config with the right structure.

En inspectant la méthode __init__ de GPTModel dans gpt_utils.py, on constate que le dictionnaire cfg attend des clés spécifiques comme vocab_size, emb_dim, context_length, n_layers et n_heads. Cependant, l'objet settings récupéré d'OpenAI utilise une nomenclature différente (n_vocab, n_embd, n_ctx, etc.). Conclusion : La variable settings n'est pas directement compatible. Nous devons effectuer un mapping explicite pour créer un dictionnaire model_config valide.

```python
# Configure the model, mapping OpenAI specific keys to our model's keys (if needed)
model_config = {
    # TODO: add your code here, but keep the two lines below.
    "vocab_size": settings["n_vocab"],
    "context_length": settings["n_ctx"],
    "emb_dim": settings["n_embd"],
    "n_heads": settings["n_head"],
    "n_layers": settings["n_layer"],
    "drop_rate": 0.1,
    "qkv_bias": True,
}

model = GPTModel(model_config)

# Load the pre-trained weights
load_weights_into_gpt(model, params)
model.eval() 

print("GPT-2 Model Loaded and Configured successfully!")
```

Question 5.1 : In the cell above, why did we do df = df.sample(frac=1, random_state=123) when creating the train/test split?

L'instruction df.sample(frac=1, random_state=123) effectue un mélange aléatoire de l'intégralité du dataset avant la division train/test.

Pourquoi ?

Éviter les biais de tri : Les datasets bruts sont souvent triés par catégorie (ex: tous les "ham" d'abord, puis les "spam") ou par date. Si l'on découpait les 80% premiers sans mélanger, le jeu d'entraînement pourrait contenir uniquement une classe et le jeu de test l'autre, ce qui empêcherait le modèle d'apprendre correctement. Le mélange garantit que la distribution des classes est homogène dans les deux sous-ensembles.

Reproductibilité : Le paramètre random_state=123 fixe la graine aléatoire. Cela assure que le mélange est toujours identique à chaque exécution du code, permettant de comparer nos résultats futurs sur exactement les mêmes données d'entraînement et de test.

Question 5.2 : Analyse the datasets, what is the distribution of the two classes in the train set? Are they balanced or unbalanced? In case they are unbalanced, might this lead to issues for the fine-tuning of the model?

```python
# TODO: Your code here.

import pandas as pd
import matplotlib.pyplot as plt

class_counts = train_df['Label'].value_counts()

print("Class Counts:\n", class_counts)
```

Class Counts:
 Label
ham     3860
spam     597
Name: count, dtype: int64

En termes de pourcentage :

Ham : 86,6 %
Spam : 13,4 %

Le jeu de données est déséquilibré (unbalanced). La classe ham est largement majoritaire, représentant près de 6,5 fois le volume de la classe spam. Ce déséquilibre peut effectivement poser plusieurs problèmes lors de l'entraînement (fine-tuning) du modèle :
- Biais vers la classe majoritaire : Le modèle risque d'optimiser sa fonction de perte en prédisant systématiquement la classe dominante (ham). Il apprendra que parier sur "ham" est une stratégie statistiquement sûre, au détriment de l'apprentissage des caractéristiques subtiles des spams.
- Métriques trompeuses (Paradoxe de l'exactitude) : Un modèle qui prédit "ham" pour 100% des messages obtiendrait une exactitude (accuracy) de 86,6%. Cela semble être un bon score, alors que le modèle serait totalement inutile pour détecter les spams.
- Faible Rappel (Recall) pour la classe Spam : Le modèle risque d'avoir beaucoup de "Faux Négatifs" (des spams non détectés), ce qui est critique pour un filtre anti-spam.

Question 6 : Create the dataloaders for training and test.

```python
# TODO: add any imports which are needed

# Create the Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Instantiate the Dataset
train_dataset = SpamDataset("train.csv", tokenizer)
test_dataset = SpamDataset("test.csv", tokenizer)

# --- TODO: Create DataLoaders ---
# 1. Create a train_loader with batch_size=16 and shuffle=True
train_loader = # YOUR CODE HERE (hint, use class DataLoader from torch.utils.data)
# 2. Create a test_loader with batch_size=16 and shuffle=False
test_loader =  # YOUR CODE HERE
# Check your work
for input_batch, target_batch in train_loader:
    print("Input batch shape:", input_batch.shape) # Should be [16, 120] (unless you use batch_size != 16)
    print("Target batch shape:", target_batch.shape) # Should be [16]
    break

Résultats :
Input batch shape: torch.Size([16, 120])
Target batch shape: torch.Size([16])
```

Question 7 : Looking at the batch size and the training size, how many batches will you have in total? Please report the size of the subsampled training data, you reduce it due to performance constraints.

Question 8 :

```python
# Freeze the internal layers
for param in model.parameters():
    param.requires_grad = False

print(f"Original output head: {model.out_head}") # TODO: YOUR CODE HERE

num_classes = 2 # TODO: YOUR CODE HERE
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes) # TODO: YOUR CODE HERE
# Hint: The input size of the last layer in GPT-2 small is 768.

# Enable gradient calculation ONLY for the new head and the final LayerNorm
for param in model.out_head.parameters():
    param.requires_grad = True
for param in model.trf_blocks[-1].norm2.parameters():
    param.requires_grad = True

print(f"New output head: {model.out_head}") # TODO: YOUR CODE HERE
```

8.3: Why do we freeze the internal layers with param.requires_grad = False?

Nous définissons param.requires_grad = False (on "gèle" les poids) pour plusieurs raisons cruciales dans le cadre du Transfer Learning (apprentissage par transfert) :
- Préserver les connaissances : GPT-2 a été entraîné sur une immense quantité de texte et "comprend" déjà la structure de la langue anglaise. Nous voulons conserver cette capacité d'extraction de caractéristiques sans la détruire en modifiant les poids trop brutalement avec notre petit dataset.
- Éviter le surapprentissage (Overfitting) : Vous n'avez que 2000 exemples d'entraînement. Si vous essayez de réentraîner les 124 millions de paramètres du modèle complet, le modèle va apprendre par cœur ces 2000 exemples et sera incapable de généraliser. En ne réentraînant que la dernière couche (la "tête"), on limite considérablement ce risque.
- Performance et rapidité : Calculer les gradients pour tout le modèle est très coûteux en calcul et en mémoire. En ne mettant à jour que la dernière couche, l'entraînement est beaucoup plus rapide.

Question 9 :

```python
# TODO: Add your code where needed in this cell
import torch.nn.functional as F

num_epochs = 3  # TODO: (Optional questions: update this to see how the fine-tuning changes with more epochs)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # 9.1. Reset Gradients (of the `optimizer`)
        # TODO: YOUR CODE HERE
        optimizer.zero_grad()

        # Forward Pass. The model outputs (batch, seq_len, vocab_size). 
        # We only want the prediction for the LAST token in the sequence.
        logits = model(inputs)[:, -1, :]

        # 9.2. Calculate the cross entropy loss
        loss = F.cross_entropy(logits, targets, weight=class_weights) # TODO: YOUR CODE HERE
        
        # 9.3. Backward Pass
        # TODO: YOUR CODE HERE
        loss.backward()

        # 9.4 Optimizer Step
        # TODO: YOUR CODE HERE
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # 9.5 Add code to calculate the accuracy on train and test
    train_acc, train_spam_acc = calc_accuracy(train_loader, model, device) # TODO: YOUR CODE HERE
    test_acc, test_spam_acc = calc_accuracy(test_loader, model, device) # TODO: YOUR CODE HERE
    print(f"Epoch {epoch+1}: Train Acc: {train_acc*100:.2f}% (Spam: {train_spam_acc*100:.2f}%) | Test Acc: {test_acc*100:.2f}% (Spam: {test_spam_acc*100:.2f}%)")

Résultats :
Epoch 1: Train Acc: 14.83% (Spam: 100.00%) | Test Acc: 15.43% (Spam: 99.33%)
Epoch 2: Train Acc: 13.39% (Spam: 100.00%) | Test Acc: 13.45% (Spam: 100.00%)
Epoch 3: Train Acc: 33.32% (Spam: 99.50%) | Test Acc: 34.80% (Spam: 99.33%)
```

Question 10 : Now run the cell above. You should see how the training loss changes after each batch (and epoch). Describe thie trend: what do you see, is the model learning?

1. Tendance de la Loss (Perte) : Oui, le modèle apprend. On observe clairement que la loss diminue au fil du temps :
- Elle commence très haut (~5.07) au tout début.
- Elle descend rapidement autour de 1.0 - 2.0 à la fin de l'époque 1.
- Elle se stabilise sous la barre des 1.0 (ex: 0.5 - 0.8) durant l'époque 2 et 3. Cela indique que l'optimiseur fait son travail et minimise l'erreur mathématique.

1. Analyse de l'Accuracy (Exactitude) : Un cas intéressant. Cependant, l'évolution de l'exactitude révèle un comportement particulier dû à la pondération des classes que nous avons ajoutée :
- Spam Accuracy (~100%) : Le modèle détecte presque tous les spams dès le début. C'est excellent pour le rappel, mais...
- Global Accuracy (~15% -> 33%) : L'exactitude globale est très faible au début (15%), ce qui est proche du pourcentage de spams dans le dataset (13.4%).

Interprétation : À cause du fort poids donné à la classe "Spam" pour contrer le déséquilibre, le modèle a "sur-corrigé". Au début (Epoques 1 et 2), il prédit "Spam" presque tout le temps pour éviter la lourde pénalité.
- Il a 100% de bon sur les Spams.
- Il a presque tout faux sur les "Hams" (ce qui explique l'accuracy globale faible).

Conclusion : Le modèle apprend, mais il part d'un extrême (prédire tout "Ham") pour aller à l'autre extrême (prédire tout "Spam" à cause des poids). À l'époque 3, l'accuracy globale remonte à 33%, signe qu'il commence enfin à trouver un équilibre et à distinguer réellement les deux classes au lieu de deviner aveuglément. Il faudrait probablement plus d'époques pour que l'accuracy globale atteigne un niveau satisfaisant (ex: >90%).