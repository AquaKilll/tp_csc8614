Question 1 : Do you see any difference between "Original Model Structure (Truncated)" and "Model Structure After LoRA (Truncated)"? Do you see the LinearWithLoRA you have defined above?

Oui, la différence est très nette.

Dans la structure "Originale", les couches (comme W_query, W_key, ou celles du FeedForward) sont des couches Linear standards :
 - (W_query): Linear(in_features=768, out_features=768, bias=True)

Dans la structure "Après LoRA", ces mêmes couches ont été remplacées par notre wrapper LinearWithLoRA. À l'intérieur de ce wrapper, on distingue clairement deux branches :
- (linear) : La couche linéaire originale (qui est maintenant gelée).
- (lora) : La nouvelle couche LoRALayer que nous avons définie (qui contient les matrices entraînables A et B).

Cela confirme que l'injection dynamique des couches LoRA a fonctionné sur l'ensemble du bloc Transformer.

Question 2 : What is the number of trainable parameters, all parameters, and the fraction of trainable parameters?

Nombre de paramètres entraînables : 1 327 104
Nombre total de paramètres : 164 364 288
Fraction de paramètres entraînables : 0,81 %

Question 3: Check the number (and fraction) of trainable parameters, and compare it with the one above. Do you see any differences? Can you describe them?

| Métrique                | Question 2 (LoRA seul) | Question 3 (LoRA + Classification Head) | Différence |
| :---------------------- | :--------------------: | :-------------------------------------: | ---------: |
| Paramètres entraînables |       1 327 104        |                1 328 642                |    + 1 538 |
| Total des paramètres    |        ~164,3 M        |                ~125,8 M                 |   - 38,6 M |
| Fraction entraînable    |         0,81 %         |                 1,06 %                  |   + 0,25 % |

Analyse des différences :

1. Pourquoi le Total a-t-il chuté drastiquement (~164M $\rightarrow$ ~125M) ?
C'est la différence la plus marquante. Nous avons remplacé la couche finale originale (out_head), qui servait à prédire le mot suivant parmi un vocabulaire de 50 257 mots ($768 \times 50257 \approx 38,6$ millions de paramètres), par une couche minuscule qui prédit seulement 2 classes ($768 \times 2 = 1536$ paramètres). Le modèle est donc devenu physiquement plus léger.

2. Pourquoi les paramètres entraînables ont-ils augmenté (+1538) ?Les paramètres entraînables sont maintenant constitués de :Les couches LoRA (matrices A et B) qui étaient déjà là ($1 327 104$).PLUS la nouvelle tête de classification (out_head) que nous venons d'ajouter et de débloquer.Le calcul est exact : $1 328 642 - 1 327 104 = 1538$. Cela correspond aux poids ($768 \times 2$) et aux biais ($2$) de la nouvelle couche linéaire ($1536 + 2 = 1538$).

3. Pourquoi le pourcentage a-t-il augmenté (0,81% $\rightarrow$ 1,06%) ?
C'est purement mathématique : le numérateur (trainable) a très peu changé, mais le dénominateur (total) a beaucoup diminué. La part des paramètres LoRA devient donc proportionnellement plus importante dans ce modèle "allégé" de sa tête de vocabulaire.

Question 4: Can you describe the trend of the loss, and the final accuracy. Is it reasonable considering the task at hand?

1. Tendance de la Loss (Perte) La courbe d'apprentissage est excellente et caractéristique d'un Transfer Learning efficace :

Chute rapide : La perte commence assez haut (~2.79) mais s'effondre très vite (0.15 dès le batch 10). Cela montre que le modèle s'adapte presque immédiatement à la tâche.

Stabilisation basse : La perte atteint des valeurs très faibles (proches de 0.001) sur de nombreux batchs.

Quelques pics : On observe quelques remontées sporadiques (ex: 0.64 au batch 60), ce qui est normal (bruit du gradient stochastique ou batchs plus difficiles), mais la moyenne reste très basse (0.1685).

2. Exactitude Finale (Accuracy) L'exactitude atteint 95,47 % en une seule époque. C'est un score très élevé.

3. Est-ce raisonnable ? Oui, c'est tout à fait raisonnable et cela démontre la puissance de la méthode LoRA couplée à un modèle pré-entraîné :

Efficacité du pré-entraînement : GPT-2 "comprend" déjà l'anglais. Il n'a pas besoin d'apprendre la syntaxe, juste de comprendre la différence entre le style "spam" et le style "normal".

Rapidité de convergence : Contrairement à un entraînement "from scratch" qui prendrait des heures pour atteindre ce score, LoRA permet d'obtenir un modèle performant en moins de 40 secondes et une seule époque.

Comparaison : Obtenir >95% sur de la détection de spam est standard pour des LLM modernes.

Conclusion : Le modèle a appris très vite et très bien. C'est un succès.

Question 5: How is the accuracy, and how does it compare to the Train set accuracy?

1. Niveau d'Exactitude (Accuracy) L'exactitude sur le jeu de test est excellente : 97,66 %. Cela confirme que le modèle est très performant pour distinguer les spams des messages normaux.

2. Comparaison avec le jeu d'Entraînement (Train set) C'est un résultat intéressant et positif : l'exactitude en test (97,66 %) est légèrement supérieure à celle observée à la fin de l'entraînement (95,47 %).

3. Pourquoi le test est-il meilleur que l'entraînement ? En général, on s'attend à l'inverse (overfitting), mais voir Test > Train est courant lors du fine-tuning pour deux raisons principales :

Le Dropout : Pendant l'entraînement, des neurones sont désactivés aléatoirement (dropout) pour forcer l'apprentissage, ce qui "bride" un peu le modèle. Pendant le test (model.eval()), le dropout est désactivé et le modèle utilise 100% de sa capacité.

Moyenne vs Instantané : L'accuracy d'entraînement (95,47%) est une moyenne sur toute l'époque (incluant les premiers batchs où le modèle était mauvais). L'accuracy de test est calculée avec le modèle final, déjà entraîné.

Conclusion : Le modèle généralise parfaitement et ne souffre pas de surapprentissage.