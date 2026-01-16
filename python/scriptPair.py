import random # module à importer pour manipuler des variables aléatoires

x = random.randint(0,100) # génère un entier dans [0,100[

if x%2==0 : # test si le reste de x/2 est égal à 0
     # bloc exécuté ssi la condition de boucle est vrai
     print('x=' + str(x))
     print('x est pair')
else : # alternative (optionnelle)
     # bloc exécuté ssi la condition de boucle est fausse
     print('x=' + str(x))
     print('x est impair')