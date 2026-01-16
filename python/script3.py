poisson1={} # dictionnaire vide
poisson1={"nom":"merouBrun","age":4,"taille":40}
print(poisson1["nom"])
print("---------")
for cle in poisson1:
    print(cle, poisson1[cle])
print("---------")
for cle,valeur in poisson1.items():
    print(cle,"\t: ", valeur)
list(poisson1.values())