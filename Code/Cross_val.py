import pandas as pd
import numpy as np 

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == "__main__": 

    # Importamos el dataset del 2017 
    dataset = pd.read_csv('./data/whr2017.csv')

    # En nuestros featuares vamos a utilizar todos nuestros datasets, sin la columna del pais que queremos
    # evaluar y sin la columna del score 
    X = dataset.drop(['country','score'], axis=1)
    # Ahora la columna que vamos a predecir nuestro target, es la coluna de score 
    y = dataset['score']

    # Definimos nuestro modelo 
    model = DecisionTreeRegressor()
    # En vez de usar la funcion fit, aquí de forma rapida mandamos un modelo, nuestros features y nuestros targets 
    # y definimos de forma opcional definimos el scoring que es el valor medio cuadrado  
    score = cross_val_score(model, X,y, scoring='neg_mean_squared_error')
    # Calculamos el valor absoluto de la media de este score, esta es la implementación más básica para 
    # implementar la validación cruzada 
    print(np.abs(np.mean(score)))  

    # Ahora si queremos ver como funciona cross validation de fondo, lo que tenemos que utilizar es la función kf 
    # que recibe como parametro, que recibe el numero de splits, si queremos que los datos se organicen aleatoriamente 
    # o simplemente en orden de estos splits y el parametros que nos ayuda a mantener la replicabilidad que es el random state
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # Ahora vamos a recorrer esto, para cada uno de nuestro elementos que bota el kf va a relizar la partición de train y de test
    for train, test in kf.split(dataset):
        print(train)
        print(test)
    # Los arreglos que aparecen contiene los indices de nuestro dataset según la partición que el realizó 

    





