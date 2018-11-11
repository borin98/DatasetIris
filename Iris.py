import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import plot_model, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def redeComKeras ( dadosEntrada, dadosSaida ) :

    aNN = KerasClassifier ( build_fn = montaRedeNeural )

    otimizador = {
        "batch_size" : [10, 50],
        "epochs" : [100, 200],
        "optimizer" : ["adam", "sgd", "adadelta"],
        "kernel_initializer": [ "random_uniform" ,"normal"],
        'activation': ['relu', 'linear'],
        'neurons': [3, 10],
        'dropout': [0.2, 0.50]
    }

    resultadoGrid = GridSearchCV(
        estimator = aNN,
        param_grid = otimizador,
        cv = 3
    )

    resultadoGrid = resultadoGrid.fit ( dadosEntrada, dadosSaida )

    print("Melhores parâmetros : {}".format ( resultadoGrid ) )
    print("\nMelhor precisão : {}".format ( resultadoGrid.best_score_ ) )

    #   Lembrando que a quantidade de vezes que a rede
    #   treina é da forma T = cv*epochs
    #   cv salva os melhores pesos para a próxima geração
    resultados = cross_val_score ( estimator = aNN,
                                   X = dadosEntrada,
                                   y = dadosSaida,
                                   cv = 10,
                                   scoring = "accuracy")

    media = resultados.mean (  )
    desvioPadrao = resultados.std (  )

    print("Média da rede neural : {}".format ( media ) )
    print("Desvio padrão da rede neural : {}".format ( desvioPadrao ) )

    return

def montaRedeNeural (optimizer, kernel_initializer, activation, neurons, dropout ) :
    """
    Função que monta a
    rede neural

    :return:
    """
    aNN = Sequential()

    # camadas escondidas
    aNN.add(Dense(
        units = neurons,
        activation = activation,
        kernel_initializer = kernel_initializer,
        input_dim = 4
    ))

    aNN.add ( Dropout ( dropout ) )

    aNN.add(Dense(
        units = neurons,
        kernel_initializer = kernel_initializer,
        activation = activation
    ))

    aNN.add ( Dropout ( dropout ) )

    # camada de saída
    aNN.add(Dense(
        units = 3,
        activation = "softmax"
    ))

    aNN.compile(optimizer = optimizer,
                loss = "sparse_categorical_crossentropy",
                metrics = ["accuracy"])

    #plot_model ( aNN, to_file = "aNN.png" )

    return aNN

def main (  ) :

    df = pd.read_csv("iris.csv")

    dadosEntrada = df.iloc[:, 0:4].values
    dadosSaida = df.iloc[:, 4].values

    labelEnconder = LabelEncoder()

    dadosSaida = labelEnconder.fit_transform ( dadosSaida )

    # dado convertido para classificação correta do objeto
    dadosSaidaDummy = np_utils.to_categorical(dadosSaida)

    dadosEntradaTreinamento, dadosEntradaTeste, dadosSaidaTreinamento, dadosSaidaTeste = train_test_split(
        dadosEntrada,
        dadosSaidaDummy,
        test_size = 0.25
    )

    #aNN = montaRedeNeural()

    #aNN.fit ( dadosEntradaTreinamento, dadosSaidaTreinamento,
    #          batch_size = 10,
    #         epochs = 1000 )

    #resultado = aNN.evaluate ( dadosEntradaTeste, dadosSaidaTeste )
    #print("precisão da rede : {} % ".format(resultado))

    #previsao = aNN.predict ( dadosEntradaTeste )
    #previsao = ( previsao > 0.9 )
    #print("matriz de previsão boleana : {}".format(previsao))

    # retorna qual tipo de planta é a classificação
    #previsaoSaida = [np.argmax(i) for i in dadosSaidaTeste ]
    #previsaoEntrada = [np.argmax ( i ) for i in previsao ]

    #confusionMatriz = confusion_matrix ( previsaoEntrada, previsaoSaida )
    #print("matriz de confusão : {}".format(confusionMatriz))

    #print("Montando a rede neural utilizando o keras :")
    redeComKeras ( dadosEntrada, dadosSaida )

if __name__ == '__main__':
    main()