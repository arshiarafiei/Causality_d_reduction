import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse




def autoencoder(X,Y,drop_out,dim,lr):
    df = pd.read_csv('data_org.csv')

    Y = df['target']
    X = df.iloc[:, :-1]





    input_dim = X.shape[1]
    encoding_dim = dim
    drop_out_rate = drop_out

    encoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='sigmoid'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(64, activation='relu'),
        layers.Dropout(drop_out_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(32, activation='relu'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(16, activation='relu'),
        layers.Dropout(drop_out_rate),
        layers.Dense(encoding_dim, activation='linear')
    ])


    decoder = models.Sequential([
        layers.Dense(16, activation='linear'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(32, activation='sigmoid'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(64, activation='relu'),
        layers.Dropout(drop_out_rate),
        layers.Dense(64, activation='relu'),
        layers.Dropout(drop_out_rate), 
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_out_rate),  
        layers.Dense(input_dim, activation='linear')
    ])



    autoencoder = models.Sequential([encoder, decoder])



    learning_rate = lr
    autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    autoencoder.fit(X, X, epochs=1000)

    X_encoded = encoder.predict(X)



    reduced_feature_names = [f'reduced_feature_{i}' for i in range(encoding_dim)]
    reduced_df = pd.DataFrame(X_encoded, columns=reduced_feature_names)
    reduced_df['target'] = Y

    reduced_df.round(1).to_csv('data_red.csv',index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--init_dim", help="Number of classes", default=4)
    parser.add_argument("--init_drop_out", help="Drop out", default=0.05)
    parser.add_argument("--init_lr", help="Lr", default=0.0001)


    args = parser.parse_args()

    dim = int(args.init_dim)
    lr = float(args.init_lr)
    drop_out = float(args.init_drop_out)


    df = pd.read_csv('data_org.csv')

    Y = df['target']
    X = df.iloc[:, :-1]


    autoencoder(X,Y,drop_out,dim,lr)