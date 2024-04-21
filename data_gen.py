import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import argparse



def data_gen(n_classes,n_samples,n_features, per ):
    weights = [1-per, per]
    X, y = make_classification(n_samples=n_samples,
                            n_features=n_features,
                            n_informative=n_features,
                            n_redundant=0,
                            n_clusters_per_class=1,
                            weights=weights,
                            flip_y=0,  
                            class_sep=n_classes,  
                            hypercube=False,  
                            shift=0,  
                            scale=1,  
                            )

    
    scaler = MinMaxScaler(feature_range=(1, 5))
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.round(X_scaled).astype(int)  

    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    y_df = pd.DataFrame(y, columns=['target'])

    
    df = pd.concat([X_df, y_df], axis=1)
    df.to_csv('data_org.csv',index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument("--init_n_class", help="Number of classes", default=2)
    parser.add_argument("--init_n_samples", help="Number of samples", default=5000)
    parser.add_argument("--init_n_features", help="Number of features", default=128)
    parser.add_argument("--init_per", help="Number of percentage", default=0.2)

    args = parser.parse_args()

    n_classes = int(args.init_n_class)
    n_samples = int(args.init_n_samples) * n_classes
    n_features = int(args.init_n_features)
    per = float(args.init_per)

    data_gen(n_classes, n_samples, n_features, per)
    