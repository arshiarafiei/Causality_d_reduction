import pandas as pd
import itertools



def SMT(df, max_differing_features):
    
    for index, row in df.iterrows():
        
        for num_differing in range(1, max_differing_features + 1):
            
            for cols in itertools.combinations([c for c in df.columns if c != 'target'], num_differing):
                
                masks = {c: (df[c] != row[c]) if c in cols else (df[c] == row[c]) for c in df.columns if c != 'target'}
                
               
                combined_mask = pd.Series(True, index=df.index)
                for c, m in masks.items():
                    combined_mask &= m
                
                
                df_check = df[combined_mask & (df['target'] == False)]
                
                
                if not df_check.empty:
                    differing_cols = ', '.join(cols)
                    return True ,"Cause is {} in row {}".format(differing_cols,index)
    return False, None


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    Sat, cause = SMT(df, 3)
    print(Sat,cause)