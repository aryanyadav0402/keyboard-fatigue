import pandas as pd, sys
pd.set_option('display.width',280); pd.set_option('display.max_columns',50)
for f in sys.argv[1:]: print(f); print(pd.read_csv(f).to_string()); print()
