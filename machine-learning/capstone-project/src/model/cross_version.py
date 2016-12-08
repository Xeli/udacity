import sklearn

if sklearn.__version__ >= '0.18.1':
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split
