#!/usr/bin/env python3
"""
Neural network classifier for 2EL1730.

"""
# NB: The runtime of this program is ~6 minutes on my machine.
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate
import re

train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)

# Various preprocessing functions
timezone_re = re.compile(r"[+-][0-9]{4}")
def parse_timezone(s):
    """
    Extract a time zone from the date string and convert it to a number.
    e.g.: s == '-0530' to '-5.5'
    """
    try:
        new_str = timezone_re.search(s).group()
        if len(new_str) != 5:
            return None # Couldn't find a valid time zone in the string
        hour = int(new_str[:3])
        minutes = int(new_str[3:])
        return hour + (minutes/60) * (1 if new_str[0] == '+' else -1)
    except AttributeError:
        return None

def standardise_str(x):
    """
    Remove leading and trailing whitespace and replace missing string
    data with a placeholder.
    """
    # Some of the string values include leading or trailing whitespace,
    # and some also have inconsistent capitalisation (particularly for
    # mail_type).
    # Also, some data is missing - such data needs to be replaced by
    # some placeholder string value, otherwise OneHotEncoder will raise
    # an exception and refuse to process it.
    return x.strip().lower() if type(x) is str else 'unknown'

def preprocess_domain(org, tld):
    """
    Extract the correct organisation and TLD from the org and tld data,
    where multi-level domains such as *.ac.in, *.co.uk are not handled
    correctly.
    Returns the new values of org and tld, and a boolean variable for
    whether there was a subdomain (the exact subdomain is ignored).
    """
    # E.g. an email from m.mail.coursera.org is described in the source
    # data as org=m and tld=mail.coursera.org, but a more logical
    # division would be org=coursera and tld=org.
    if not (type(org) is str and type(tld) is str): # Missing data
        return [standardise_str(org), standardise_str(tld), False]
    # Split the domain into elements
    elements = (org.strip() + '.' + tld.strip()).lower().split('.')
    # Finding out which element of the domain actually represents the
    # organisation name is non-trivial. E.g. for a domain such as
    # *.wordpress.com, the * is the name of the website, but for
    # mail.coursera.org, the name of the website is coursera.
    # In web browsers (or at least Firefox?), this is done using a list
    # which is regularly updated. Without this list, the best guess is
    # to take the *longest element* of the domain (in any case, it is
    # likely to be the most informative for ML purposes).
    new_org = ""
    index = -1
    for (k, v) in enumerate(elements):
        if len(v) > len(new_org):
            new_org = v
            index = k
    # Now take everything after new_org to be the "TLD". Strictly
    # speaking, this is not a TLD since it may take values like "ac.in",
    # but it has the same semantic meaning (e.g. we usually think of
    # "co.uk", "fr", "ac.uk", "org" and "com.mx" as the same sort of
    # thing even though not all of them are TLDs).
    new_tld = '.'.join(elements[index+1:])
    # The construction of this pd.Series object is supposedly quite
    # expensive and is probably to blame for how long the pre-processing
    # takes.
    return pd.Series([new_org, new_tld, index > 0], index=['org', 'tld', 'has_subdomain'])

def preprocess(df):
    "Preprocess a data frame."
    # Preprocess date
    df["date_parsed"] = pd.to_datetime(df["date"], format='%d %b %Y %H:%M:%S', exact=False, errors='coerce')
    df["timezone"] = df["date"].apply(parse_timezone)
    df["weekday"] = df["date_parsed"].apply(lambda x: x.weekday()/6)
    df["month"] = df["date_parsed"].apply(lambda x: (x.month-1)/12)
    df["day"] = df["date_parsed"].apply(lambda x: (x.day-1)/31)
    df["hour"] = df["date_parsed"].apply(lambda x: ((x.hour + x.minute/60)%24)/24)
    # Remove zeroes in mail_type
    df["mail_type"] = df["mail_type"].apply(standardise_str)
    # Re-process org and tld to be more informative
    # Removal of zeroes (equivalent to standardise_str) is included in
    # this step.
    df[["org", "tld", "has_subdomain"]] = df.apply(lambda x: preprocess_domain(x["org"], x["tld"]), axis=1)
    # Scale certain columns
    df["chars_in_subject"] = df["chars_in_subject"]/100
    df["chars_in_body"] = df["chars_in_body"]/200000
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Select only the columns which will be used for training
filter_columns_x = ['timezone', 'weekday', 'month', 'day', 'hour',
                    'ccs', 'bcced', 'mail_type', 'images', 'urls',
                    'salutations', 'designation', 'chars_in_subject',
                    'chars_in_body', 'tld', 'org', 'has_subdomain']
classes = ['updates', 'personal', 'promotions', 'forums', 'purchases',
           'travel', 'spam', 'social']
train_x = train_df[filter_columns_x]
train_x = train_x.fillna(value=0)
train_y = train_df[classes]

test_x = test_df[filter_columns_x]
test_x = test_x.fillna(value=0)
print(train_x)

# Encode categorical features.
# NB: It's not OK to directly run OneHotEncoder on train_x and test_x
# because it will encode even numeric values! Therefore, this extra
# step with ColumnTransformer is needed.
ohe_list = [(x, OneHotEncoder(), [x]) for x in ['mail_type', 'tld', 'org']]
transformer = ColumnTransformer(transformers=ohe_list, remainder='passthrough')
# Can't use np.vstack here as in skeleton_code_ml.py
transformer.fit(pd.concat([train_x, test_x], ignore_index=True))
train_x_featurised = transformer.transform(train_x)
test_x_featurised = transformer.transform(test_x)
print("Featurised dimensions:", train_x_featurised.shape)

# Reduce dimensionality.
# Removing this step provides a very marginal increase in accuracy.
# n_components=50 is much faster and still usable.
svd = TruncatedSVD(n_components=128)
svd.fit(train_x_featurised)
train_x_svd = svd.transform(train_x_featurised)
test_x_svd = svd.transform(test_x_featurised)
# print(svd.explained_variance_ratio_)

# Do not use solver='lbfgs' as it terminates far too soon.
# 'adam' and 'sgd' both seem to work.
classif = MLPClassifier(solver='adam', activation='relu', max_iter=250, verbose=True)

# Get cross-validation score which is taken to be the training error.
# If you do not need the training error, comment this out for the
# program to run 6 times faster...
cv_results = cross_validate(classif, train_x_svd, train_y, cv=5)
print("*******CV SCORE*******", cv_results["test_score"].mean())

# Now do the final fit and make the predictions.
classif.set_params(warm_start=True) # Reuse parameters from CV to save time
classif.fit(train_x_svd, train_y)
pred_y = classif.predict_proba(test_x_svd)
pred_df = pd.DataFrame(pred_y, columns=classes)
pred_df.to_csv("ML_TEAM12_nn_submission.csv", index=True, index_label='Id')
