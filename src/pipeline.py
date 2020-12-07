import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

cols = ['QUESTID2', 'AGE2', 'BOOKED', 'MOVSINPYR2', 'PDEN10', 'COUTYP4',
        'YESTSCIG', 'YESTSMJ', 'YESTSALC', 'YESTSDNK', 'YEYFGTSW', 'YEYFGTGP',
        'YESCHACT', 'YECOMACT', 'YEFAIACT', 'YEOTHACT', 'YEATNDYR', 'YESCHFLT',
        'YESCHWRK', 'YESCHIMP', 'YESCHINT', 'YETCGJOB', 'YELSTGRD', 'YEPCHKHW',
        'YEPHLPHW', 'YEPCHORE', 'YEPLMTTV', 'YEPLMTSN', 'YEPGDJOB', 'YEPPROUD',
        'YEYARGUP', 'YEPPKCIG', 'YEPMJEVR', 'YEPMJMO', 'YEPALDLY', 'YEPRTDNG',
        'YEGPKCIG', 'YEGMJEVR', 'YEGMJMO', 'YEGALDLY', 'YEFPKCIG', 'YEFMJEVR',
        'YEFMJMO', 'YEFALDLY', 'YERLGSVC', 'YERLGIMP', 'YERLDCSN', 'YERLFRND',
        'YODPREV', 'YODSCEV', 'YOLOSEV', 'YOWRHRS', 'YOWRDST', 'YOWRCHR',
        'YOWRIMP', 'HEALTH', 'DIFFTHINK', 'YEPRBSLV', 'YEVIOPRV', 'YEDGPRGP',
        'YEDECLAS', 'YEDERGLR', 'YEDESPCL', 'YEPVNTYR', 'CIGEVER', 'SMKLSSEVR',
        'CIGAREVR', 'PIPEVER', 'ALCEVER', 'MJEVER', 'COCEVER', 'CRKEVER',
        'HEREVER', 'HALLUCEVR', 'INHALEVER', 'METHAMEVR', 'PNRANYLIF',
        'TRQANYLIF', 'STMANYLIF', 'SEDANYLIF', 'PNRNMLIF', 'COLDMEDS']

path = r'../data/NSDUH_2019.SAV'

def youth_df():
    """Return survey w/ ages 12-17 & relevant features as DataFrame"""

    df = pd.read_spss(path, usecols = cols)
    df = df.loc[df['AGE2'].isin([1, 2, 3, 4, 5, 6])]
    df.reset_index(inplace=True)
    return df

def recoder(df):
    """Formats data for imputing.

    Converts all non-informative entries (e.g. skips or refusals to
    answer) to np.nan values.  Maps the 1: Yes 2: No scale to normal
    binary mapping of {0, 1}.  Sets rating schema polarity. Constructs
    target columns.  Drops columns with diminished value after recoding.

    Parameters
    ----------
    df : DataFrame
        youth_df()
    Returns
    -------
    df
    """

    # Properly code in NaN values
    df.replace(to_replace = [85, 94, 97, 98, 99, 985, 994, 997, 998, 999],
        value = np.nan, inplace=True)

    # Recode Environemntal Indicators
    df.BOOKED = df.BOOKED.map({1:1, 2:0, 3: 1})
    df.PDEN10 = df.PDEN10.map({1:2, 2: 1, 3: 0})
    df.COUTYP4 = df.COUTYP4.map({1:2, 2: 1, 3: 0})

    # Recode Peer/Social Indicators
    df.YESTSCIG = df.YESTSCIG.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YESTSMJ = df.YESTSMJ.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YESTSALC = df.YESTSALC.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YESTSDNK = df.YESTSDNK.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YEYFGTSW = df.YEYFGTSW.map({1: 0, 2: 1, 3: 3, 4: 6, 5: 10})
    df.YEYFGTGP = df.YEYFGTGP.map({1: 0, 2: 1, 3: 3, 4: 6, 5: 10})

    # Recode School Indicators
    df.YEATNDYR = df.YEATNDYR.map({1: 1, 2: 0})
    df.YESCHFLT = df.YESCHFLT.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YESCHWRK = df.YESCHWRK.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YESCHIMP = df.YESCHIMP.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YESCHINT = df.YESCHINT.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YETCGJOB = df.YETCGJOB.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YELSTGRD = df.YELSTGRD.map({1: 3, 2: 2, 3: 1, 4: 0, 5: 5})

    # Recode Parental Indicators
    df.YEPCHKHW = df.YEPCHKHW.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPHLPHW = df.YEPHLPHW.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPCHORE = df.YEPCHORE.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPLMTTV = df.YEPLMTTV.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPLMTSN = df.YEPLMTSN.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPGDJOB = df.YEPGDJOB.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEPPROUD = df.YEPPROUD.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YEYARGUP = df.YEYARGUP.map({1: 0, 2: 1, 3: 3, 4: 6, 5: 10})
    df.YEPPKCIG = df.YEPPKCIG.map({1: 2, 2: 1, 3: 0})
    df.YEPMJEVR = df.YEPMJEVR.map({1: 2, 2: 1, 3: 0})
    df.YEPMJMO = df.YEPMJMO.map({1: 2, 2: 1, 3: 0})
    df.YEPALDLY = df.YEPALDLY.map({1: 2, 2: 1, 3: 0})
    df.YEPRTDNG = df.YEPRTDNG.map({1: 1, 2: 0})

    # Recode Belief Indicators
    df.YEGPKCIG = df.YEGPKCIG.map({1: 2, 2: 1, 3: 0})
    df.YEGMJEVR = df.YEGMJEVR.map({1: 2, 2: 1, 3: 0})
    df.YEGMJMO = df.YEGMJMO.map({1: 2, 2: 1, 3: 0})
    df.YEGALDLY = df.YEGALDLY.map({1: 2, 2: 1, 3: 0})
    df.YEFPKCIG = df.YEFPKCIG.map({1: 2, 2: 1, 3: 0})
    df.YEFMJEVR = df.YEFMJEVR.map({1: 2, 2: 1, 3: 0})
    df.YEFMJMO = df.YEFMJMO.map({1: 2, 2: 1, 3: 0})
    df.YEFALDLY = df.YEFALDLY.map({1: 2, 2: 1, 3: 0})
    df.YERLGSVC = df.YERLGSVC.map({1: 0, 2: 1, 3: 3, 4: 6, 5: 25, 6: 53})
    df.YERLGIMP = df.YERLGIMP.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YERLDCSN = df.YERLDCSN.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YERLFRND = df.YERLFRND.map({1: 0, 2: 1, 3: 2, 4: 3})
    df.YODPREV = df.YODPREV.map({1: 1, 2: 0})
    df.YODSCEV = df.YODSCEV.map({1: 1 , 2: 0})
    df.YOLOSEV = df.YOLOSEV.map({1: 1 , 2: 0})
    df.YOWRHRS = df.YOWRHRS.map({1: 1, 2: 2, 3: 3, 4: 0})
    df.YOWRCHR = df.YOWRCHR.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.YOWRIMP = df.YOWRIMP.map({1: 3, 2: 2, 3: 1, 4: 0})
    df.HEALTH = df.HEALTH.map({1: 4, 2: 3, 3: 2, 4: 1, 5: 0})
    df.DIFFTHINK = df.DIFFTHINK.map({1: 0, 2: 0})

    # Recode Substance Specific Education Indicators
    df.YEPRBSLV = df.YEPRBSLV.map({1: 1, 2: 0})
    df.YEVIOPRV = df.YEVIOPRV.map({1: 1, 2: 0})
    df.YEDGPRGP = df.YEDGPRGP.map({1: 1, 2: 0})
    df.YEDECLAS = df.YEDECLAS.map({1: 1, 2: 0})
    df.YEDERGLR = df.YEDERGLR.map({1: 1, 2: 0})
    df.YEDESPCL = df.YEDESPCL.map({1: 1, 2: 0})
    df.YEPVNTYR = df.YEPVNTYR.map({1: 1, 2: 0})

    # Build Targets
    df.loc[ (df.CIGEVER == 1) | (df.SMKLSSEVR == 1) | (df.CIGAREVR == 1) |
        (df.PIPEVER == 1), 'TOBACCO'] = 1
    df.loc[ (df.CIGEVER != 1) & (df.SMKLSSEVR != 1) & (df.CIGAREVR != 1) &
        (df.PIPEVER != 1), 'TOBACCO'] = 0
    df.loc[df.ALCEVER == 1, 'ALCOHOL'] = 1
    df.loc[df.ALCEVER != 1, 'ALCOHOL'] = 0
    df.loc[df.MJEVER == 1, 'CANNABIS'] = 1
    df.loc[df.MJEVER != 1, 'CANNABIS'] = 0
    df.loc[ (df.COCEVER == 1) | (df.CRKEVER == 1) | (df.HEREVER == 1) |
        (df.HALLUCEVR == 1) | (df.INHALEVER == 1) | (df.METHAMEVR == 1) |
        (df.PNRANYLIF == 1) | (df.TRQANYLIF == 1) | (df.STMANYLIF == 1) |
        (df.SEDANYLIF == 1) | (df.PNRNMLIF == 1) | (df.COLDMEDS == 1),
        'FURTHER'] = 1
    df.loc[ (df.COCEVER != 1) & (df.CRKEVER != 1) & (df.HEREVER != 1) &
        (df.HALLUCEVR != 1) & (df.INHALEVER != 1) & (df.METHAMEVR != 1) &
        (df.PNRANYLIF != 1) & (df.TRQANYLIF != 1) & (df.STMANYLIF != 1) &
        (df.SEDANYLIF != 1) & (df.PNRNMLIF != 1) & (df.COLDMEDS != 1),
        'FURTHER'] = 0

    # Drop diminished columns
    df.drop(columns = ['index', 'CIGEVER', 'SMKLSSEVR', 'CIGAREVR', 'PIPEVER',
        'ALCEVER', 'MJEVER', 'COCEVER', 'CRKEVER', 'HEREVER', 'HALLUCEVR',
        'INHALEVER', 'METHAMEVR', 'PNRANYLIF', 'TRQANYLIF', 'STMANYLIF',
        'SEDANYLIF', 'PNRNMLIF', 'COLDMEDS'], inplace = True)
    return df

def imputer(df, drop=True):
    """Logically imputes select NaN values; can drop those remaining.
   
    Logic explained case-by-case with comments throughout. In general,
    imputation done covers data loss from the nature of census style
    collection. Imputation gives following improvment in data:

        Without Imputing -- Total NaNs: 75,486
                         -- Rows Lost if Dropped: 13,319 of 13,397

        With Imputing    -- Total NaNs: 16,775
                         -- Rows Lost if Dropped: 3,206 of 13,397

    Parameters
    ----------
    df : DataFrame
        recoded youth_df()

    drop : Boolean
        Default = True. If true return DataFrame with droped NaNs
    Returns
    -------
    df
    """

    # Impute Peer/Social Indicators 
    # - Nan : mean() for those not in school
    df.loc[df.YEATNDYR == 0, ['YESTSCIG']] = \
        df.loc[df.YEATNDYR == 0, ['YESTSCIG']].fillna(df.YESTSCIG.mean())
    df.loc[df.YEATNDYR == 0, ['YESTSMJ']] = \
        df.loc[df.YEATNDYR == 0, ['YESTSMJ']].fillna(df.YESTSMJ.mean())
    df.loc[df.YEATNDYR == 0, ['YESTSALC']] = \
        df.loc[df.YEATNDYR == 0, ['YESTSALC']].fillna(df.YESTSALC.mean())
    df.loc[df.YEATNDYR == 0, ['YESTSDNK']] = \
        df.loc[df.YEATNDYR == 0, ['YESTSDNK']].fillna(df.YESTSDNK.mean())

    # Impute School Indicators
    # - NaN : mean() for those with alt grade systems
    # - NaN : 0 for positive school features for those not in school
    is_graded = (df.YELSTGRD == 0) | (df.YELSTGRD == 1) | (df.YELSTGRD == 2) \
                | (df.YELSTGRD == 3)
    df.YELSTGRD = df.YELSTGRD.map({0: 0, 1: 1, 2: 2, 3: 3, 
                                   5: df[is_graded].YELSTGRD.mean()})
    df.loc[df.YEATNDYR == 0, ['YESCHFLT']] = df.loc[df.YEATNDYR == 0,
                                                    ['YESCHFLT']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YESCHWRK']] = df.loc[df.YEATNDYR == 0,
                                                    ['YESCHWRK']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YESCHIMP']] = df.loc[df.YEATNDYR == 0,
                                                    ['YESCHIMP']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YESCHINT']] = df.loc[df.YEATNDYR == 0,
                                                    ['YESCHINT']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YETCGJOB']] = df.loc[df.YEATNDYR == 0,
                                                    ['YETCGJOB']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YELSTGRD']] = df.loc[df.YEATNDYR == 0,
                                                    ['YELSTGRD']].fillna(0)
 
    # Impute Parental Indicators
    # - NaN: 0 for parent involvement w/ school for those not in school
    # - NaN : mean() for those not in school (i.e w/o school nights)
    df.loc[df.YEATNDYR == 0, ['YEPCHKHW']] = df.loc[df.YEATNDYR == 0,
                                                    ['YEPCHKHW']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YEPHLPHW']] = df.loc[df.YEATNDYR == 0,
                                                    ['YEPHLPHW']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YEPLMTSN']] = \
        df.loc[df.YEATNDYR == 0, ['YEPLMTSN']].fillna(df.YEPLMTSN.mean())

    # Impute Belief Indicators
    # - NaN : 1 on reworded questions for depression where interviewee
    #   has already stated they've been depressed
    # - NaN : 0 on questions gauging depression since skipped ~= 0
    df.loc[df.YODPREV == 1, ['YODSCEV']] = df.loc[df.YODPREV == 1,
                                                  ['YODSCEV']].fillna(1)
    df.loc[df.YODPREV == 1, ['YOLOSEV']] = df.loc[df.YODPREV == 1,
                                                  ['YOLOSEV']].fillna(1)
    df.YOWRHRS = df.YOWRHRS.fillna(0)
    df.YOWRDST = df.YOWRDST.fillna(0)
    df.YOWRCHR = df.YOWRCHR.fillna(0)
    df.YOWRIMP = df.YOWRIMP.fillna(0)
    df.DIFFTHINK = df.DIFFTHINK.fillna(0)

    # Impute Substance Specific Education Indicators
    # - NaN : 0 for in school if they have not attended school
    df.loc[df.YEATNDYR == 0, ['YEDECLAS']] = df.loc[df.YEATNDYR == 0,
                                                    ['YEDECLAS']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YEDERGLR']] = df.loc[df.YEATNDYR == 0,
                                                    ['YEDERGLR']].fillna(0)
    df.loc[df.YEATNDYR == 0, ['YEDESPCL']] = df.loc[df.YEATNDYR == 0,
                                                    ['YEDESPCL']].fillna(0)
                                        
    if drop: df = df.dropna()
    return df

def build_vars(df, col):
    """Helper function to reserve test set for given target."""

    X = df.drop(columns=['TOBACCO', 'CANNABIS', 'ALCOHOL', 'FURTHER'])
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.2,
                                                        random_state=1618)
    return X_train, X_test, y_train, y_test
